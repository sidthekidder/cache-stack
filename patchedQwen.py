import torch
from transformers import Cache


class PagedKVCache(Cache):
    """
    HuggingFace Cache interface backed by paged block storage.

    Slots are pre-allocated in __init__ so all 24 layers write to the same
    physical (block_id, slot_id) positions during a single forward pass.
    """

    def __init__(self, sequence_manager, kv_cache_manager, sequence_id, num_new_tokens):
        super().__init__()
        self.sm = sequence_manager
        self.kv = kv_cache_manager
        self.seq_id = sequence_id

        # Capture prior token count before allocating slots for this step.
        # HuggingFace uses get_seq_length() to offset position IDs, so this
        # must reflect tokens already in cache (not including the new ones).
        self.prior_tokens = sequence_manager.token_counts[sequence_id]

        # Pre-allocate one slot per new token. All 24 layer update() calls
        # reuse these same (block_id, slot_id) pairs.
        self.new_slots = []
        for _ in range(num_new_tokens):
            block_id, slot_id = sequence_manager.get_next_slot(sequence_id)
            self.new_slots.append((block_id, slot_id))

        self.total_tokens = sequence_manager.token_counts[sequence_id]

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Write new K/V to block cache, then gather and return the full
        accumulated K/V for this sequence at this layer.

        key_states / value_states: (batch=1, num_kv_heads, seq_len, head_dim)
        Returns: same shapes but seq_len = total_tokens so far.
        """
        num_kv_heads = key_states.shape[1]
        head_dim = key_states.shape[3]

        # Write each new token into its pre-allocated slot.
        for t, (block_id, slot_id) in enumerate(self.new_slots):
            self.kv.write_cache(layer_idx, 0, block_id, slot_id, key_states[0, :, t, :])
            self.kv.write_cache(layer_idx, 1, block_id, slot_id, value_states[0, :, t, :])

        # Gather all blocks owned by this sequence for this layer.
        block_table = self.sm.block_table[self.seq_id]
        # Shape: (2, n_blocks, block_size, num_kv_heads, head_dim)
        gathered = self.kv.kv_cache[layer_idx, :, block_table, :, :, :]

        block_size = self.sm.block_size
        n_blocks = len(block_table)

        # Reshape to (num_kv_heads, total_padded_tokens, head_dim), then trim.
        k = gathered[0].reshape(n_blocks * block_size, num_kv_heads, head_dim)
        k = k.permute(1, 0, 2)[:, :self.total_tokens, :]

        v = gathered[1].reshape(n_blocks * block_size, num_kv_heads, head_dim)
        v = v.permute(1, 0, 2)[:, :self.total_tokens, :]

        # Add batch dimension: (1, num_kv_heads, total_tokens, head_dim)
        return k.unsqueeze(0), v.unsqueeze(0)

    def get_seq_length(self, layer_idx=0):
        return self.prior_tokens

    def get_max_length(self):
        return None
