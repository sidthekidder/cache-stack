"""
Hash-based prefix cache with LRU eviction and copy-on-write.

Design:
  - Tokens are hashed in block_size chunks, cumulatively:
      hash(tokens[0:block_size]), hash(tokens[0:2*block_size]), ...
  - A cached block stores the K/V for one aligned chunk of tokens.
  - Multiple sequences can share a block (ref-counted).
  - Unreferenced blocks are evicted LRU when the allocator runs dry.
  - Copy-on-write: a sequence that needs to write to a shared block
    gets a fresh copy instead, leaving the shared block intact.

Integration with SequenceManager:
  - Call prefix_cache.lookup(prompt_tokens) before init_sequence to get
    already-computed block IDs.
  - Call sm.init_sequence_with_prefix(seq_id, cached_blocks, n_cached_tokens)
    to seed the block table with those blocks (incrementing their ref counts).
  - Only prefill the remaining (uncached) tokens.
  - After each newly computed block, call prefix_cache.insert(hash, block_id).
  - On sequence free, call prefix_cache.release(block_id) for each block
    that came from the cache.
"""

import hashlib
from collections import OrderedDict
from typing import List, Optional, Tuple

from .block_allocator import BlockAllocator


def _hash_tokens(tokens: Tuple[int, ...]) -> str:
    return hashlib.sha256(bytes(tokens)).hexdigest()


class PrefixCache:
    def __init__(self, block_allocator: BlockAllocator, block_size: int):
        self.allocator = block_allocator
        self.block_size = block_size
        self.hash_to_block: dict = {}           # hash -> block_id
        self.ref_counts: dict = {}              # block_id -> int
        self.lru: OrderedDict = OrderedDict()   # block_id -> True, LRU order

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, prompt_tokens: List[int]) -> Tuple[List[int], int]:
        """
        Find the longest prefix of prompt_tokens that is fully cached.

        Returns:
            (cached_block_ids, n_cached_tokens)
            where n_cached_tokens is a multiple of block_size.
        """
        matched_blocks = []
        n = 0
        while n + self.block_size <= len(prompt_tokens):
            chunk = tuple(prompt_tokens[: n + self.block_size])
            h = _hash_tokens(chunk)
            if h not in self.hash_to_block:
                break
            block_id = self.hash_to_block[h]
            matched_blocks.append(block_id)
            self._increment_ref(block_id)
            n += self.block_size

        return matched_blocks, n

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, prompt_tokens: List[int], block_idx: int, block_id: int):
        """
        Register a newly computed block in the cache.

        block_idx: which block of the prompt this is (0-based).
        block_id:  the physical block ID that was just filled.
        """
        end = (block_idx + 1) * self.block_size
        if end > len(prompt_tokens):
            return  # partial block — don't cache
        chunk = tuple(prompt_tokens[:end])
        h = _hash_tokens(chunk)
        if h not in self.hash_to_block:
            self.hash_to_block[h] = block_id
            self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1
            # Not in LRU because it's actively referenced.

    # ------------------------------------------------------------------
    # Reference counting
    # ------------------------------------------------------------------

    def release(self, block_id: int):
        """Decrement ref count; move to LRU pool if it hits zero."""
        if block_id not in self.ref_counts:
            return
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] <= 0:
            self.lru[block_id] = True

    def _increment_ref(self, block_id: int):
        self.ref_counts[block_id] = self.ref_counts.get(block_id, 0) + 1
        self.lru.pop(block_id, None)  # remove from eviction pool

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_lru(self) -> Optional[int]:
        """
        Evict the least-recently-used unreferenced block.
        Returns the freed block_id, or None if nothing to evict.
        """
        if not self.lru:
            return None
        block_id, _ = self.lru.popitem(last=False)
        # Remove all hash entries pointing at this block.
        evict_hashes = [h for h, b in self.hash_to_block.items() if b == block_id]
        for h in evict_hashes:
            del self.hash_to_block[h]
        self.ref_counts.pop(block_id, None)
        return block_id

    # ------------------------------------------------------------------
    # Copy-on-write
    # ------------------------------------------------------------------

    def is_shared(self, block_id: int) -> bool:
        return self.ref_counts.get(block_id, 0) > 1

    def cow_block(self, block_id: int) -> int:
        """
        If block_id is shared, allocate a fresh block and return it.
        The caller is responsible for copying K/V data.
        The original block's ref count is decremented.
        """
        if not self.is_shared(block_id):
            return block_id
        new_block_id = self.allocator.allocate()
        self.release(block_id)
        return new_block_id

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "cached_blocks": len(self.hash_to_block),
            "lru_evictable": len(self.lru),
            "total_refs": sum(self.ref_counts.values()),
        }
