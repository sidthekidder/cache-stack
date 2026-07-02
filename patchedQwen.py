import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from cache_stack.sequence_manager import SequenceManager
from cache_stack.kv_cache_manager import KVCacheManager

class kvPatchedQwen2Attention(Qwen2Attention):
	def __init__(self, config, layer_idx):
		self.sequence_manager = SequenceManager()
		self.kv_cache_manager = KVCacheManager(
			num_blocks,
			block_size,
			num_kv_heads,
			head_dim,
			num_layers,
			device,
			dtype
		)
		super().__init__(config, layer_idx)

	def forward(
		self,
		hidden_states,
		position_embeddings,
		attention_mask,
		past_key_values,
		**kwargs,
	):
		########## pre-existing code

		input_shape = hidden_states.shape[:-1]
		hidden_shape = (*input_shape, -1, self.head_dim)

		query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
		value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

		cos, sin = position_embeddings
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
		
		#################################
		########## patched code below

		# kv_cache_manager.write_cache()

		# kv_cache_manager.read_cache()


		########## patched code above, pre-existing code below
		#################################
		attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
			self.config._attn_implementation, eager_attention_forward
		)

		attn_output, attn_weights = attention_interface(
			self,
			query_states,
			key_states,
			value_states,
			attention_mask,
			dropout=0.0 if not self.training else self.attention_dropout,
			scaling=self.scaling,
			sliding_window=self.sliding_window,  # main diff with Llama
			**kwargs,
		)

		attn_output = attn_output.reshape(*input_shape, -1).contiguous()
		attn_output = self.o_proj(attn_output)
		return attn_output, attn_weights

