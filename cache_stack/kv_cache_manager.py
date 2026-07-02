import torch


class KVCacheManager:
	def __init__(
		self,
		num_blocks,
		block_size,
		num_kv_heads,
		head_dim,
		num_layers,
		device,
		dtype
	):
		self.kv_cache = torch.zeros(
			num_layers, 2, num_blocks, block_size, num_kv_heads,
			head_dim, dtype=dtype, device=device
		)

	def write_cache(
		self,
		layer: int,
		keyOrValue: int, # 0 for key, 1 for value
		block_id: int,
		slot_id: int,
		vector):
		self.kv_cache[layer, keyOrValue, block_id, slot_id] = vector


	def read_cache(self, block_table):
		return self.kv_cache[:, :, block_table, :, :, :]





