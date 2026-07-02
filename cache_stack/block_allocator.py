from collections import deque

class BlockAllocator:
	def __init__(self, total_blocks):
		self.free_list = deque(range(total_blocks))

	# returns a free block ID
	def allocate(self):
		if len(self.free_list) == 0:
			raise RuntimeError("Out of blocks")
		return self.free_list.pop()

	# restores a block ID to the free list
	def free(self, blockId):
		self.free_list.append(blockId)