from .block_allocator import BlockAllocator

MAX_BLOCKS = 10


class SequenceManager:
	def __init__(self):
		self.block_table = {}
		self.block_allocator = BlockAllocator(MAX_BLOCKS)

	def init_sequence(self, sequenceID):
		# make sure sequence doesn't already exist
		assert sequenceID not in self.block_table

		# allocate a new block and initialize the dict
		new_block_id = self.block_allocator.allocate()
		self.block_table[sequenceID] = [new_block_id]

	def extend_sequence(self, sequenceID):
		# make sure sequence already exists before extending
		assert sequenceID in self.block_table

		# allocate a new block and append
		new_block_id = self.block_allocator.allocate()		
		self.block_table[sequenceID].append(new_block_id)

	def free_sequence(self, sequenceID):
		for blockId in self.block_table[sequenceID]:
			self.block_allocator.free(blockId)
		del self.block_table[sequenceID]

