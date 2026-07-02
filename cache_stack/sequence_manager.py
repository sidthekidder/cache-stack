from .block_allocator import BlockAllocator

MAX_BLOCKS = 10
BLOCK_SIZE = 5


class SequenceManager:
	def __init__(self):
		self.block_table = {}
		self.token_counts = {}
		self.block_allocator = BlockAllocator(MAX_BLOCKS)

	def init_sequence(self, sequenceID):
		# make sure sequence doesn't already exist
		assert sequenceID not in self.block_table

		# allocate a new block and initialize the dict
		new_block_id = self.block_allocator.allocate()
		self.block_table[sequenceID] = [new_block_id]

		# keep track of token count for this sequence id
		self.token_counts[sequenceID] = 0

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
		del self.token_counts[sequenceID]

	def get_next_slot(self, sequenceID):
		num_tokens = self.token_counts[sequenceID]

		# get the slot id - index within block
		slot_id = num_tokens % BLOCK_SIZE

		# if block is filled then new block needs to be allocated
		if slot_id == 0 and num_tokens > 0:
			self.extend_sequence(sequenceID)

		# get the block index
		block_id = self.block_table[sequenceID][num_tokens // BLOCK_SIZE]

		self.token_counts[sequenceID] += 1
		return (block_id, slot_id)




