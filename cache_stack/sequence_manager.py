from .block_allocator import BlockAllocator
from .config import MAX_BLOCKS, BLOCK_SIZE


class SequenceManager:
    def __init__(self, max_blocks=MAX_BLOCKS, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.block_table = {}
        self.token_counts = {}
        self.block_allocator = BlockAllocator(max_blocks)

    def init_sequence(self, sequence_id):
        assert sequence_id not in self.block_table
        new_block_id = self.block_allocator.allocate()
        self.block_table[sequence_id] = [new_block_id]
        self.token_counts[sequence_id] = 0

    def extend_sequence(self, sequence_id):
        assert sequence_id in self.block_table
        new_block_id = self.block_allocator.allocate()
        self.block_table[sequence_id].append(new_block_id)

    def free_sequence(self, sequence_id):
        for block_id in self.block_table[sequence_id]:
            self.block_allocator.free(block_id)
        del self.block_table[sequence_id]
        del self.token_counts[sequence_id]

    def get_next_slot(self, sequence_id):
        num_tokens = self.token_counts[sequence_id]
        slot_id = num_tokens % self.block_size

        if slot_id == 0 and num_tokens > 0:
            self.extend_sequence(sequence_id)

        block_id = self.block_table[sequence_id][num_tokens // self.block_size]
        self.token_counts[sequence_id] += 1
        return (block_id, slot_id)
