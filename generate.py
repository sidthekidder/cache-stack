import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_stack.config import BLOCK_SIZE, MAX_BLOCKS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
from cache_stack.sequence_manager import SequenceManager
from cache_stack.kv_cache_manager import KVCacheManager
from patchedQwen import PagedKVCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 50
SEQ_ID = "req_0"

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
model.eval()

seq_manager = SequenceManager(MAX_BLOCKS, BLOCK_SIZE)
kv_manager = KVCacheManager(MAX_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, NUM_LAYERS, device, dtype)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_len = inputs.input_ids.shape[1]

seq_manager.init_sequence(SEQ_ID)

# Prefill: process entire prompt in one pass.
with torch.no_grad():
    cache = PagedKVCache(seq_manager, kv_manager, SEQ_ID, prompt_len)
    out = model(input_ids=inputs.input_ids, past_key_values=cache, use_cache=True)

generated_token = out.logits[0, -1, :].argmax()
generated_tokens = [generated_token.item()]

# Decode: feed one token at a time, reusing the paged cache.
with torch.no_grad():
    for _ in range(MAX_NEW_TOKENS - 1):
        cache = PagedKVCache(seq_manager, kv_manager, SEQ_ID, 1)
        out = model(
            input_ids=generated_token.reshape(1, 1),
            past_key_values=cache,
            use_cache=True,
        )
        generated_token = out.logits[0, -1, :].argmax()
        generated_tokens.append(generated_token.item())
        if generated_token.item() == tokenizer.eos_token_id:
            break

seq_manager.free_sequence(SEQ_ID)

print(prompt, end="")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
