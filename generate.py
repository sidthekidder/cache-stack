import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from patchedQwen import kvPatchedAutoModelForCausalLM
import numpy as np


prompt = "The capital of France is "

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available():
	device = "mps"
else:
	device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
	model_name,
	dtype=torch.float16 if device != 'cpu' else torch.float32,
).to(device)

# now prefill step once
inputs = tokenizer(prompt, return_tensors="pt").to(device)
logits = model(**inputs, use_cache=True)


# argmax from logits to get the first generated token
generated_token = logits.logits[0, -1, :].argmax()

past_kv = logits.past_key_values


# now decode step N times (N = number of tokens we want to generate)
MAX_TOKENS = 20
generated_tokens_list = [generated_token.item()]
for i in range(MAX_TOKENS):
	out = model(input_ids=generated_token.reshape(1, 1).to(device), past_key_values=past_kv, use_cache=True)

	# get the new generated token
	generated_token = out.logits[0, -1, :].argmax()
	generated_tokens_list.append(generated_token.item())

	# update with the latest kv cache
	past_kv = out.past_key_values

print("Prefill / decode finished, here is the generated tokens:")
print(tokenizer.decode(generated_tokens_list, skip_special_tokens=True))

