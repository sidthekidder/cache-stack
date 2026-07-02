import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
	torch_dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)

prompt = "Capital of France is "
inputs = tokenizer(prompt, return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))

