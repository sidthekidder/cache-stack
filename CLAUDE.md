# cache-stack
Mini LLM inference engine, built from scratch to learn the serving stack.
Full plan: docs/plan.md

## Working agreement
I write these by hand (do NOT generate them for me): block allocator,
scheduler loop, prefix-cache logic. You handle plumbing: FastAPI, benchmark
client, plotting, tests-after-the-fact. When I ask about a concept, explain
it — don't just write the code.

## Environment
Qwen2.5-0.5B-Instruct, device=mps (Apple Silicon), torch fp16.