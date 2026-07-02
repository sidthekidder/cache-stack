# cache-stack

A mini LLM inference engine built from scratch to internalize the serving stack — paged KV cache, continuous batching, prefix caching.

## Architecture

```
HTTP handler → async queue → scheduler loop → paged KV engine → token stream
```

Core components (hand-written):
- `BlockAllocator` — free list + ref counts, 16-token pages
- Scheduler — continuous batching, preemption on OOM
- Prefix cache — hash-based block reuse, LRU eviction, copy-on-write

## Roadmap

- [ ] Naive inference loop (prefill + decode)
- [ ] Paged KV cache
- [ ] Continuous batching scheduler
- [ ] Prefix caching
- [ ] FastAPI server + SSE streaming
- [ ] Benchmarks vs. vLLM
