# cache-stack

A mini LLM inference engine built from scratch to internalize the serving stack — paged KV cache, continuous batching, prefix caching.

**Not** a production system. The goal is to be able to whiteboard the full request lifecycle from memory.

## Stack

- Model: Qwen2.5-0.5B-Instruct (HuggingFace)
- Device: Apple Silicon MPS, fp16
- API: OpenAI-compatible `/v1/chat/completions` (FastAPI + SSE)

## Architecture

```
HTTP handler → asyncio.Queue → Scheduler loop → PagedKVCache → token stream
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                  SequenceManager         KVCacheManager
                  (block tables,          (physical tensor
                   token counts,           num_layers × 2 ×
                   preemption)             num_blocks × block_size ×
                         │                num_kv_heads × head_dim)
                  BlockAllocator
                  (free list,
                   ref counts)
                         │
                  PrefixCache
                  (hash→block,
                   LRU eviction,
                   copy-on-write)
```

### Key design decisions

**Paged KV cache** — instead of one contiguous tensor per sequence, all sequences draw fixed-size blocks (16 tokens each) from a shared pool. A `SequenceManager` holds each sequence's block table (list of physical block IDs) and a token count. The `BlockAllocator` owns the free list. When a sequence fills its last block, `extend_sequence` allocates another.

**`PagedKVCache`** implements the HuggingFace `Cache` interface so it drops into the existing Qwen2 attention code with no kernel changes. Slots are pre-allocated in `__init__` (before the forward pass) so all 24 layers write to the same physical locations during a single step. The `update()` method writes new K/V vectors then gathers all blocks for the sequence and returns them for attention.

**Continuous batching** — the scheduler maintains a `waiting` queue and a `running` set. Each step: admit waiting requests (until OOM), run prefill for newly admitted ones, run decode for the rest, retire finished sequences. On OOM the youngest decode request is preempted (blocks freed, request re-queued).

**Prefix cache** — tokens hashed in cumulative `block_size`-aligned chunks. Matching blocks are shared across sequences via ref counts. Unreferenced blocks evicted LRU. Diverging sequences get copy-on-write: a new block is allocated and the original's ref count decremented.

## Usage

```bash
pip install -r requirements.txt

# Single prompt (paged cache)
python generate.py "Your prompt here"

# API server
uvicorn cache_stack.server:app --port 8000

# Point any OpenAI-compatible client at it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Benchmark (server must be running)
python benchmark.py --concurrency 1 8 32
python benchmark.py --concurrency 8 --prefix-heavy
python benchmark.py --compare --vllm-port 8001  # vs vLLM
```

## Benchmarks

_To be filled after running against real vLLM._

### Unique prompts workload

| Concurrency | TTFT (ms) cache-stack | TTFT (ms) vLLM | Throughput (tok/s) cache-stack | Throughput (tok/s) vLLM |
|:-----------:|:---------------------:|:--------------:|:------------------------------:|:-----------------------:|
| 1           | —                     | —              | —                              | —                       |
| 8           | —                     | —              | —                              | —                       |
| 32          | —                     | —              | —                              | —                       |

### Prefix-heavy workload (~100 token shared system prompt)

| Concurrency | Prefix hit rate | TTFT (ms) | Throughput (tok/s) |
|:-----------:|:---------------:|:---------:|:------------------:|
| 8           | —               | —         | —                  |
| 32          | —               | —         | —                  |

## File map

```
generate.py              naive paged-cache inference loop
benchmark.py             async load generator + stats
plot_results.py          matplotlib bar charts from benchmark JSON

cache_stack/
  config.py              model + cache constants (Qwen2.5-0.5B)
  block_allocator.py     free list (deque), allocate / free
  sequence_manager.py    per-sequence block tables + token counts
  kv_cache_manager.py    physical KV tensor, read/write by block
  prefix_cache.py        hash→block map, LRU eviction, copy-on-write
  scheduler.py           continuous batching, preemption
  server.py              FastAPI + SSE /v1/chat/completions
patchedQwen.py           PagedKVCache — HuggingFace Cache interface
```

## Roadmap

- [x] Naive inference loop (prefill + decode)
- [x] Paged KV cache (BlockAllocator, SequenceManager, KVCacheManager)
- [x] PagedKVCache — HuggingFace Cache interface
- [x] Continuous batching scheduler
- [x] Prefix caching (hash-based, LRU, copy-on-write)
- [x] FastAPI server + SSE streaming
- [x] Benchmark client + load generator
- [ ] Run benchmarks + fill results table
- [ ] Blog post write-up
