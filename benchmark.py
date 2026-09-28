"""
Benchmark client for cache-stack.

Fires N concurrent requests against the server and reports:
  - TTFT (time to first token)
  - TPOT (time per output token)
  - Throughput (tokens/sec)

Usage:
    # Start server first:
    uvicorn cache_stack.server:app --port 8000

    # Unique prompts:
    python benchmark.py --concurrency 1 8 32

    # Prefix-heavy workload (all share a 200-token system prompt):
    python benchmark.py --concurrency 8 --prefix-heavy

    # Compare mode: run against vLLM on port 8001:
    python benchmark.py --compare --vllm-port 8001
"""

import argparse
import asyncio
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

BASE_URL = "http://localhost:{port}/v1/chat/completions"
MAX_TOKENS = 50

SYSTEM_PROMPT = (
    "You are a helpful assistant. Always answer concisely. "
    "Provide accurate information. Be polite. Never refuse reasonable requests. " * 5
)  # ~100 tokens — used for prefix-heavy workload

PROMPTS = [
    "The capital of France is",
    "Explain how neural networks work in one sentence.",
    "What is the largest planet in the solar system?",
    "Write a haiku about autumn.",
    "What is 17 multiplied by 13?",
    "Name three programming languages invented after 2000.",
    "What causes thunder?",
    "Who wrote Pride and Prejudice?",
    "Describe the water cycle briefly.",
    "What is the speed of light?",
]


@dataclass
class RequestResult:
    prompt: str
    ttft: Optional[float] = None
    tokens: List[str] = field(default_factory=list)
    total_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def n_tokens(self):
        return len(self.tokens)

    @property
    def tpot(self):
        if self.total_time and self.n_tokens > 1:
            return self.total_time / self.n_tokens
        return None


async def send_request(session: aiohttp.ClientSession, prompt: str,
                        port: int, prefix_heavy: bool = False) -> RequestResult:
    result = RequestResult(prompt=prompt)
    messages = []
    if prefix_heavy:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    payload = {"messages": messages, "max_tokens": MAX_TOKENS, "stream": True}
    url = BASE_URL.format(port=port)
    start = time.perf_counter()

    try:
        async with session.post(url, json=payload) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    if result.ttft is None:
                        result.ttft = time.perf_counter() - start
                    result.tokens.append(delta)

        result.total_time = time.perf_counter() - start
    except Exception as e:
        result.error = str(e)

    return result


async def run_workload(concurrency: int, port: int,
                        prefix_heavy: bool = False) -> List[RequestResult]:
    prompts = []
    for i in range(concurrency):
        prompts.append(PROMPTS[i % len(PROMPTS)])

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, p, port, prefix_heavy) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)


def print_stats(results: List[RequestResult], label: str):
    ok = [r for r in results if r.error is None and r.ttft is not None]
    if not ok:
        print(f"  {label}: all requests failed")
        return

    ttfts = [r.ttft * 1000 for r in ok]
    tpots = [r.tpot * 1000 for r in ok if r.tpot]
    total_tokens = sum(r.n_tokens for r in ok)
    total_time = max(r.total_time for r in ok)
    throughput = total_tokens / total_time if total_time else 0

    print(f"\n  [{label}]")
    print(f"    requests:   {len(ok)}/{len(results)} succeeded")
    print(f"    TTFT (ms):  avg={statistics.mean(ttfts):.1f}  p50={statistics.median(ttfts):.1f}  "
          f"p95={sorted(ttfts)[int(0.95*len(ttfts))]:.1f}")
    if tpots:
        print(f"    TPOT (ms):  avg={statistics.mean(tpots):.1f}")
    print(f"    throughput: {throughput:.1f} tok/s")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefix-heavy", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Also benchmark vLLM on --vllm-port")
    parser.add_argument("--vllm-port", type=int, default=8001)
    args = parser.parse_args()

    workload = "prefix-heavy" if args.prefix_heavy else "unique-prompts"
    print(f"Benchmark: {workload}\n")

    for c in args.concurrency:
        print(f"=== concurrency={c} ===")
        results = await run_workload(c, args.port, args.prefix_heavy)
        print_stats(results, f"cache-stack port={args.port}")

        if args.compare:
            vllm_results = await run_workload(c, args.vllm_port, args.prefix_heavy)
            print_stats(vllm_results, f"vLLM port={args.vllm_port}")

    print("\nDone. Run with --compare to benchmark against vLLM.")


if __name__ == "__main__":
    asyncio.run(main())
