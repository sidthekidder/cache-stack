"""
Continuous batching scheduler.

Maintains a waiting queue and a running set. Each step():
  1. Admits waiting requests until blocks run out.
  2. Runs prefill for newly admitted requests (one at a time).
  3. Runs decode for all active requests (sequential; true batching needs
     custom attention kernels beyond this project's scope).
  4. Retires finished sequences and frees their blocks.
  5. Prints batch composition so you can watch it change each step.

Preemption: if a waiting request can't be admitted due to OOM, the youngest
running decode request is evicted (blocks freed, request re-queued).
"""

import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from .sequence_manager import SequenceManager
from .kv_cache_manager import KVCacheManager
from patchedQwen import PagedKVCache


@dataclass
class Request:
    seq_id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    generated_tokens: List[int] = field(default_factory=list)
    state: str = "waiting"   # waiting | prefilling | decoding | done
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None


class Scheduler:
    def __init__(self, model, tokenizer, sequence_manager: SequenceManager,
                 kv_manager: KVCacheManager, device: str, eos_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.sm = sequence_manager
        self.kv = kv_manager
        self.device = device
        self.eos_id = eos_token_id

        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []
        self._req_counter = 0

    def add_request(self, prompt: str, max_new_tokens: int = 50) -> str:
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        seq_id = f"req_{self._req_counter}"
        self._req_counter += 1
        self.waiting.append(Request(seq_id=seq_id, prompt_tokens=tokens,
                                    max_new_tokens=max_new_tokens))
        return seq_id

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self):
        self._admit_waiting()

        prefill_reqs = [r for r in self.running if r.state == "prefilling"]
        decode_reqs  = [r for r in self.running if r.state == "decoding"]

        print(f"  batch: {len(prefill_reqs)} prefill, {len(decode_reqs)} decode, "
              f"{len(self.waiting)} waiting")

        for req in prefill_reqs:
            self._run_prefill(req)

        for req in decode_reqs:
            self._run_decode(req)

        # Retire finished requests.
        still_running = []
        for req in self.running:
            if req.state == "done":
                self.sm.free_sequence(req.seq_id)
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running

    def run(self) -> List[Request]:
        step = 0
        while self.waiting or self.running:
            print(f"step {step}")
            self.step()
            step += 1
        return self.finished

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _admit_waiting(self):
        admitted = []
        for req in self.waiting:
            try:
                self.sm.init_sequence(req.seq_id)
                req.state = "prefilling"
                self.running.append(req)
                admitted.append(req)
            except RuntimeError:
                # Out of blocks — try to preempt the youngest decode request.
                if not self._preempt_youngest():
                    break
                try:
                    self.sm.init_sequence(req.seq_id)
                    req.state = "prefilling"
                    self.running.append(req)
                    admitted.append(req)
                except RuntimeError:
                    break
        for req in admitted:
            self.waiting.remove(req)

    def _preempt_youngest(self) -> bool:
        decode_reqs = [r for r in self.running if r.state == "decoding"]
        if not decode_reqs:
            return False
        youngest = max(decode_reqs, key=lambda r: r.arrival_time)
        self.sm.free_sequence(youngest.seq_id)
        youngest.state = "waiting"
        youngest.generated_tokens = []
        self.running.remove(youngest)
        self.waiting.insert(0, youngest)
        print(f"  preempted {youngest.seq_id}")
        return True

    def _run_prefill(self, req: Request):
        input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            cache = PagedKVCache(self.sm, self.kv, req.seq_id, len(req.prompt_tokens))
            out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        token = out.logits[0, -1, :].argmax().item()
        req.generated_tokens.append(token)
        req.first_token_time = time.perf_counter()
        req.state = "decoding"
        self._check_done(req)

    def _run_decode(self, req: Request):
        last_token = torch.tensor([[req.generated_tokens[-1]]], dtype=torch.long,
                                   device=self.device)
        with torch.no_grad():
            cache = PagedKVCache(self.sm, self.kv, req.seq_id, 1)
            out = self.model(input_ids=last_token, past_key_values=cache, use_cache=True)
        token = out.logits[0, -1, :].argmax().item()
        req.generated_tokens.append(token)
        self._check_done(req)

    def _check_done(self, req: Request):
        if (req.generated_tokens[-1] == self.eos_id or
                len(req.generated_tokens) >= req.max_new_tokens):
            req.state = "done"
