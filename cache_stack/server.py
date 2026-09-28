"""
OpenAI-compatible /v1/chat/completions endpoint with SSE streaming.

Architecture:
  HTTP handler → asyncio.Queue (per request) → background engine loop → token stream

The engine loop runs as a background asyncio task. HTTP handlers enqueue
requests and await a per-request token queue. This decouples HTTP from
the engine so the engine can batch across concurrent requests.
"""

import asyncio
import json
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BLOCK_SIZE, MAX_BLOCKS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM
from .sequence_manager import SequenceManager
from .kv_cache_manager import KVCacheManager
from patchedQwen import PagedKVCache

# ---------------------------------------------------------------------------
# Models and engine globals (initialised on startup)
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI()

_model = None
_tokenizer = None
_seq_manager = None
_kv_manager = None
_device = None
_request_queue: asyncio.Queue = None


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0 or temperature == 1.0:
        return logits.argmax().item()
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def _generate_tokens(prompt: str, max_tokens: int, temperature: float):
    """Synchronous generator yielding (token_str, is_last)."""
    seq_id = str(uuid.uuid4())
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    prompt_len = inputs.input_ids.shape[1]

    _seq_manager.init_sequence(seq_id)

    try:
        with torch.no_grad():
            # Prefill
            cache = PagedKVCache(_seq_manager, _kv_manager, seq_id, prompt_len)
            out = _model(input_ids=inputs.input_ids, past_key_values=cache, use_cache=True)

        token_id = _sample(out.logits[0, -1, :], temperature)
        token_str = _tokenizer.decode([token_id], skip_special_tokens=True)
        yield token_str, False

        # Decode
        for _ in range(max_tokens - 1):
            with torch.no_grad():
                cache = PagedKVCache(_seq_manager, _kv_manager, seq_id, 1)
                out = _model(
                    input_ids=torch.tensor([[token_id]], device=_device),
                    past_key_values=cache,
                    use_cache=True,
                )
            token_id = _sample(out.logits[0, -1, :], temperature)
            if token_id == _tokenizer.eos_token_id:
                yield "", True
                return
            token_str = _tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_str, False

        yield "", True
    finally:
        _seq_manager.free_sequence(seq_id)


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------

def _sse_chunk(delta: str, finish_reason=None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": delta},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global _model, _tokenizer, _seq_manager, _kv_manager, _device, _request_queue

    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    dtype = torch.float16 if _device != "cpu" else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(_device)
    _model.eval()

    _seq_manager = SequenceManager(MAX_BLOCKS, BLOCK_SIZE)
    _kv_manager = KVCacheManager(MAX_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                  NUM_LAYERS, _device, dtype)
    _request_queue = asyncio.Queue()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)

    if request.stream:
        async def event_stream():
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(
                None,
                lambda: list(_generate_tokens(prompt, request.max_tokens, request.temperature))
            )
            for token_str, is_last in gen:
                finish_reason = "stop" if is_last else None
                yield _sse_chunk(token_str, finish_reason)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    tokens = list(_generate_tokens(prompt, request.max_tokens, request.temperature))
    content = "".join(t for t, _ in tokens)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
