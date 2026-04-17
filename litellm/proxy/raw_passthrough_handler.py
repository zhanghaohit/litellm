"""
Raw byte passthrough for chat completions and Anthropic messages.

When a model is configured with `litellm_params.passthrough_raw: true`, the
proxy short-circuits the normal pipeline for BOTH streaming and non-streaming
requests:

    client → /v1/chat/completions or /v1/messages
           → auth → router pick deployment
           → aiohttp POST → raw bytes → client

Skipped for both modes:
    request transformation (OpenAI ↔ provider), response parse + re-serialize
    (pydantic round-trip), chunk_creator, CustomStreamWrapper, per-chunk
    async_post_call_streaming_{hook,iterator_hook}, stream_chunk_builder,
    callback fan-out.

Kept: auth (FastAPI dependency), pre_call_hook (budget / rate-limit /
     guardrails), model routing via router.async_get_available_deployment
     (load balancing + cooldowns), usage extraction for spend tracking.

Supported formats:
  - OpenAI chat completions (both streaming SSE and non-streaming JSON).
    `stream_options.include_usage: true` is injected on streaming requests
    so the terminal SSE chunk carries token counts.
  - Anthropic messages (both streaming SSE and non-streaming JSON).
    `message_start` carries input_tokens and `message_delta` carries
    output_tokens for streams; non-streaming JSON has the full `usage` field.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Tuple

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
    DeploymentAffinityCheck,
)

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router


# ─── session affinity ────────────────────────────────────────────────────────


async def _write_deployment_affinity(
    llm_router: "Router",
    data: dict,
    deployment: dict,
) -> None:
    """Write session/user-key affinity to cache after deployment selection.

    Replicates the fields that router._prepare_request injects into kwargs
    (model_info + metadata.deployment_model_name) so that
    DeploymentAffinityCheck.async_pre_call_deployment_hook can persist
    the session_id -> model_id mapping.
    """
    if not getattr(llm_router, "optional_callbacks", None):
        return

    affinity_cb: Optional[DeploymentAffinityCheck] = None
    for cb in llm_router.optional_callbacks:
        if isinstance(cb, DeploymentAffinityCheck):
            affinity_cb = cb
            break
    if affinity_cb is None:
        return

    model_info = deployment.get("model_info", {})
    deployment_model_name = deployment.get("model_name", "")

    data["model_info"] = model_info
    data.setdefault("metadata", {}).update(
        {
            "model_info": model_info,
            "deployment_model_name": deployment_model_name,
        }
    )

    try:
        await affinity_cb.async_pre_call_deployment_hook(data, None)
    except Exception as e:
        verbose_proxy_logger.debug(f"passthrough affinity write error: {e}")


# ─── flag lookup ─────────────────────────────────────────────────────────────


def _find_passthrough_model_cfg(
    model_name: str, llm_model_list: Optional[list]
) -> Optional[Dict[str, Any]]:
    """Return the model_list entry if `model_name` has `passthrough_raw: true`
    set in `litellm_params`."""
    if not llm_model_list or not model_name:
        return None
    for m in llm_model_list:
        if m.get("model_name") == model_name:
            lp = m.get("litellm_params") or {}
            if lp.get("passthrough_raw"):
                return m
            return None
    return None


# ─── URL + body + headers ────────────────────────────────────────────────────


def _build_upstream_url(api_base: str, endpoint: str) -> str:
    """endpoint: "chat_completions" | "completions" | "messages" """
    b = api_base.rstrip("/")
    if endpoint == "messages":
        if b.endswith("/v1/messages") or b.endswith("/messages"):
            return b
        if b.endswith("/v1"):
            return f"{b}/messages"
        return f"{b}/v1/messages"
    if endpoint == "completions":
        if b.endswith("/v1/completions") or b.endswith("/completions"):
            return b
        if b.endswith("/v1"):
            return f"{b}/completions"
        return f"{b}/v1/completions"
    # chat_completions (default)
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return f"{b}/chat/completions"
    return f"{b}/v1/chat/completions"


def _build_request(
    data: dict,
    litellm_params: dict,
    endpoint: str,
    is_stream: bool,
) -> Tuple[dict, dict]:
    """Strip litellm-internal fields from body and build upstream headers."""
    upstream_model = litellm_params.get("model") or data["model"]
    if "/" in upstream_model:
        upstream_model = upstream_model.split("/", 1)[1]

    body = dict(data)
    body["model"] = upstream_model
    body.pop("metadata", None)  # litellm internal
    body.pop("model_info", None)  # litellm internal

    if endpoint in ("chat_completions", "completions") and is_stream:
        # OpenAI chat/completions + legacy text completions: ask upstream
        # to include usage in the terminal SSE chunk.
        opts = dict(body.get("stream_options") or {})
        opts["include_usage"] = True
        body["stream_options"] = opts
    else:
        # Non-stream or Anthropic: no stream_options
        body.pop("stream_options", None)

    api_key = litellm_params.get("api_key") or "sk-none"
    headers = {"Content-Type": "application/json"}
    if endpoint == "messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = data.get("anthropic_version", "2023-06-01")
    headers["Authorization"] = f"Bearer {api_key}"

    return body, headers


# ─── usage extraction ────────────────────────────────────────────────────────


def _extract_usage_openai_stream(tail: bytes) -> Optional[Dict[str, int]]:
    if b"usage" not in tail:
        return None
    for line in reversed(tail.split(b"\n")):
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except Exception:
            continue
        u = ev.get("usage")
        if u:
            return {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(u.get("completion_tokens", 0) or 0),
                "total_tokens": int(u.get("total_tokens", 0) or 0),
            }
    return None


def _extract_usage_anthropic_stream(
    head: bytes, tail: bytes
) -> Optional[Dict[str, int]]:
    input_tokens = 0
    output_tokens = 0
    found = False
    if b"message_start" in head:
        for line in head.split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            if ev.get("type") == "message_start":
                u = (ev.get("message") or {}).get("usage") or {}
                input_tokens = int(u.get("input_tokens", 0) or 0)
                output_tokens = int(u.get("output_tokens", 0) or 0)
                found = True
                break
    if b"message_delta" in tail:
        for line in reversed(tail.split(b"\n")):
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            if ev.get("type") == "message_delta":
                u = ev.get("usage") or {}
                if "output_tokens" in u:
                    output_tokens = int(u.get("output_tokens", 0) or 0)
                    found = True
                    break
    if not found:
        return None
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _extract_usage_json(body: bytes, endpoint: str) -> Optional[Dict[str, int]]:
    """For non-streaming responses, parse the full JSON once and extract usage."""
    try:
        obj = json.loads(body)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    u = obj.get("usage") or {}
    if endpoint == "messages":
        # Anthropic non-stream: usage has input_tokens/output_tokens
        prompt_tokens = int(u.get("input_tokens", 0) or 0)
        completion_tokens = int(u.get("output_tokens", 0) or 0)
    else:
        prompt_tokens = int(u.get("prompt_tokens", 0) or 0)
        completion_tokens = int(u.get("completion_tokens", 0) or 0)
    if not (prompt_tokens or completion_tokens):
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ─── main entrypoint ─────────────────────────────────────────────────────────


async def raw_passthrough_request(
    data: dict,
    llm_router: "Router",
    user_api_key_dict: "UserAPIKeyAuth",
    proxy_logging_obj: "ProxyLogging",
    endpoint: str = "chat_completions",  # or "messages"
) -> Response:
    """Short-circuit handler. Returns StreamingResponse for stream=true
    requests, JSONResponse otherwise."""
    is_stream = bool(data.get("stream"))
    call_type = "completion" if endpoint == "chat_completions" else "anthropic_messages"

    # 1. Pre-call hook (auth already done by FastAPI dependency)
    try:
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=data,
            call_type=call_type,
        )
    except Exception as e:
        verbose_proxy_logger.exception(f"passthrough pre_call_hook error: {e}")
        raise

    # 2. Pick deployment
    deployment = await llm_router.async_get_available_deployment(
        model=data["model"],
        messages=data.get("messages"),
        request_kwargs=data,
    )

    # Persist session/user-key affinity for passthrough requests.
    await _write_deployment_affinity(llm_router, data, deployment)

    litellm_params = deployment["litellm_params"]
    api_base = litellm_params.get("api_base")
    if not api_base:
        raise RuntimeError(
            f"passthrough model {data['model']} has no api_base on deployment"
        )
    upstream_url = _build_upstream_url(api_base, endpoint)
    body, headers = _build_request(data, litellm_params, endpoint, is_stream)
    upstream_model = body["model"]

    start_ts = time.time()
    model_id = deployment.get("model_info", {}).get("id") if deployment else None

    # Shared timeout: no wall-clock limit, no per-read limit.
    timeout = aiohttp.ClientTimeout(
        total=None, sock_read=None, sock_connect=30, connect=30
    )

    if is_stream:
        return _streaming_response(
            upstream_url=upstream_url,
            body=body,
            headers=headers,
            endpoint=endpoint,
            timeout=timeout,
            start_ts=start_ts,
            model_id=model_id,
            upstream_model=upstream_model,
            data=data,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
        )
    return await _non_streaming_response(
        upstream_url=upstream_url,
        body=body,
        headers=headers,
        endpoint=endpoint,
        timeout=timeout,
        start_ts=start_ts,
        model_id=model_id,
        upstream_model=upstream_model,
        data=data,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
    )


# ─── streaming path ──────────────────────────────────────────────────────────


def _streaming_response(
    *,
    upstream_url,
    body,
    headers,
    endpoint,
    timeout,
    start_ts,
    model_id,
    upstream_model,
    data,
    user_api_key_dict,
    proxy_logging_obj,
) -> StreamingResponse:
    async def _stream() -> AsyncIterator[bytes]:
        head = b""
        tail = b""
        head_cap = 4096
        bytes_sent = 0
        usage: Optional[Dict[str, int]] = None
        error: Optional[str] = None

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(upstream_url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        err_body = (await resp.text())[:500]
                        error = f"upstream HTTP {resp.status}: {err_body}"
                        if endpoint == "messages":
                            yield (
                                "event: error\ndata: "
                                + json.dumps(
                                    {
                                        "type": "error",
                                        "error": {
                                            "type": "upstream_error",
                                            "message": err_body,
                                        },
                                    }
                                )
                                + "\n\n"
                            ).encode()
                        else:
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "error": {
                                            "message": err_body,
                                            "status": resp.status,
                                        }
                                    }
                                )
                                + "\n\ndata: [DONE]\n\n"
                            ).encode()
                        return
                    async for raw in resp.content.iter_any():
                        if not raw:
                            continue
                        yield raw
                        bytes_sent += len(raw)
                        if len(head) < head_cap:
                            head = (head + raw)[:head_cap]
                        tail = (tail + raw)[-8192:]
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            verbose_proxy_logger.warning(f"passthrough stream error: {error}")
            return
        finally:
            try:
                if endpoint == "messages":
                    usage = _extract_usage_anthropic_stream(head, tail)
                else:
                    # OpenAI SSE format: chat_completions and legacy completions
                    # both put usage in the terminal `data: {... "usage": ...}` chunk.
                    usage = _extract_usage_openai_stream(tail)
            except Exception:
                usage = None
            _fire_spend_task(
                proxy_logging_obj,
                user_api_key_dict,
                data,
                usage,
                model_id,
                upstream_model,
                time.time() - start_ts,
                bytes_sent,
                error,
            )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "x-litellm-passthrough-raw": "1",
            "x-litellm-passthrough-endpoint": endpoint,
            "x-litellm-model-id": model_id or "",
        },
    )


# ─── non-streaming path ──────────────────────────────────────────────────────


async def _non_streaming_response(
    *,
    upstream_url,
    body,
    headers,
    endpoint,
    timeout,
    start_ts,
    model_id,
    upstream_model,
    data,
    user_api_key_dict,
    proxy_logging_obj,
) -> Response:
    usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    response_body = b""
    status = 500
    resp_headers: Dict[str, str] = {}

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(upstream_url, json=body, headers=headers) as resp:
                status = resp.status
                response_body = await resp.read()
                # Only forward content-type; drop hop-by-hop / encoding headers
                ct = resp.headers.get("Content-Type", "application/json")
                resp_headers["Content-Type"] = ct
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        verbose_proxy_logger.warning(f"passthrough non-stream error: {error}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_error",
                    "message": error,
                }
            },
        )
    finally:
        if status == 200 and response_body:
            try:
                usage = _extract_usage_json(response_body, endpoint)
            except Exception:
                usage = None
        _fire_spend_task(
            proxy_logging_obj,
            user_api_key_dict,
            data,
            usage,
            model_id,
            upstream_model,
            time.time() - start_ts,
            len(response_body),
            error if error else (None if status == 200 else f"HTTP {status}"),
        )

    # Return upstream response bytes verbatim (no pydantic round-trip)
    return Response(
        content=response_body,
        status_code=status,
        media_type=resp_headers.get("Content-Type", "application/json"),
        headers={
            "x-litellm-passthrough-raw": "1",
            "x-litellm-passthrough-endpoint": endpoint,
            "x-litellm-model-id": model_id or "",
        },
    )


# ─── spend tracking ──────────────────────────────────────────────────────────


def _fire_spend_task(
    proxy_logging_obj,
    user_api_key_dict,
    data,
    usage,
    model_id,
    upstream_model,
    duration,
    bytes_sent,
    error,
):
    """Fire-and-forget spend tracking. Must never block or raise."""
    try:
        import asyncio

        asyncio.create_task(
            _record_spend(
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                data=data,
                usage=usage,
                model_id=model_id,
                upstream_model=upstream_model,
                duration=duration,
                bytes_sent=bytes_sent,
                error=error,
            )
        )
    except Exception as e:
        verbose_proxy_logger.debug(f"passthrough spend task error: {e}")


async def _record_spend(
    proxy_logging_obj: "ProxyLogging",
    user_api_key_dict: "UserAPIKeyAuth",
    data: dict,
    usage: Optional[Dict[str, int]],
    model_id: Optional[str],
    upstream_model: str,
    duration: float,
    bytes_sent: int,
    error: Optional[str],
) -> None:
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    try:
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        resp = ModelResponse(
            id=f"passthrough-{int(time.time()*1000)}",
            model=data.get("model", upstream_model),
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(role="assistant", content=""),
                )
            ],
            usage=Usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            ),
        )
        resp._hidden_params = {
            "model_id": model_id,
            "response_ms": duration * 1000,
            "litellm_call_id": data.get("litellm_call_id", ""),
            "raw_passthrough": True,
        }

        await proxy_logging_obj.post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=resp,
        )
        verbose_proxy_logger.debug(
            f"passthrough spend recorded: model={upstream_model} "
            f"prompt_tokens={usage['prompt_tokens']} "
            f"completion_tokens={usage['completion_tokens']} "
            f"dur={duration:.2f}s bytes={bytes_sent} error={error}"
        )
    except Exception as e:
        verbose_proxy_logger.debug(f"passthrough post_call_success_hook error: {e}")


