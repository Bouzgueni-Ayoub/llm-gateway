import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from config import (
    GENERATION_PATHS,
    GatewaySettings,
    ModelConfig,
    TokenizerManager,
    USAGE_STATUS_FAILED,
    USAGE_STATUS_RECONCILIATION_PENDING,
    extract_bearer_token,
    extract_possible_dict,
    mask_key,
    normalize_text,
    parse_json_dict,
    parse_optional_positive_int,
    safe_json_dumps,
)
from db import GatewayDB
from metrics import GatewayMetrics


LOGGER = logging.getLogger("llm_gateway")


# `GatewayRequestContext` is the handoff object between admission and proxying.
# `prepare_gateway_request` fills it after auth, model lookup, quota reservation,
# and payload rewriting. The response handlers then use this one object instead
# of re-reading headers or repeating database lookups.
@dataclass
class GatewayRequestContext:
    company_mapping: dict[str, Any]
    user_context: dict[str, str | None]
    request_id: str
    logical_model_id: str
    model_config: ModelConfig
    estimated_prompt_tokens: int
    reserved_completion_tokens: int
    reserved_total_tokens: int
    path: str
    is_stream: bool
    upstream_base_url: str
    body: bytes
    started_at: datetime


# vLLM streams OpenAI responses as Server-Sent Events, a text protocol where
# each event is usually sent as a `data: ...` line. `StreamUsageCapture` watches
# the bytes as they pass through and remembers the latest `usage` object so the
# gateway can settle billing after the stream ends.
class StreamUsageCapture:

    def __init__(self) -> None:
        # Network chunks do not always line up with line endings. `_buffer`
        # stores partial text until a full newline-delimited SSE line arrives.
        self._buffer = ""
        self.last_usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        # `httpx` gives raw bytes. The parser decodes to text, appends it to any
        # previous partial line, and only processes complete lines.
        self._buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line.strip())

    def finish(self) -> dict[str, Any] | None:
        # The final chunk may not end with a newline. Process any leftover text
        # before returning the usage block that settlement code should trust.
        if self._buffer:
            self._process_line(self._buffer.strip())
            self._buffer = ""
        return self.last_usage

    def _process_line(self, line: str) -> None:
        # Ignore non-data lines and the `[DONE]` marker. Valid JSON events with
        # a top-level `usage` dict update `last_usage`; malformed events are
        # ignored so streaming to the client is not interrupted by parser noise.
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            parsed = json.loads(payload)
        except Exception:
            return
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if isinstance(usage, dict):
            self.last_usage = usage


def extract_request_user_context(request: Request, payload: dict[str, Any]) -> dict[str, str | None]:
    # Clients can send user identity in the JSON body or in forwarded HTTP
    # headers. The returned dict is stored on usage events so a request can be
    # audited back to a tenant user without making billing depend on the client.
    user_info = extract_possible_dict(payload.get("user"))
    if not user_info:
        metadata = extract_possible_dict(payload.get("metadata"))
        if metadata:
            user_info = extract_possible_dict(metadata.get("user"))

    request_user_id = None
    username = None
    email = None
    if user_info:
        request_user_id = normalize_text(user_info.get("id"))
        username = normalize_text(user_info.get("username") or user_info.get("name"))
        email = normalize_text(user_info.get("email"))

    request_user_id = request_user_id or normalize_text(request.headers.get("x-gateway-user-id") or request.headers.get("x-openwebui-user-id"))
    username = username or normalize_text(request.headers.get("x-gateway-user-name") or request.headers.get("x-openwebui-user-name"))
    email = email or normalize_text(request.headers.get("x-gateway-user-email") or request.headers.get("x-openwebui-user-email"))

    request_user_identity = None
    if username and email:
        request_user_identity = f"{username.lower()}::{email.lower()}"
    elif request_user_id:
        request_user_identity = f"id::{request_user_id}"
    elif email:
        request_user_identity = f"email::{email.lower()}"
    elif username:
        request_user_identity = f"username::{username.lower()}"

    return {
        "request_user_identity": request_user_identity,
        "request_user_id": request_user_id,
        "request_username": username,
        "request_user_email": email,
    }


def require_admin(request: Request) -> None:
    # Admin routes share this guard. If no admin token is configured, the local
    # dev API stays open; otherwise callers must provide the token in the custom
    # header or as a bearer token.
    token = request.app.state.settings.gateway_admin_token
    if not token:
        return
    provided = normalize_text(request.headers.get("x-gateway-admin-token"))
    if not provided:
        provided = extract_bearer_token(request.headers.get("authorization"))
    if provided != token:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")


async def authenticate_company_for_request(app: FastAPI, request: Request) -> dict[str, Any]:
    gateway_key = extract_bearer_token(request.headers.get("authorization"))
    if not gateway_key:
        raise HTTPException(status_code=401, detail="Missing gateway key in Authorization header")
    company_mapping = await run_db(app, app.state.db.get_company_mapping_by_openwebui_key, gateway_key)
    if not company_mapping:
        raise HTTPException(status_code=403, detail="No active company mapping for the provided gateway key")
    company_mapping["company_id"] = company_mapping.get("company_id") or company_mapping["company_name"]
    return company_mapping


def filtered_forward_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    # Some HTTP headers only describe the current network hop. The gateway drops
    # them so FastAPI/httpx can set correct values for the new downstream response.
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in {"content-length", "transfer-encoding", "connection"}
    }


def build_upstream_url(upstream_base_url: str, path: str) -> str:
    base = upstream_base_url.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        return f"{base}{path[3:]}"
    return f"{base}{path}"


def upstream_request_error_code(exc: httpx.RequestError) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "upstream_timeout"
    if isinstance(exc, httpx.ConnectError):
        return "upstream_connection_failed"
    return "upstream_request_failed"


def upstream_request_error_status(exc: httpx.RequestError) -> int:
    if isinstance(exc, httpx.TimeoutException):
        return 504
    return 503


def upstream_request_error_response(context: GatewayRequestContext, exc: httpx.RequestError) -> JSONResponse:
    code = upstream_request_error_code(exc)
    message = f"Upstream model runtime is unavailable for logical_model_id '{context.logical_model_id}'"
    if code == "upstream_timeout":
        message = f"Upstream model runtime timed out for logical_model_id '{context.logical_model_id}'"
    return JSONResponse(
        status_code=upstream_request_error_status(exc),
        content={
            "error": {
                "message": message,
                "type": "upstream_unavailable",
                "code": code,
                "request_id": context.request_id,
            }
        },
    )


def log_upstream_request_error(context: GatewayRequestContext, exc: httpx.RequestError) -> None:
    LOGGER.warning(
        "Upstream request failed before response: request_id=%s company_id=%s logical_model_id=%s upstream_base_url=%s error=%s: %s",
        context.request_id,
        context.company_mapping["company_id"],
        context.logical_model_id,
        context.upstream_base_url,
        type(exc).__name__,
        exc,
    )


def extract_usage_from_json_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    # Non-stream OpenAI-compatible responses report final billing numbers in a
    # top-level `usage` object. Missing usage means settlement must be deferred.
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return usage if isinstance(usage, dict) else None


def usage_totals(usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    # Normalize vLLM usage fields into integers. If total is omitted but prompt
    # and completion counts exist, the gateway can derive total safely.
    if not usage:
        return None, None, None
    prompt = parse_usage_token_int(usage.get("prompt_tokens"))
    completion = parse_usage_token_int(usage.get("completion_tokens"))
    total = parse_usage_token_int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, total


def parse_usage_token_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed < 0:
        return None
    return parsed


def latency_ms(started_at: datetime) -> int:
    # Metrics and usage rows store request duration in milliseconds.
    return int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)


def set_gateway_completion_cap(payload: dict[str, Any], path: str, effective_completion_cap: int) -> None:
    # The client may ask for a larger generation limit than policy or wallet
    # balance allows. This rewrites the outbound request so vLLM sees only the
    # gateway-approved completion cap.
    if path not in GENERATION_PATHS:
        return
    payload["max_tokens"] = effective_completion_cap
    if "max_completion_tokens" in payload or path == "/v1/chat/completions":
        payload["max_completion_tokens"] = effective_completion_cap


def requested_completion_cap(payload: dict[str, Any], path: str) -> int | None:
    if path == "/v1/chat/completions":
        return parse_optional_positive_int(payload.get("max_completion_tokens")) or parse_optional_positive_int(payload.get("max_tokens"))
    if path == "/v1/completions":
        return parse_optional_positive_int(payload.get("max_tokens"))
    return None


async def run_db(app: FastAPI, func, *args, **kwargs):
    # GatewayDB uses synchronous drivers. Running those calls in a bounded executor
    # keeps slow Postgres work from blocking the asyncio event loop.
    if app.state.db.backend == "sqlite":
        return func(*args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(app.state.db_executor, partial(func, *args, **kwargs))


async def finalize_known_usage(app: FastAPI, context: GatewayRequestContext, usage: dict[str, Any]) -> None:
    # This is the normal billing closeout path. `context` tells us which wallet
    # reservation exists, and `usage` is the final vLLM-reported token count.
    # The database first records usage, then moves reserved tokens according to
    # actual usage, then marks the request reconciled.
    prompt_tokens, completion_tokens, total_tokens = usage_totals(usage)
    if total_tokens is None:
        total_tokens = 0
    final_status = await run_db(
        app,
        app.state.db.settle_known_usage,
        context.company_mapping["company_id"],
        context.request_id,
        context.reserved_total_tokens,
        actual_prompt_tokens=prompt_tokens,
        actual_completion_tokens=completion_tokens,
        actual_total_tokens=total_tokens,
        latency_ms=latency_ms(context.started_at),
        metadata={
            "logical_model_id": context.logical_model_id,
            "backend_model": context.model_config.backend_model,
            "usage": usage,
        },
    )
    app.state.metrics.observe_request(
        context.company_mapping["company_id"],
        context.logical_model_id,
        context.path,
        context.is_stream,
        final_status,
        latency_ms(context.started_at),
    )


async def mark_reconciliation_pending(app: FastAPI, context: GatewayRequestContext) -> None:
    # Use this when upstream likely produced billable work but did not return
    # usable final token counts. The wallet reservation remains visible for a
    # later reconciliation process instead of guessing a charge.
    await run_db(
        app,
        app.state.db.update_usage_event,
        context.request_id,
        latency_ms=latency_ms(context.started_at),
        status=USAGE_STATUS_RECONCILIATION_PENDING,
    )
    app.state.metrics.observe_request(
        context.company_mapping["company_id"],
        context.logical_model_id,
        context.path,
        context.is_stream,
        USAGE_STATUS_RECONCILIATION_PENDING,
        latency_ms(context.started_at),
    )


async def release_without_usage(app: FastAPI, context: GatewayRequestContext, status: str, reason: str) -> None:
    # Use this only when the gateway believes no billable inference completed.
    # It returns all reserved tokens to the wallet and marks the usage event with
    # the supplied failure/cancel status.
    await run_db(
        app,
        app.state.db.release_request_without_usage,
        context.company_mapping["company_id"],
        context.request_id,
        context.reserved_total_tokens,
        latency_ms=latency_ms(context.started_at),
        status=status,
        metadata={"reason": reason, "logical_model_id": context.logical_model_id},
    )
    app.state.metrics.observe_request(
        context.company_mapping["company_id"],
        context.logical_model_id,
        context.path,
        context.is_stream,
        status,
        latency_ms(context.started_at),
    )


async def prepare_gateway_request(app: FastAPI, request: Request, path: str) -> GatewayRequestContext:
    # Admission is the gate before any model runtime is called. It authenticates
    # the company, loads the assigned model route from Postgres, estimates prompt
    # cost, reserves wallet balance, rewrites the payload, and returns a
    # `GatewayRequestContext` for the proxy phase.
    db: GatewayDB = app.state.db
    tokenizers: TokenizerManager = app.state.tokenizers
    started_at = datetime.now(timezone.utc)

    # The gateway rewrites JSON fields before forwarding, so it needs a parsed
    # object in addition to the raw request.
    gateway_key = extract_bearer_token(request.headers.get("authorization"))
    if not gateway_key:
        raise HTTPException(status_code=401, detail="Missing gateway key in Authorization header")
    body = await request.body()
    payload = parse_json_dict(body)
    # Extract user metadata once and carry it on the context into audit rows.
    user_context = extract_request_user_context(request, payload)

    # Clients request a logical model id. The database lookup proves that this
    # company has that model assigned and returns the backend route to call.
    logical_model_id = normalize_text(payload.get("model"))
    if not logical_model_id:
        raise HTTPException(status_code=400, detail="Request is missing model")

    resolved = await run_db(app, db.resolve_request_model_for_estimation, gateway_key, logical_model_id)
    if not resolved:
        raise HTTPException(status_code=403, detail="Requested model is not assigned to this company")
    company_mapping, model_config = resolved
    if not model_config.route:
        raise HTTPException(status_code=503, detail="Assigned model has no route_target configured")

    # Reservation uses an estimate because final vLLM usage is only known after
    # inference. Metrics record which estimation method was used.
    prompt_estimate = await tokenizers.estimate_prompt_tokens(model_config, path, payload)
    estimated_prompt_tokens = prompt_estimate.tokens
    app.state.metrics.observe_estimation(company_mapping["company_id"], logical_model_id, prompt_estimate.method)

    # One request id ties together wallet ledger rows, usage events, gateway
    # metadata sent upstream, and later reconciliation.
    request_id = str(uuid.uuid4())
    admission = await run_db(
        app,
        db.admit_request,
        gateway_key=gateway_key,
        logical_model_id=logical_model_id,
        request_id=request_id,
        user_id=user_context.get("request_user_identity"),
        expected_backend_model=model_config.backend_model,
        expected_tokenizer_repo=model_config.tokenizer_repo,
        expected_tokenizer_revision=model_config.tokenizer_revision,
        estimated_prompt_tokens=estimated_prompt_tokens,
        estimation_method=prompt_estimate.method,
        requested_completion_cap=requested_completion_cap(payload, path),
        stream=bool(payload.get("stream")),
        path=path,
    )
    company_mapping = admission["company_mapping"]
    model_config = admission["model_config"]
    effective_completion_cap = admission["reserved_completion_tokens"]
    reserved_total_tokens = admission["reserved_total_tokens"]

    # Upstream runtimes know backend model names, not tenant-facing logical ids.
    payload["model"] = model_config.backend_model
    # Apply the admitted generation cap to the payload that vLLM will receive.
    set_gateway_completion_cap(payload, path, effective_completion_cap)
    if bool(payload.get("stream")):
        # Streaming responses need a final usage event for exact settlement.
        # This option asks vLLM-compatible runtimes to include it.
        stream_options = extract_possible_dict(payload.get("stream_options")) or {}
        stream_options["include_usage"] = True
        payload["stream_options"] = stream_options
    # Metadata travels to upstream logs without letting the client choose billing
    # identifiers. These values come from trusted gateway state.
    metadata = extract_possible_dict(payload.get("metadata")) or {}
    metadata["gateway_request_id"] = request_id
    metadata["logical_model_id"] = logical_model_id
    metadata["company_id"] = company_mapping["company_id"]
    payload["metadata"] = metadata
    # httpx sends bytes, so convert the rewritten Python dict back to JSON bytes.
    body = safe_json_dumps(payload).encode("utf-8")
    upstream_base_url = model_config.route.rstrip("/")

    # From here on, proxy code should use this context instead of trusting the
    # original request body or headers.
    return GatewayRequestContext(
        company_mapping=company_mapping,
        user_context=user_context,
        request_id=request_id,
        logical_model_id=logical_model_id,
        model_config=model_config,
        estimated_prompt_tokens=estimated_prompt_tokens,
        reserved_completion_tokens=effective_completion_cap,
        reserved_total_tokens=reserved_total_tokens,
        path=path,
        is_stream=bool(payload.get("stream")),
        upstream_base_url=upstream_base_url,
        body=body,
        started_at=started_at,
    )


async def proxy_nonstream_response(app: FastAPI, context: GatewayRequestContext, upstream_response: httpx.Response) -> Response:
    # Non-stream responses can be fully read before returning to the client.
    # That makes usage extraction and wallet settlement straightforward.
    response_headers = filtered_forward_headers(upstream_response.headers)
    try:
        raw = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    # The response body is returned unchanged, but the parsed copy is inspected
    # for the billing `usage` block.
    parsed = parse_json_dict(raw)
    usage = extract_usage_from_json_payload(parsed)
    may_have_inferred = upstream_response.status_code < 400

    if usage:
        # Exact vLLM usage is available, so the wallet reservation can be settled.
        await finalize_known_usage(
            app,
            context,
            usage,
        )
    elif may_have_inferred:
        # A successful response without usage may have consumed tokens. Keep the
        # reservation pending rather than charging an invented number.
        await mark_reconciliation_pending(app, context)
    else:
        # A failed upstream response is treated as not billable here.
        await release_without_usage(app, context, USAGE_STATUS_FAILED, "upstream_failed_before_inference")

    return Response(
        content=raw,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


async def proxy_stream_response(app: FastAPI, context: GatewayRequestContext, upstream_response: httpx.Response) -> StreamingResponse:
    # Streaming requires two coordinated tasks: one task reads upstream chunks
    # and settles billing at the end; the response iterator yields those same
    # chunks to the client. The queue passes bytes between the two tasks.
    response_headers = filtered_forward_headers(upstream_response.headers)
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
    parser = StreamUsageCapture()

    async def pump() -> None:
        # The pump is the only code that consumes the upstream response. It also
        # performs final settlement because it sees the entire stream.
        saw_body = False
        error: Exception | None = None
        try:
            async for chunk in upstream_response.aiter_raw():
                saw_body = True
                # Parse a copy for usage while preserving the exact bytes for the client.
                parser.feed(chunk)
                await queue.put(chunk)
        except Exception as exc:
            error = exc
        finally:
            # Settlement happens after the upstream stream closes, when any
            # final usage event has had a chance to arrive.
            usage = parser.finish()
            await upstream_response.aclose()
            try:
                if usage:
                    await finalize_known_usage(app, context, usage)
                elif error is not None or upstream_response.status_code >= 400:
                    await release_without_usage(app, context, USAGE_STATUS_FAILED, "stream_failed_before_inference")
                elif saw_body:
                    await mark_reconciliation_pending(app, context)
                else:
                    await mark_reconciliation_pending(app, context)
            finally:
                # `None` is not sent to the client; it only tells `iter_body` to stop.
                await queue.put(None)

    # `create_task` starts upstream reading before FastAPI begins pulling from
    # the response iterator.
    task = asyncio.create_task(pump())

    async def iter_body() -> AsyncIterator[bytes]:
        # FastAPI pulls bytes from this async generator and writes them to the client.
        try:
            while True:
                # Waiting on the queue lets the pump and client response move at
                # different speeds without loading the whole stream into memory.
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Surface pump errors if they happened before the iterator finished.
            if task.done():
                task.result()

    return StreamingResponse(
        iter_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


async def proxy_to_upstream(app: FastAPI, request: Request, path: str) -> Response:
    # Chat, completion, and embedding routes all use this same path: admit the
    # request, build an upstream httpx request, then choose stream or non-stream handling.
    context = await prepare_gateway_request(app, request, path)
    # Forward ordinary headers, but never forward the tenant gateway key to vLLM.
    upstream_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "authorization", "content-length"}
    }
    upstream_headers["x-gateway-request-id"] = context.request_id

    # Build the request separately so the send call can choose streaming mode.
    upstream_request = app.state.client.build_request(
        method=request.method,
        url=build_upstream_url(context.upstream_base_url, path),
        headers=upstream_headers,
        content=context.body,
        params=request.query_params,
    )

    try:
        # In stream mode, httpx leaves the response body open for the streaming handler.
        upstream_response = await app.state.client.send(upstream_request, stream=context.is_stream)
    except httpx.RequestError as exc:
        # No upstream response means no reliable billable work, so release reservation
        # and return a controlled gateway error instead of surfacing an ASGI 500.
        reason = upstream_request_error_code(exc)
        await release_without_usage(app, context, USAGE_STATUS_FAILED, reason)
        log_upstream_request_error(context, exc)
        return upstream_request_error_response(context, exc)
    except Exception:
        # Preserve fail-closed reservation behavior for unexpected local errors.
        await release_without_usage(app, context, USAGE_STATUS_FAILED, "upstream_call_never_started")
        raise

    if context.is_stream:
        return await proxy_stream_response(app, context, upstream_response)
    return await proxy_nonstream_response(app, context, upstream_response)


def create_app(settings: GatewaySettings | None = None, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    # Building the app through a factory keeps production and tests on the same
    # wiring. Tests can inject settings or a fake `httpx.AsyncClient`; production
    # uses environment-derived settings and a real client.
    app_settings = settings or GatewaySettings.from_env()
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
    db = GatewayDB(app_settings.database_url)
    tokenizers = TokenizerManager(app_settings.hf_home)
    metrics = GatewayMetrics()
    db_executor = None if db.backend == "sqlite" else ThreadPoolExecutor(max_workers=8, thread_name_prefix="gateway-db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Code before `yield` runs at startup; code after `yield` runs at
        # shutdown. Startup initializes DB tables and warms tokenizers for active
        # registry rows so first requests do less work.
        await run_db(app, db.open)
        await run_db(app, db.init)
        active_models = [
            ModelConfig(
                logical_model_id=row["logical_model_id"],
                backend_model=row["backend_model"],
                tokenizer_repo=row["tokenizer_repo"],
                tokenizer_revision=row["tokenizer_revision"],
                route=row.get("route_target"),
                model_policy_cap=int(row["model_policy_cap"]),
                preload=False,
            )
            for row in await run_db(app, db.list_model_registry)
            if int(row.get("is_active") or 0) == 1
        ]
        await tokenizers.preload(active_models)
        yield
        # Close shared resources so sockets and worker threads do not leak.
        await client.aclose()
        tokenizers.close()
        await run_db(app, db.close)
        if db_executor is not None:
            db_executor.shutdown(wait=True)

    # `app.state` is FastAPI's shared object bag. Route handlers read database,
    # tokenizer, metrics, and client objects from here.
    app = FastAPI(lifespan=lifespan)
    app.state.settings = app_settings
    app.state.client = client
    app.state.db = db
    app.state.tokenizers = tokenizers
    app.state.metrics = metrics
    app.state.db_executor = db_executor

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Lightweight process liveness check; it does not verify model backends.
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        # Prometheus scrapes this endpoint. It should be protected by network
        # policy or ingress rules rather than per-request app auth.
        content = await run_db(app, app.state.metrics.render, db)
        return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/company-mappings")
    async def company_mappings(request: Request) -> list[dict[str, Any]]:
        # Show tenant mapping rows while masking the stored gateway keys.
        require_admin(request)
        rows = await run_db(app, db.list_company_mappings)
        for row in rows:
            row["gateway_api_key"] = mask_key(normalize_text(row.pop("openwebui_api_key", None)))
        return rows

    @app.post("/company-mappings/upsert")
    async def upsert_company_mapping(request: Request) -> dict[str, Any]:
        # Manual admin path for local/dev changes. Production control-plane sync
        # should normally be the writer of these rows.
        require_admin(request)
        payload = parse_json_dict(await request.body())
        openwebui_api_key = normalize_text(payload.get("gateway_api_key") or payload.get("openwebui_api_key"))
        company_name = normalize_text(payload.get("company_name"))
        company_id = normalize_text(payload.get("company_id")) or company_name
        team_id = normalize_text(payload.get("team_id"))
        litellm_url = normalize_text(payload.get("litellm_url"))
        default_max_tokens = parse_optional_positive_int(payload.get("default_max_tokens"))
        initial_balance_tokens = parse_optional_positive_int(payload.get("initial_balance_tokens")) or 0
        low_balance_threshold = parse_optional_positive_int(payload.get("low_balance_threshold")) or 0
        if not openwebui_api_key or not company_name or not company_id or not team_id:
            raise HTTPException(status_code=400, detail="gateway_api_key, company_name, company_id, and team_id are required")

        # Upsert changes the active tenant mapping used by authentication.
        await run_db(app, db.upsert_company_mapping, openwebui_api_key, company_id, company_name, team_id, litellm_url, default_max_tokens, 1)
        # Wallet rows are separate from tenant mappings; create one if needed.
        await run_db(app, db.ensure_wallet_account, company_id, low_balance_threshold)
        if initial_balance_tokens:
            # This admin endpoint can fund a wallet explicitly; sync code avoids
            # silently refilling existing wallets.
            await run_db(app, db.adjust_wallet_balance, company_id, initial_balance_tokens, "purchase", metadata={"source": "admin_upsert"})
        return {
            "status": "ok",
            "company_id": company_id,
            "company_name": company_name,
            "team_id": team_id,
            "litellm_url": litellm_url,
            "default_max_tokens": default_max_tokens,
        }

    @app.post("/company-mappings/deactivate")
    async def deactivate_company_mapping(request: Request) -> dict[str, str]:
        # Deactivation keeps history while preventing future authentication with the key.
        require_admin(request)
        payload = parse_json_dict(await request.body())
        openwebui_api_key = normalize_text(payload.get("gateway_api_key") or payload.get("openwebui_api_key"))
        if not openwebui_api_key:
            raise HTTPException(status_code=400, detail="gateway_api_key is required")
        await run_db(app, db.deactivate_company_mapping, openwebui_api_key)
        return {"status": "ok"}

    @app.get("/wallets")
    async def wallets(request: Request) -> list[dict[str, Any]]:
        # Return current available/reserved balances for operators.
        require_admin(request)
        return await run_db(app, db.list_wallets)

    @app.post("/wallets/adjust")
    async def wallet_adjust(request: Request) -> dict[str, Any]:
        # Append an operator-driven wallet movement. The database writes both
        # the wallet balance update and ledger row in one transaction.
        require_admin(request)
        payload = parse_json_dict(await request.body())
        company_id = normalize_text(payload.get("company_id"))
        delta_tokens = int(payload.get("delta_tokens") or 0)
        # `entry_type` labels the ledger row, for example purchase or adjustment.
        entry_type = normalize_text(payload.get("entry_type")) or "adjustment"
        if not company_id or delta_tokens == 0:
            raise HTTPException(status_code=400, detail="company_id and non-zero delta_tokens are required")
        await run_db(app, db.ensure_wallet_account, company_id)
        await run_db(app, db.adjust_wallet_balance, company_id, delta_tokens, entry_type, metadata={"source": "admin_adjust"})
        return {"status": "ok", "company_id": company_id, "delta_tokens": delta_tokens}

    @app.get("/usage-events")
    async def usage_events(request: Request) -> list[dict[str, Any]]:
        # Expose request audit rows, including pending reconciliation cases.
        require_admin(request)
        return await run_db(app, db.list_usage_events)

    @app.get("/model-registry")
    async def model_registry(request: Request) -> list[dict[str, Any]]:
        # Return the trusted model registry rows currently stored in Postgres.
        require_admin(request)
        return await run_db(app, db.list_model_registry)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        # OpenAI-compatible chat route; billing and routing are handled by the proxy path.
        return await proxy_to_upstream(app, request, "/v1/chat/completions")

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        # Embeddings use the same auth, assignment, wallet, and routing controls.
        return await proxy_to_upstream(app, request, "/v1/embeddings")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        # Legacy completions route for clients that do not use chat messages.
        return await proxy_to_upstream(app, request, "/v1/completions")

    @app.get("/models")
    async def models_compat(request: Request) -> JSONResponse:
        # Some clients use `/models` instead of `/v1/models`; both require a tenant key.
        company_mapping = await authenticate_company_for_request(app, request)
        content = await run_db(app, db.list_openai_models_for_company, company_mapping["company_id"])
        return JSONResponse(content=content)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        # Return only the logical models assigned to this authenticated company.
        company_mapping = await authenticate_company_for_request(app, request)
        content = await run_db(app, db.list_openai_models_for_company, company_mapping["company_id"])
        return JSONResponse(content=content)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        # Keep errors JSON-shaped for OpenAI-compatible clients. Existing dicts
        # pass through, JSON strings are decoded, and plain text becomes `detail`.
        if isinstance(exc.detail, (dict, list)):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        try:
            parsed = json.loads(exc.detail)
            return JSONResponse(status_code=exc.status_code, content=parsed)
        except Exception:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app
