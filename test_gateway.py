import asyncio
import json
from pathlib import Path

import httpx

from app_factory import create_app
from config import GatewaySettings, TokenizerManager
from control_plane_sync import load_desired_state, validated_tenants


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        size = 3 + len(messages)
        if tokenize:
            return list(range(size))
        return "rendered"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 2, 3]}


class BrokenChatTemplateTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        raise NameError("Extension is not defined")


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self):
        yield self.body


def build_settings(tmp_path: Path, initial_balance: int = 100) -> GatewaySettings:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    return GatewaySettings(
        database_url=f"sqlite:///{runtime_root / 'gateway.db'}",
        gateway_admin_token="admin-token",
        bootstrap_company_mappings_json="",
        model_registry_json="",
        preload_logical_models_json="[]",
        hf_home=str(runtime_root / "hf-cache"),
    )


def seed_control_plane(
    client,
    *,
    initial_balance: int = 100,
    enabled_models: list[str] | None = None,
    litellm_url: str | None = None,
    route_target: str | None = "http://vllm-tenant-demo-support-bot.test/v1",
) -> None:
    db = client.app.state.db
    db.upsert_company_mapping(
        openwebui_api_key="tenant-demo-key",
        company_id="tenant-demo",
        company_name="Tenant Demo",
        team_id="group-demo",
        litellm_url=litellm_url,
        default_max_tokens=1024,
        is_active=1,
    )
    db.upsert_model_registry_entry(
        logical_model_id="support-bot",
        backend_model="qwen2.5-1.5b",
        tokenizer_repo="repo/shared",
        tokenizer_revision="rev-1",
        model_policy_cap=10,
        is_active=1,
    )
    db.upsert_model_registry_entry(
        logical_model_id="support-bot-2",
        backend_model="qwen2.5-1.5b",
        tokenizer_repo="repo/shared",
        tokenizer_revision="rev-1",
        model_policy_cap=8,
        is_active=1,
    )
    for logical_model_id in enabled_models or ["support-bot"]:
        db.upsert_tenant_model_assignment("tenant-demo", logical_model_id, route_target=route_target, is_active=1)
    created = db.ensure_wallet_account("tenant-demo")
    if created and initial_balance > 0:
        db.adjust_wallet_balance(
            "tenant-demo",
            initial_balance,
            "purchase",
            metadata={"source": "test"},
            idempotency_key="test:wallet-seed:tenant-demo",
        )


def make_transport(
    captured_requests: list[dict],
    stream_body: bytes | None = None,
    captured_urls: list[str] | None = None,
    captured_authorizations: list[str | None] | None = None,
    chat_status_code: int = 200,
) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/key/generate":
            raise AssertionError("llm-gateway must not call key generation")

        if request.url.path == "/v1/chat/completions":
            if captured_urls is not None:
                captured_urls.append(str(request.url))
            if captured_authorizations is not None:
                captured_authorizations.append(request.headers.get("authorization"))
            payload = json.loads(request.content.decode("utf-8"))
            captured_requests.append(payload)
            if stream_body is not None:
                return httpx.Response(chat_status_code, stream=ByteStream(stream_body), headers={"content-type": "text/event-stream"})
            return httpx.Response(
                chat_status_code,
                json={
                    "id": "resp-1",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                },
            )

        raise AssertionError(f"Unexpected path {request.url.path}")

    return httpx.MockTransport(handler)


class GatewayHarness:
    def __init__(self, settings: GatewaySettings, transport: httpx.MockTransport) -> None:
        self.upstream_client = httpx.AsyncClient(transport=transport)
        self.app = create_app(settings=settings, http_client=self.upstream_client)
        self._lifespan = self.app.router.lifespan_context(self.app)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver")

    async def __aenter__(self):
        await self._lifespan.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()
        await self._lifespan.__aexit__(exc_type, exc, tb)

    async def post(self, *args, **kwargs) -> httpx.Response:
        return await self.client.post(*args, **kwargs)

    async def get(self, *args, **kwargs) -> httpx.Response:
        return await self.client.get(*args, **kwargs)


def build_client(settings: GatewaySettings, transport: httpx.MockTransport) -> GatewayHarness:
    TokenizerManager._load_tokenizer = lambda self, model_config: FakeTokenizer()  # type: ignore[method-assign]
    return GatewayHarness(settings, transport)


def run_with_client(settings: GatewaySettings, transport: httpx.MockTransport, scenario):
    async def runner():
        async with build_client(settings, transport) as client:
            return await scenario(client)

    return asyncio.run(runner())


def test_nonstream_request_rewrites_model_and_bills_actual_usage(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        forwarded = captured_requests[0]
        assert forwarded["model"] == "qwen2.5-1.5b"
        assert forwarded["max_tokens"] == 10
        assert forwarded["max_completion_tokens"] == 10

        wallet = client.app.state.db.get_wallet("tenant-demo")
        usage_events = client.app.state.db.list_usage_events()
        assert wallet["available_tokens"] == 93
        assert wallet["reserved_tokens"] == 0
        assert usage_events[0]["actual_total_tokens"] == 7
        assert usage_events[0]["estimation_method"] == "chat_template_tokens"
        assert usage_events[0]["status"] == "reconciled"

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_request_completion_cap_reduces_reservation_and_forwarded_cap(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
            },
            json={
                "model": "support-bot",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        forwarded = captured_requests[0]
        assert forwarded["max_tokens"] == 2
        assert forwarded["max_completion_tokens"] == 2
        usage_event = client.app.state.db.list_usage_events()[0]
        assert usage_event["reserved_completion_tokens"] == 2

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_unknown_model_rejected_before_upstream(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "not-allowed",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 403
        assert captured_requests == []
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert wallet["available_tokens"] == 100
        assert wallet["reserved_tokens"] == 0

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_unassigned_model_rejected_before_upstream(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100, enabled_models=["support-bot"])
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot-2",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 403
        assert captured_requests == []
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert wallet["available_tokens"] == 100
        assert wallet["reserved_tokens"] == 0

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_assigned_model_without_route_fails_closed_before_upstream(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100, route_target=None)
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tenant-demo-key"},
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 503
        assert response.json() == {"detail": "Assigned model has no route_target configured"}
        assert captured_requests == []
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert wallet["available_tokens"] == 100
        assert wallet["reserved_tokens"] == 0

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_upstream_connect_error_returns_503_and_releases_reservation(tmp_path: Path):
    captured_urls: list[str] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        raise httpx.ConnectError("All connection attempts failed", request=request)

    async def scenario(client):
        seed_control_plane(
            client,
            initial_balance=100,
            route_target="http://vllm-tenant-demo-mistral-7b.test/v1",
        )
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tenant-demo-key"},
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 503
        body = response.json()
        request_id = body["error"].pop("request_id")
        assert request_id
        assert body == {
            "error": {
                "message": "Upstream model runtime is unavailable for logical_model_id 'support-bot'",
                "type": "upstream_unavailable",
                "code": "upstream_connection_failed",
            }
        }
        assert captured_urls == ["http://vllm-tenant-demo-mistral-7b.test/v1/chat/completions"]

        wallet = client.app.state.db.get_wallet("tenant-demo")
        usage_event = client.app.state.db.list_usage_events()[0]
        assert wallet["available_tokens"] == 100
        assert wallet["reserved_tokens"] == 0
        assert usage_event["request_id"] == request_id
        assert usage_event["status"] == "failed"

    run_with_client(settings, httpx.MockTransport(handler), scenario)


def test_generation_rejected_when_completion_cap_is_zero(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=4)

    async def scenario(client):
        seed_control_plane(client, initial_balance=4)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 402
        assert response.json() == {
            "error": {
                "message": "Insufficient prepaid balance for completion tokens",
                "type": "insufficient_quota",
                "code": "insufficient_prepaid_balance",
            }
        }
        assert captured_requests == []
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert wallet["available_tokens"] == 4
        assert wallet["reserved_tokens"] == 0

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_generation_rejected_with_retry_message_when_balance_is_only_reserved(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=4)

    async def scenario(client):
        seed_control_plane(client, initial_balance=4)
        client.app.state.db.reserve_tokens("tenant-demo", "held-request", 4, 0)

        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 402
        assert response.json() == {
            "error": {
                "message": "Allowance is temporarily reserved by in-flight requests. Please retry in a few seconds.",
                "type": "insufficient_quota",
                "code": "temporarily_reserved",
            }
        }
        assert captured_requests == []
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert wallet["available_tokens"] == 0
        assert wallet["reserved_tokens"] == 4

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_stream_without_usage_marks_reconciliation_pending(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)
    stream_body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        usage_event = client.app.state.db.list_usage_events()[0]
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert usage_event["status"] == "reconciliation_pending"
        assert wallet["available_tokens"] == 86
        assert wallet["reserved_tokens"] == 14

    run_with_client(settings, make_transport(captured_requests, stream_body=stream_body), scenario)


def test_stream_upstream_400_releases_reservation(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)
    stream_body = (
        b"400: backend.ContextWindowExceededError: backend.BadRequestError: "
        b"ContextWindowExceededError: OpenAIException - max_tokens is too large"
    )

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        assert b"ContextWindowExceededError" in response.content
        usage_event = client.app.state.db.list_usage_events()[0]
        wallet = client.app.state.db.get_wallet("tenant-demo")
        assert usage_event["status"] == "failed"
        assert wallet["available_tokens"] == 100
        assert wallet["reserved_tokens"] == 0

    run_with_client(settings, make_transport(captured_requests, stream_body=stream_body, chat_status_code=400), scenario)


def test_stream_request_forces_include_usage_in_forwarded_payload(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)
    stream_body = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n'
        b'data: [DONE]\n\n'
    )

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        forwarded = captured_requests[0]
        assert forwarded["stream_options"]["include_usage"] is True

    run_with_client(settings, make_transport(captured_requests, stream_body=stream_body), scenario)


def test_tokenizer_cache_is_keyed_by_repo_and_revision(tmp_path: Path):
    settings = build_settings(tmp_path, initial_balance=100)
    TokenizerManager._load_tokenizer = lambda self, model_config: FakeTokenizer()  # type: ignore[method-assign]
    manager = TokenizerManager(settings.hf_home)
    from config import ModelConfig

    async def load_both():
        await manager.get_tokenizer(
            ModelConfig(
                logical_model_id="support-bot",
                backend_model="qwen2.5-1.5b",
                tokenizer_repo="repo/shared",
                tokenizer_revision="rev-1",
                route=None,
                model_policy_cap=10,
            )
        )
        await manager.get_tokenizer(
            ModelConfig(
                logical_model_id="support-bot-2",
                backend_model="qwen2.5-1.5b",
                tokenizer_repo="repo/shared",
                tokenizer_revision="rev-1",
                route=None,
                model_policy_cap=8,
            )
        )

    asyncio.run(load_both())
    assert len(manager._cache) == 1
    manager.close()


def test_chat_estimation_method_records_fallback_when_template_fails(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)
    TokenizerManager._load_tokenizer = lambda self, model_config: BrokenChatTemplateTokenizer()  # type: ignore[method-assign]

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Reply in one short sentence about prepaid billing."}],
            },
        )

        assert response.status_code == 200
        usage_event = client.app.state.db.list_usage_events()[0]
        assert usage_event["estimation_method"] == "fallback_word_estimate"

    async def runner():
        async with GatewayHarness(settings, make_transport(captured_requests)) as client:
            await scenario(client)

    asyncio.run(runner())


def test_company_model_route_is_used_directly_without_key_generation(tmp_path: Path):
    captured_requests: list[dict] = []
    captured_urls: list[str] = []
    captured_authorizations: list[str | None] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(
            client,
            initial_balance=100,
            litellm_url="http://unused-legacy-route.invalid",
            route_target="http://vllm-tenant-demo-support-bot.tenant-demo.svc.cluster.local:8000/v1",
        )
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer tenant-demo-key"},
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 200
        assert captured_urls == [
            "http://vllm-tenant-demo-support-bot.tenant-demo.svc.cluster.local:8000/v1/chat/completions",
        ]
        assert captured_authorizations == [None]

    run_with_client(
        settings,
        make_transport(captured_requests, captured_urls=captured_urls, captured_authorizations=captured_authorizations),
        scenario,
    )


def test_control_plane_sync_reads_company_model_route_map():
    tenants = validated_tenants(
        {
            "tenants": [
                {
                    "gateway_api_key": "tenant-demo-key",
                    "company_id": "tenant-demo",
                    "company_name": "Tenant Demo",
                    "team_id": "group-demo",
                    "enabled_models": ["support-bot"],
                    "enabled_model_routes": {
                        "support-bot": "http://vllm-tenant-demo-support-bot.tenant-demo.svc.cluster.local:8000/v1"
                    },
                }
            ]
        },
        {"support-bot"},
    )

    assert tenants[0]["enabled_models"] == [
        {
            "logical_model_id": "support-bot",
            "route_target": "http://vllm-tenant-demo-support-bot.tenant-demo.svc.cluster.local:8000/v1",
        }
    ]


def test_control_plane_sync_prefers_mounted_desired_state_file(tmp_path: Path, monkeypatch):
    desired_state_file = tmp_path / "desired-state.json"
    desired_state_file.write_text(
        json.dumps(
            {
                "sync_prune_mode": "deactivate",
                "models": [
                    {
                        "logical_model_id": "support-bot",
                        "backend_model": "support-bot",
                        "tokenizer_repo": "Qwen/Qwen2.5-0.5B-Instruct",
                        "tokenizer_revision": "c89bee90d9f811437d9735454613c35b4a3c4dc8",
                        "model_policy_cap": 256,
                        "route_target": None,
                    }
                ],
                "tenants": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROL_PLANE_DESIRED_STATE_FILE", str(desired_state_file))
    monkeypatch.setenv("CONTROL_PLANE_DESIRED_STATE_JSON", "{not-json")

    desired_state = load_desired_state()

    assert desired_state["models"][0]["logical_model_id"] == "support-bot"


def test_model_listing_is_served_from_gateway_allowlist(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100, enabled_models=["support-bot", "support-bot-2"])
        response = await client.get("/v1/models", headers={"Authorization": "Bearer tenant-demo-key"})

        assert response.status_code == 200
        assert captured_requests == []
        assert response.json() == {
            "object": "list",
            "data": [
                {
                    "id": "support-bot",
                    "object": "model",
                    "created": 0,
                    "owned_by": "gateway",
                    "metadata": {
                        "backend_model": "qwen2.5-1.5b",
                        "tokenizer_repo": "repo/shared",
                        "tokenizer_revision": "rev-1",
                        "route": "http://vllm-tenant-demo-support-bot.test/v1",
                        "model_policy_cap": 10,
                    },
                },
                {
                    "id": "support-bot-2",
                    "object": "model",
                    "created": 0,
                    "owned_by": "gateway",
                    "metadata": {
                        "backend_model": "qwen2.5-1.5b",
                        "tokenizer_repo": "repo/shared",
                        "tokenizer_revision": "rev-1",
                        "route": "http://vllm-tenant-demo-support-bot.test/v1",
                        "model_policy_cap": 8,
                    },
                },
            ],
        }

    run_with_client(settings, make_transport(captured_requests), scenario)


def test_metrics_endpoint_exposes_gateway_business_metrics(tmp_path: Path):
    captured_requests: list[dict] = []
    settings = build_settings(tmp_path, initial_balance=100)

    async def scenario(client):
        seed_control_plane(client, initial_balance=100)
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer tenant-demo-key",
                "x-gateway-user-id": "user-1",
                "x-gateway-user-name": "User One",
                "x-gateway-user-email": "user@example.com",
            },
            json={
                "model": "support-bot",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200

        metrics_response = await client.get("/metrics")

        assert metrics_response.status_code == 200
        assert 'gateway_requests_total{company_id="tenant-demo",logical_model_id="support-bot",path="/v1/chat/completions"' in metrics_response.text
        assert 'gateway_estimation_method_total{company_id="tenant-demo",logical_model_id="support-bot",method="chat_template_tokens"} 1' in metrics_response.text
        assert 'gateway_wallet_available_tokens{company_id="tenant-demo"} 93' in metrics_response.text

    run_with_client(settings, make_transport(captured_requests), scenario)
