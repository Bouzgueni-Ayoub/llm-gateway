# LLM Gateway

LLM Gateway is an OpenAI-compatible HTTP gateway for self-hosted model
runtimes. It authenticates inbound API keys, checks tenant/model assignments,
reserves prepaid token balance before inference, forwards requests to the
assigned backend route, and reconciles billing from usage reported by the
backend.

The gateway is designed to sit in front of OpenAI-compatible inference servers
such as vLLM, SGLang, TGI-compatible adapters, or any backend that accepts
OpenAI-style requests and returns OpenAI-style usage fields.

## Features

- OpenAI-compatible endpoints:
  - `/v1/chat/completions`
  - `/v1/completions`
  - `/v1/embeddings`
  - `/v1/models`
- Tenant authentication with `Authorization: Bearer <gateway-key>`.
- Tenant/model allowlisting through `(company_id, logical_model_id)`.
- Trusted model registry that maps logical model IDs to backend model names.
- Per-assignment backend routing with `route_target`.
- Prompt-token estimation using Hugging Face tokenizers.
- Prepaid wallet reservation before inference.
- Gateway-owned generation cap enforcement.
- Final billing reconciliation from backend-reported usage.
- Streaming response support with final usage capture from Server-Sent Events.
- Prometheus metrics endpoint.
- Optional desired-state sync script for bootstrapping tenants, models, and
  assignments.

## Repository Contents

| File | Purpose |
| --- | --- |
| `main.py` | Uvicorn entrypoint exposing the FastAPI app. |
| `app_factory.py` | App factory, API routes, request admission, proxying, streaming usage capture, and settlement. |
| `db.py` | Database schema, tenant/model lookups, wallet transactions, ledger writes, and usage-event updates. |
| `config.py` | Runtime settings, request parsing helpers, tokenizer loading, cache management, and prompt estimation. |
| `metrics.py` | Prometheus metrics renderer. |
| `control_plane_sync.py` | Optional desired-state sync utility for writing tenants, models, assignments, wallet seeds, and sync audit rows. |
| `requirements.txt` | Python runtime and test dependencies. |
| `test_gateway.py` | Integration-style tests using an in-process ASGI app and mocked upstream backends. |

## How Requests Flow

![LLM Gateway request flow](assets/architecture.png)

1. A client sends a request with `Authorization: Bearer <gateway-key>`.
2. The gateway looks up the key in `company_team_map`.
3. The incoming `model` value is treated as a logical model ID.
4. The gateway checks that `(company_id, logical_model_id)` is active in
   `tenant_model_assignments`.
5. The `model_registry` row provides the trusted backend model, tokenizer,
   tokenizer revision, and policy cap.
6. The assignment row provides the direct backend `route_target`.
7. The gateway estimates prompt tokens with the configured tokenizer.
8. The gateway checks wallet balance and reserves prompt estimate plus allowed
   completion tokens.
9. The outbound request is rewritten:
   - `model` becomes `backend_model`
   - `max_tokens` and `max_completion_tokens` become the gateway-approved cap
   - streaming requests get `stream_options.include_usage = true`
   - gateway metadata is attached
10. The request is forwarded to the assigned backend route.
11. Usage is read from the JSON response or final streaming event.
12. The wallet reservation is settled against actual usage.
13. The usage event is marked `reconciled`, `failed`, or
   `reconciliation_pending`.

## Requirements

- Python 3.10 or newer.
- PostgreSQL for shared or production deployments.
- SQLite for local development and tests.
- Network access from the gateway to each configured backend route.
- Network access to Hugging Face, or a pre-populated tokenizer cache, for
  tokenizer loading.
- Pinned tokenizer revisions. Avoid floating refs such as `main`, `master`, or
  `latest`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API Server

Set the runtime environment:

```bash
export DATABASE_URL='postgresql://gateway:gateway_pass@postgres.example.internal:5432/gateway'
export GATEWAY_ADMIN_TOKEN='replace-with-a-secret'
export HF_HOME='/var/lib/llm-gateway/hf-cache'
```

Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

For local development, SQLite is supported:

```bash
export DATABASE_URL='sqlite:///./gateway.db'
uvicorn main:app --host 127.0.0.1 --port 5000
```

The gateway creates missing tables at startup. It does not automatically create
tenants, model registry rows, assignments, routes, or wallet funds for you.
Populate those rows before sending inference requests.

## Environment Variables

### API Server

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Database connection string. Supports `postgresql://`, `postgres://`, and `sqlite:///`. |
| `GATEWAY_ADMIN_TOKEN` | Recommended | Token required by admin endpoints. If empty, admin endpoints are unauthenticated. |
| `HF_HOME` | Recommended | Hugging Face tokenizer cache path. Defaults to `/hf-cache`. |

### Optional Sync Utility

`control_plane_sync.py` can apply a desired-state JSON document into the
database. This is optional. You can also manage the tables with migrations,
SQL, an admin service, or the admin API.

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Database connection string. `GATEWAY_DATABASE_URL` is accepted as a fallback. |
| `CONTROL_PLANE_DESIRED_STATE_FILE` | Recommended | Path to a JSON file containing models, tenants, assignments, route targets, and initial wallet seeds. |
| `CONTROL_PLANE_DESIRED_STATE_JSON` | Fallback | Inline JSON object for local development when no file path is set. |
| `SYNC_PRUNE_MODE` | Optional | Currently only `deactivate` is supported. Missing rows are marked inactive instead of deleted. |

Run the sync utility:

```bash
export CONTROL_PLANE_DESIRED_STATE_FILE=./desired-state.json
python control_plane_sync.py
```

## Desired-State JSON Example

```json
{
  "sync_prune_mode": "deactivate",
  "models": [
    {
      "logical_model_id": "chat-small",
      "backend_model": "llama-3.2-3b-instruct",
      "tokenizer_repo": "meta-llama/Llama-3.2-3B-Instruct",
      "tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
      "model_policy_cap": 512,
      "route_target": null
    }
  ],
  "tenants": [
    {
      "gateway_api_key": "tenant-demo-key",
      "company_id": "tenant-demo",
      "company_name": "Tenant Demo",
      "team_id": "tenant-demo",
      "initial_balance_tokens": 50000,
      "low_balance_threshold": 1000,
      "enabled_models": ["chat-small"],
      "enabled_model_routes": {
        "chat-small": "http://llm-backend.default.svc.cluster.local:8000/v1"
      }
    }
  ]
}
```

Notes:

- `gateway_api_key` is the inbound API key expected in the client's
  `Authorization` header.
- `team_id` is an optional grouping identifier retained by the current schema.
  Direct request routing uses `tenant_model_assignments.route_target`.
- `enabled_model_routes` writes per-tenant, per-model backend routes into
  `tenant_model_assignments`.
- `model_registry.route_target` is retained for compatibility, but assignment
  route targets are preferred.
- The sync utility creates a wallet if one is missing.
- `initial_balance_tokens` is seeded only when the wallet is first created.
  Existing wallets are not silently topped up by repeated sync runs.
- `SYNC_PRUNE_MODE=deactivate` marks missing tenants, models, or assignments
  inactive instead of deleting historical rows.

## Minimum Database State

A usable tenant/model route needs:

1. An active row in `company_team_map`.
2. An active row in `model_registry`.
3. An active row in `tenant_model_assignments` for the exact
   `(company_id, logical_model_id)` pair.
4. A non-empty `route_target` on the assignment.
5. A wallet row in `wallet_accounts` with enough `available_tokens`.

## Database Tables

### `company_team_map`

Stores inbound gateway keys and tenant identity.

| Column | Meaning |
| --- | --- |
| `gateway_api_key` | Inbound gateway API key from `Authorization`. Stored in the current schema's key column. |
| `company_id` | Stable tenant ID used for assignment and billing. |
| `company_name` | Human-readable tenant name. |
| `team_id` | Optional grouping identifier retained for compatibility. |
| `is_active` | Active flag. |
| `created_at`, `updated_at` | Audit timestamps. |

### `model_registry`

Stores logical model metadata trusted by the gateway.

| Column | Meaning |
| --- | --- |
| `logical_model_id` | Client-facing model ID and primary key. |
| `backend_model` | Model ID sent to the backend after request rewriting. |
| `tokenizer_repo` | Hugging Face tokenizer repo used for reservation estimates. |
| `tokenizer_revision` | Pinned tokenizer revision. Use an immutable commit SHA. |
| `model_policy_cap` | Gateway-enforced max completion reservation cap. |
| `route_target` | Compatibility/global route field; assignment route is preferred. |
| `is_active` | Active flag. |
| `created_at`, `updated_at` | Audit timestamps. |

### `tenant_model_assignments`

Stores allowlisted tenant/model pairs and their backend routes.

| Column | Meaning |
| --- | --- |
| `company_id` | Tenant ID. |
| `logical_model_id` | Assigned logical model ID. |
| `route_target` | Required upstream base URL for this tenant/model. |
| `is_active` | Active flag. |
| `created_at`, `updated_at` | Audit timestamps. |

Primary key: `(company_id, logical_model_id)`.

### `wallet_accounts`

Stores current prepaid balance state.

| Column | Meaning |
| --- | --- |
| `company_id` | Tenant ID and primary key. |
| `available_tokens` | Spendable prepaid balance. |
| `reserved_tokens` | Tokens held by in-flight requests. |
| `status` | Wallet state, currently `active`. |
| `low_balance_threshold` | Alert/metric threshold. |
| `version` | Incremented on wallet mutations. |
| `created_at`, `updated_at` | Audit timestamps. |

### `wallet_ledger`

Append-only wallet movement history.

| Column | Meaning |
| --- | --- |
| `id` | Ledger row ID. |
| `company_id` | Tenant ID. |
| `request_id` | Gateway request ID when movement is request-related. |
| `entry_type` | `purchase`, `reserve`, `release`, `finalize`, or admin adjustment. |
| `delta_tokens` | Token movement. Negative reserves/charges, positive purchases/releases. |
| `status` | Ledger state such as `committed` or `reconciled`. |
| `idempotency_key` | Prevents duplicate wallet mutation. |
| `metadata_json` | Request/model/source details. |
| `created_at` | Audit timestamp. |

### `usage_events`

Request audit and reconciliation state.

| Column | Meaning |
| --- | --- |
| `request_id` | Gateway request ID and primary key. |
| `company_id` | Tenant ID. |
| `user_id` | Optional request user identity if provided by the client. |
| `logical_model_id` | Client-facing model ID. |
| `backend_model` | Backend model sent upstream. |
| `tokenizer_repo`, `tokenizer_revision` | Tokenizer used for estimation. |
| `estimated_prompt_tokens` | Pre-inference prompt estimate. |
| `estimation_method` | Estimation path, for example `chat_template_tokens`. |
| `reserved_completion_tokens` | Gateway-enforced completion reservation. |
| `actual_prompt_tokens`, `actual_completion_tokens`, `actual_total_tokens` | Final backend-reported usage. |
| `latency_ms` | Request latency. |
| `status` | `pending`, `completed`, `failed`, `reconciliation_pending`, or `reconciled`. |
| `stream` | `1` for streaming requests, otherwise `0`. |
| `created_at`, `updated_at` | Audit timestamps. |

### `control_plane_sync_runs`

Records each optional desired-state sync.

| Column | Meaning |
| --- | --- |
| `id` | Sync run ID. |
| `desired_state_sha256` | Hash of normalized desired state. |
| `sync_prune_mode` | Currently `deactivate`. |
| `tenant_count`, `model_count`, `assignment_count` | Synced object counts. |
| `metadata_json` | Synced tenant/model IDs. |
| `created_at` | Audit timestamp. |

## Public API

All inference and model-listing APIs use the inbound gateway key:

```http
Authorization: Bearer tenant-demo-key
```

### `GET /health`

Unauthenticated process health check.

```bash
curl http://localhost:5000/health
```

### `GET /v1/models`

Returns active logical models assigned to the authenticated tenant.

```bash
curl http://localhost:5000/v1/models \
  -H "Authorization: Bearer tenant-demo-key"
```

`GET /models` is also available for clients that use the non-versioned path.

### `POST /v1/chat/completions`

OpenAI-compatible chat endpoint.

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer tenant-demo-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat-small",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 32
  }'
```

The gateway may reduce `max_tokens` or `max_completion_tokens` before
forwarding the request if policy or prepaid balance requires a lower cap.

### `POST /v1/completions`

OpenAI-compatible text completion endpoint. It uses the same authentication,
assignment, reservation, and routing path.

### `POST /v1/embeddings`

OpenAI-compatible embeddings endpoint. It uses the same authentication,
assignment, and routing path, but generation completion caps are not applied.

### `GET /metrics`

Prometheus metrics endpoint.

```bash
curl http://localhost:5000/metrics
```

Expose this endpoint only through trusted networking or ingress restrictions.

## Admin API

If `GATEWAY_ADMIN_TOKEN` is set, pass it as `x-gateway-admin-token` or as a
bearer token.

```bash
curl http://localhost:5000/wallets \
  -H "x-gateway-admin-token: replace-with-a-secret"
```

Endpoints:

- `GET /company-mappings`: list tenant key mappings with masked keys.
- `POST /company-mappings/upsert`: upsert one tenant mapping and ensure wallet.
- `POST /company-mappings/deactivate`: deactivate one inbound gateway key.
- `GET /wallets`: list wallet states.
- `POST /wallets/adjust`: add or subtract wallet balance manually.
- `GET /usage-events`: list request audit rows.
- `GET /model-registry`: list model metadata.

Example wallet adjustment:

```bash
curl http://localhost:5000/wallets/adjust \
  -H "x-gateway-admin-token: replace-with-a-secret" \
  -H "Content-Type: application/json" \
  -d '{"company_id":"tenant-demo","delta_tokens":1000,"entry_type":"purchase"}'
```

## Operational Notes

- Unknown gateway keys fail with `401` or `403` before model access.
- Missing or inactive model assignments fail before reaching the backend.
- Missing assignment `route_target` fails closed with `503`.
- The gateway strips inbound `Authorization` before forwarding upstream.
- Final billing settles from backend-reported usage, not from the pre-inference
  estimate.
- Tokenizer estimates exist to reserve prepaid balance before inference starts.
- Tokenizer revisions must be pinned to immutable revisions.
- Wallet mutations use transactions and idempotency keys to avoid duplicate
  reservation or settlement.
- Existing wallets should be funded by explicit purchase, admin, or
  control-plane actions, not by routine sync.

## Testing

Run the test suite from this directory:

```bash
pytest -vv test_gateway.py
```

The tests exercise the real FastAPI app and database logic with mocked upstream
model backends. They do not require a live model server.
