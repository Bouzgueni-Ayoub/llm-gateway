import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Module-level defaults are read from environment variables when this file is imported.
# `GatewaySettings.from_env()` copies these values into one settings object that is
# later attached to `app.state` by `app_factory.create_app`.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://gateway:gateway_pass@gateway-postgres:5432/gateway")
GATEWAY_ADMIN_TOKEN = os.getenv("GATEWAY_ADMIN_TOKEN", "")
BOOTSTRAP_COMPANY_MAPPINGS_JSON = os.getenv("BOOTSTRAP_COMPANY_MAPPINGS_JSON", "")
MODEL_REGISTRY_JSON = os.getenv("MODEL_REGISTRY_JSON", "")
PRELOAD_LOGICAL_MODELS_JSON = os.getenv("PRELOAD_LOGICAL_MODELS_JSON", "")
HF_HOME = os.getenv("HF_HOME", "/hf-cache")

LEDGER_STATUS_COMMITTED = "committed"
LEDGER_STATUS_VOIDED = "voided"
LEDGER_STATUS_RECONCILED = "reconciled"

USAGE_STATUS_PENDING = "pending"
USAGE_STATUS_COMPLETED = "completed"
USAGE_STATUS_FAILED = "failed"
USAGE_STATUS_CANCELLED = "cancelled"
USAGE_STATUS_RECONCILIATION_PENDING = "reconciliation_pending"
USAGE_STATUS_RECONCILED = "reconciled"

GENERATION_PATHS = {"/v1/chat/completions", "/v1/completions"}


# `@dataclass(frozen=True)` creates a small immutable object with named fields.
# Each `ModelConfig` instance represents one trusted model registry row from
# Postgres or from startup JSON; request code passes this object to tokenization,
# routing, reservation, and audit helpers instead of passing loose dictionaries.
@dataclass(frozen=True)
class ModelConfig:
    logical_model_id: str
    backend_model: str
    tokenizer_repo: str
    tokenizer_revision: str | None
    route: str | None
    model_policy_cap: int
    preload: bool = False


# Prompt estimation needs to return both a number and how that number was found.
# The `method` field is stored in metrics so operators can see whether requests
# used the tokenizer path or a rough fallback.
@dataclass(frozen=True)
class PromptEstimate:
    tokens: int
    method: str


# Runtime settings are grouped into one object so route handlers do not read
# environment variables directly. In tests, callers can pass a custom
# `GatewaySettings` object to `create_app` without changing process-wide env.
@dataclass
class GatewaySettings:
    database_url: str
    gateway_admin_token: str
    bootstrap_company_mappings_json: str
    model_registry_json: str
    preload_logical_models_json: str
    hf_home: str

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        # A classmethod receives the class as `cls`, then returns an instance of it.
        # This keeps environment parsing in one place and gives the rest of the
        # app normal attributes such as `settings.database_url`.
        return cls(
            database_url=DATABASE_URL,
            gateway_admin_token=GATEWAY_ADMIN_TOKEN,
            bootstrap_company_mappings_json=BOOTSTRAP_COMPANY_MAPPINGS_JSON,
            model_registry_json=MODEL_REGISTRY_JSON,
            preload_logical_models_json=PRELOAD_LOGICAL_MODELS_JSON,
            hf_home=HF_HOME,
        )


def normalize_text(value: Any) -> str | None:
    # Many inputs arrive as optional strings from headers, env vars, or JSON.
    # Converting to `str`, trimming, and returning `None` for blanks gives all
    # callers one common "missing value" shape.
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_json_dict(raw: bytes) -> dict[str, Any]:
    # FastAPI and httpx expose HTTP bodies as bytes. The gateway only handles
    # JSON objects here because OpenAI-compatible request bodies are objects;
    # invalid JSON, arrays, and scalar values become an empty dict.
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_possible_dict(value: Any) -> dict[str, Any] | None:
    # Some OpenAI payload fields are already dictionaries, while other clients
    # send nested JSON as a string. This helper accepts both forms and rejects
    # anything that does not decode to an object.
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_bearer_token(authorization_header: str | None) -> str | None:
    # The gateway accepts the usual `Authorization: Bearer token` form and also
    # a raw token value for simple test clients. The returned value is only the
    # secret token, which is then looked up in Postgres.
    value = normalize_text(authorization_header)
    if not value:
        return None
    parts = value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return normalize_text(parts[1])
    return value


def mask_key(raw_key: str | None) -> str | None:
    # Admin list endpoints need to show that a key exists without leaking it.
    # Short keys are fully hidden; longer keys show only a small prefix/suffix.
    if not raw_key:
        return None
    if len(raw_key) <= 8:
        return "***"
    return f"{raw_key[:4]}...{raw_key[-4:]}"


def parse_optional_positive_int(value: Any) -> int | None:
    # Several config fields are optional numeric caps. Returning `None` for
    # missing, invalid, zero, or negative values lets callers treat all of those
    # cases as "not configured".
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def utcnow() -> str:
    # Store timestamps as ISO-8601 UTC strings so SQLite and Postgres tests can
    # share the same value format without database-specific timestamp handling.
    return datetime.now(timezone.utc).isoformat()


def bool_to_int(value: Any) -> int:
    # Database rows use integer flags so the same schema works in SQLite and Postgres.
    return 1 if bool(value) else 0


def safe_json_dumps(value: Any) -> str:
    # Compact JSON is used for request rewrites and ledger metadata. Keeping it
    # stable avoids noisy DB values while preserving non-ASCII user content.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def estimate_text_tokens(text: str) -> int:
    # This fallback is deliberately conservative and simple: it counts words.
    # Normal billing should use tokenizer or vLLM-reported usage; this path only
    # lets the gateway reserve something when tokenizer estimation fails.
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


def db_path_from_url(database_url: str) -> str:
    # Tests use SQLite URLs such as `sqlite:///tmp/gateway.db`. The sqlite3
    # module wants only the filesystem path, so this validates and strips the scheme.
    if not database_url.startswith("sqlite:///"):
        raise ValueError("DATABASE_URL must use sqlite:/// path format")
    return database_url.replace("sqlite:///", "", 1)


def load_bootstrap_company_mappings(raw_json: str) -> list[dict[str, Any]]:
    # Legacy/dev startup input can define company mappings as a JSON array.
    # This parser normalizes each item into the shape expected by `GatewayDB`;
    # production control-plane sync should write these rows into Postgres instead.
    raw = normalize_text(raw_json)
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
    except Exception:
        print("BOOTSTRAP_COMPANY_MAPPINGS_JSON is not valid JSON; skipping bootstrap")
        return []

    if not isinstance(parsed, list):
        print("BOOTSTRAP_COMPANY_MAPPINGS_JSON must be a JSON array; skipping bootstrap")
        return []

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        openwebui_api_key = normalize_text(item.get("gateway_api_key") or item.get("openwebui_api_key"))
        company_name = normalize_text(item.get("company_name"))
        company_id = normalize_text(item.get("company_id")) or company_name
        team_id = normalize_text(item.get("team_id"))
        litellm_url = normalize_text(item.get("litellm_url"))
        if not openwebui_api_key or not company_name or not company_id or not team_id:
            continue
        out.append(
            {
                "openwebui_api_key": openwebui_api_key,
                "company_name": company_name,
                "company_id": company_id,
                "team_id": team_id,
                "litellm_url": litellm_url,
                "default_max_tokens": parse_optional_positive_int(item.get("default_max_tokens")),
                "initial_balance_tokens": parse_optional_positive_int(item.get("initial_balance_tokens")) or 0,
                "low_balance_threshold": parse_optional_positive_int(item.get("low_balance_threshold")) or 0,
            }
        )
    return out


def _default_model_registry_raw() -> list[dict[str, Any]]:
    # Fallback registry for local startup without Postgres-seeded model rows.
    # The active gateway should normally read model registry rows from Postgres.
    return [
        {
            "logical_model_id": "qwen2.5-1.5b",
            "backend_model": "qwen2.5-1.5b",
            "tokenizer_repo": "Qwen/Qwen2.5-1.5B-Instruct",
            "tokenizer_revision": None,
            "route": None,
            "model_policy_cap": 1024,
            "preload": True,
        }
    ]


# `ModelRegistry` is the in-memory form of the trusted model allowlist used by
# older bootstrap paths. Postgres-backed request handling uses equivalent rows
# from `GatewayDB.get_company_model_config`.
class ModelRegistry:

    def __init__(self, raw_json: str) -> None:
        # Parse once at startup so later model lookups are dictionary reads.
        self._models = self._load_models(raw_json)

    def _load_models(self, raw_json: str) -> dict[str, ModelConfig]:
        # The returned dict is keyed by public logical model id. That public id
        # is what clients request; the `ModelConfig` value contains the backend
        # model name, tokenizer identity, quota cap, and optional route.
        raw = normalize_text(raw_json)
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                raise RuntimeError("MODEL_REGISTRY_JSON must be valid JSON") from exc
        else:
            parsed = _default_model_registry_raw()

        if not isinstance(parsed, list):
            raise RuntimeError("MODEL_REGISTRY_JSON must be a JSON array")

        models: dict[str, ModelConfig] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            logical_model_id = normalize_text(item.get("logical_model_id") or item.get("model_id"))
            backend_model = normalize_text(item.get("backend_model"))
            tokenizer_repo = normalize_text(item.get("tokenizer_repo"))
            tokenizer_revision = normalize_text(item.get("tokenizer_revision"))
            route = normalize_text(item.get("route"))
            model_policy_cap = parse_optional_positive_int(item.get("model_policy_cap"))
            if not logical_model_id or not backend_model or not tokenizer_repo or model_policy_cap is None:
                continue
            models[logical_model_id] = ModelConfig(
                logical_model_id=logical_model_id,
                backend_model=backend_model,
                tokenizer_repo=tokenizer_repo,
                tokenizer_revision=tokenizer_revision,
                route=route,
                model_policy_cap=model_policy_cap,
                preload=bool(item.get("preload")),
            )

        if not models:
            raise RuntimeError("At least one valid model registry entry is required")
        return models

    def get(self, logical_model_id: str) -> ModelConfig | None:
        # Return `None` for unknown model ids so callers can reject access cleanly.
        return self._models.get(logical_model_id)

    def list_models(self) -> list[dict[str, Any]]:
        # Admin responses should be JSON-serializable, so dataclass objects are
        # expanded into plain dictionaries.
        return [
            {
                "logical_model_id": model.logical_model_id,
                "backend_model": model.backend_model,
                "tokenizer_repo": model.tokenizer_repo,
                "tokenizer_revision": model.tokenizer_revision,
                "route": model.route,
                "model_policy_cap": model.model_policy_cap,
                "preload": model.preload,
            }
            for model in self._models.values()
        ]

    def list_openai_models(self) -> dict[str, Any]:
        # OpenAI clients expect a top-level `object: list` response with model
        # entries under `data`. Extra gateway details live under `metadata`.
        return {
            "object": "list",
            "data": [
                {
                    "id": model.logical_model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "gateway",
                    "metadata": {
                        "backend_model": model.backend_model,
                        "tokenizer_repo": model.tokenizer_repo,
                        "tokenizer_revision": model.tokenizer_revision,
                        "route": model.route,
                        "model_policy_cap": model.model_policy_cap,
                    },
                }
                for model in self._models.values()
            ],
        }

    def preload_candidates(self, raw_json: str) -> list[ModelConfig]:
        # Tokenizer preloading can be named explicitly by logical model id. If
        # no list is provided, registry entries marked `preload` are used.
        requested: list[str] = []
        raw = normalize_text(raw_json)
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                requested = [str(item).strip() for item in parsed if str(item).strip()]
        if requested:
            return [self._models[model_id] for model_id in requested if model_id in self._models]
        return [model for model in self._models.values() if model.preload]


# `TokenizerManager` owns tokenizer objects used for prompt estimation before a
# request is admitted. It depends on `ModelConfig` for the repo/revision and
# returns `PromptEstimate` so callers know both the count and the counting path.
class TokenizerManager:

    def __init__(self, hf_home: str) -> None:
        # Cache keys include repo and revision so two logical models can safely
        # share one tokenizer when they point at the same pinned source.
        # `_guard` protects the lock dictionary; the per-key locks protect the
        # expensive load for each tokenizer.
        self.hf_home = hf_home
        self._cache: dict[tuple[str, str | None], Any] = {}
        self._locks: dict[tuple[str, str | None], asyncio.Lock] = {}
        self._guard = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-gateway-tokenizer")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_home)
            os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    async def preload(self, models: list[ModelConfig]) -> None:
        # Preloading moves tokenizer download/load cost to startup. Failure is
        # logged but not fatal because the first request can still retry loading.
        for model in models:
            try:
                await self.get_tokenizer(model)
            except Exception as exc:
                print(f"Tokenizer preload failed for {model.logical_model_id}: {exc}")

    async def _lock_for_cache_key(self, cache_key: tuple[str, str | None]) -> asyncio.Lock:
        # `asyncio.Lock` lets coroutines wait without blocking the entire event
        # loop. One lock per tokenizer prevents duplicate concurrent downloads.
        async with self._guard:
            if cache_key not in self._locks:
                self._locks[cache_key] = asyncio.Lock()
            return self._locks[cache_key]

    async def get_tokenizer(self, model_config: ModelConfig) -> Any:
        # Hugging Face loading is blocking file/network work, so it runs in the
        # thread pool. The double cache check avoids loading the same tokenizer
        # twice when two requests arrive at the same time.
        cache_key = (model_config.tokenizer_repo, model_config.tokenizer_revision)
        existing = self._cache.get(cache_key)
        if existing is not None:
            return existing

        lock = await self._lock_for_cache_key(cache_key)
        async with lock:
            existing = self._cache.get(cache_key)
            if existing is not None:
                return existing
            loop = asyncio.get_running_loop()
            tokenizer = await loop.run_in_executor(self._executor, self._load_tokenizer, model_config)
            self._cache[cache_key] = tokenizer
            return tokenizer

    def _load_tokenizer(self, model_config: ModelConfig) -> Any:
        # The repo and revision come from trusted model configuration, not the
        # client request. `AutoTokenizer` chooses the right tokenizer class from
        # Hugging Face metadata. `trust_remote_code=True` allows repository code
        # to run locally, so model revisions should stay pinned and reviewed.
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for tokenizer-based prompt estimation") from exc

        return AutoTokenizer.from_pretrained(
            model_config.tokenizer_repo,
            revision=model_config.tokenizer_revision,
            trust_remote_code=True,
        )

    async def estimate_prompt_tokens(self, model_config: ModelConfig, path: str, payload: dict[str, Any]) -> PromptEstimate:
        # Reservation happens before vLLM returns final usage, so this method
        # estimates prompt tokens from the request body. Final settlement still
        # uses vLLM-reported usage when it is available.
        tokenizer = await self.get_tokenizer(model_config)

        if path == "/v1/chat/completions":
            messages = payload.get("messages")
            if isinstance(messages, list) and hasattr(tokenizer, "apply_chat_template"):
                # Chat templates add system/user/assistant markers. Counting
                # those rendered markers makes reservations closer to backend usage.
                try:
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    # Different tokenizer implementations return different
                    # containers; both list length and tensor-like `shape` reveal
                    # how many token ids were produced.
                    if isinstance(rendered, list):
                        return PromptEstimate(tokens=len(rendered), method="chat_template_tokens")
                    if hasattr(rendered, "shape") and getattr(rendered, "shape", None):
                        return PromptEstimate(tokens=int(rendered.shape[-1]), method="chat_template_tokens")
                except Exception:
                    pass
                try:
                    # Some tokenizers cannot return token ids directly from the
                    # template call. Rendering text first and tokenizing that text
                    # still counts the formatted chat prompt.
                    rendered_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    encoded = tokenizer(rendered_text, add_special_tokens=False)
                    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
                    return PromptEstimate(tokens=len(input_ids or []), method="chat_template_rendered_text")
                except Exception:
                    pass
            text = safe_json_dumps(messages if isinstance(messages, list) else payload)
            # If tokenizer-aware chat counting fails, fall back to a rough word count.
            return PromptEstimate(tokens=estimate_text_tokens(text), method="fallback_word_estimate")

        if path == "/v1/completions":
            # Legacy completions have a `prompt` field rather than chat messages.
            prompt = payload.get("prompt")
            if isinstance(prompt, list):
                prompt = "\n".join(str(item) for item in prompt)
            prompt_text = prompt if isinstance(prompt, str) else safe_json_dumps(prompt)
            try:
                encoded = tokenizer(prompt_text, add_special_tokens=True)
                # `input_ids` are numeric token identifiers; their count is the
                # prompt token estimate.
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
                return PromptEstimate(tokens=len(input_ids or []), method="prompt_tokenizer")
            except Exception:
                return PromptEstimate(tokens=estimate_text_tokens(prompt_text), method="fallback_word_estimate")

        if path == "/v1/embeddings":
            # Embedding requests do not generate completion tokens, so only the
            # input side needs a reservation estimate.
            input_value = payload.get("input")
            if isinstance(input_value, list):
                input_text = "\n".join(str(item) for item in input_value)
            else:
                input_text = str(input_value or "")
            try:
                encoded = tokenizer(input_text, add_special_tokens=True)
                input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", [])
                return PromptEstimate(tokens=len(input_ids or []), method="embeddings_tokenizer")
            except Exception:
                return PromptEstimate(tokens=estimate_text_tokens(input_text), method="fallback_word_estimate")

        return PromptEstimate(tokens=estimate_text_tokens(safe_json_dumps(payload)), method="fallback_word_estimate")
