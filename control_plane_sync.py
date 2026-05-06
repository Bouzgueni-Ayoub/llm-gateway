from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from db import GatewayDB


def env(name: str, default: str | None = None) -> str:
    # Sync is a batch-style process: missing required environment values should
    # stop the run immediately instead of letting partial control-plane rows be written.
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"{name} is required")
    return value


def normalize_text(value: Any) -> str | None:
    # Desired-state JSON may contain empty strings. The database layer expects
    # absent optional values as `None`, so normalize blanks before validation.
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_positive_int(value: Any, *, default: int = 0) -> int:
    # Counts and token balances may be zero, but negative values would create
    # invalid policy or wallet state.
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise RuntimeError(f"Expected non-negative integer, got {value!r}")
    return parsed


def load_desired_state() -> dict[str, Any]:
    # The sync input is one JSON object. Prefer a mounted file for Kubernetes
    # jobs, while keeping the env-JSON path for local compose/dev compatibility.
    desired_state_file = normalize_text(os.getenv("CONTROL_PLANE_DESIRED_STATE_FILE"))
    if desired_state_file:
        try:
            with open(desired_state_file, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise RuntimeError(f"Could not read CONTROL_PLANE_DESIRED_STATE_FILE: {desired_state_file}") from exc
        source = "CONTROL_PLANE_DESIRED_STATE_FILE"
    else:
        raw = env("CONTROL_PLANE_DESIRED_STATE_JSON")
        source = "CONTROL_PLANE_DESIRED_STATE_JSON"

    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"{source} must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{source} must contain a JSON object")
    return payload


def validated_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # Model entries become rows in `model_registry`. Each logical model id is a
    # client-facing name, and the backend/tokenizer fields tell the gateway how
    # to route and estimate that logical model.
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise RuntimeError("desired state must contain a non-empty models list")

    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            raise RuntimeError("model entries must be objects")
        logical_model_id = normalize_text(item.get("logical_model_id"))
        backend_model = normalize_text(item.get("backend_model"))
        tokenizer_repo = normalize_text(item.get("tokenizer_repo"))
        tokenizer_revision = normalize_text(item.get("tokenizer_revision"))
        route_target = normalize_text(item.get("route_target"))
        model_policy_cap = parse_positive_int(item.get("model_policy_cap"))
        if not logical_model_id or not backend_model or not tokenizer_repo or not tokenizer_revision:
            raise RuntimeError(f"model entry is missing required fields: {item}")
        if logical_model_id in seen_ids:
            raise RuntimeError(f"duplicate logical_model_id in desired state: {logical_model_id}")
        seen_ids.add(logical_model_id)
        models.append(
            {
                "logical_model_id": logical_model_id,
                "backend_model": backend_model,
                "tokenizer_repo": tokenizer_repo,
                "tokenizer_revision": tokenizer_revision,
                "route_target": route_target,
                "model_policy_cap": model_policy_cap,
            }
        )
    return models


def validated_tenants(payload: dict[str, Any], model_ids: set[str]) -> list[dict[str, Any]]:
    # Tenant entries become rows in `company_team_map` plus assignment rows in
    # `tenant_model_assignments`. Assignments are validated against the model
    # list so sync cannot grant access to an undefined logical model.
    raw_tenants = payload.get("tenants")
    if not isinstance(raw_tenants, list) or not raw_tenants:
        raise RuntimeError("desired state must contain a non-empty tenants list")

    tenants: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_company_ids: set[str] = set()
    for item in raw_tenants:
        if not isinstance(item, dict):
            raise RuntimeError("tenant entries must be objects")
        openwebui_api_key = normalize_text(item.get("gateway_api_key") or item.get("openwebui_api_key"))
        company_id = normalize_text(item.get("company_id"))
        company_name = normalize_text(item.get("company_name"))
        team_id = normalize_text(item.get("team_id"))
        litellm_url = normalize_text(item.get("litellm_url"))
        default_max_tokens = parse_positive_int(item.get("default_max_tokens"), default=0) or None
        initial_balance_tokens = parse_positive_int(item.get("initial_balance_tokens"), default=0)
        low_balance_threshold = parse_positive_int(item.get("low_balance_threshold"), default=0)
        enabled_model_routes_raw = item.get("enabled_model_routes") or {}
        if not isinstance(enabled_model_routes_raw, dict):
            raise RuntimeError(f"tenant enabled_model_routes must be an object when provided: {item}")
        # The compact JSON form can list models separately from their per-model
        # routes. This map is used below when `enabled_models` contains strings.
        enabled_model_routes = {
            model_id: route_target
            for model_id, route_target in (
                (normalize_text(model_id), normalize_text(route_target))
                for model_id, route_target in enabled_model_routes_raw.items()
            )
            if model_id
        }

        enabled_models = item.get("enabled_models")
        if not isinstance(enabled_models, list):
            raise RuntimeError(f"tenant enabled_models must be a list: {item}")
        normalized_enabled_models: list[dict[str, str | None]] = []
        seen_enabled_models: set[str] = set()
        for enabled_model in enabled_models:
            # Two input shapes are supported:
            # a string logical model id, or an object with logical_model_id and route_target.
            if isinstance(enabled_model, dict):
                model_id = normalize_text(enabled_model.get("logical_model_id"))
                route_target = normalize_text(enabled_model.get("route_target"))
            else:
                model_id = normalize_text(enabled_model)
                route_target = enabled_model_routes.get(model_id) if model_id else None
            if not model_id:
                continue
            if model_id in seen_enabled_models:
                raise RuntimeError(f"duplicate enabled model {model_id} for tenant {company_id}")
            seen_enabled_models.add(model_id)
            normalized_enabled_models.append({"logical_model_id": model_id, "route_target": route_target})
        if not openwebui_api_key or not company_id or not company_name or not team_id:
            raise RuntimeError(f"tenant entry is missing required fields: {item}")
        if openwebui_api_key in seen_keys:
            raise RuntimeError(f"duplicate gateway_api_key in desired state for company {company_id}")
        if company_id in seen_company_ids:
            raise RuntimeError(f"duplicate company_id in desired state: {company_id}")
        unknown_models = sorted({model["logical_model_id"] for model in normalized_enabled_models} - model_ids)
        if unknown_models:
            raise RuntimeError(f"tenant {company_id} references unknown models: {', '.join(unknown_models)}")
        seen_keys.add(openwebui_api_key)
        seen_company_ids.add(company_id)
        tenants.append(
            {
                "openwebui_api_key": openwebui_api_key,
                "company_id": company_id,
                "company_name": company_name,
                "team_id": team_id,
                "litellm_url": litellm_url,
                "default_max_tokens": default_max_tokens,
                "initial_balance_tokens": initial_balance_tokens,
                "low_balance_threshold": low_balance_threshold,
                "enabled_models": normalized_enabled_models,
            }
        )
    return tenants


def main() -> None:
    # `main` turns desired state into active Postgres rows. The gateway API then
    # reads those rows during requests; it does not read this JSON directly.
    database_url = env("DATABASE_URL", os.getenv("GATEWAY_DATABASE_URL"))
    sync_prune_mode = normalize_text(os.getenv("SYNC_PRUNE_MODE", "deactivate")) or "deactivate"
    if sync_prune_mode != "deactivate":
        raise RuntimeError(f"Unsupported SYNC_PRUNE_MODE: {sync_prune_mode}")

    desired_state = load_desired_state()
    models = validated_models(desired_state)
    model_ids = {model["logical_model_id"] for model in models}
    tenants = validated_tenants(desired_state, model_ids)
    # The assignment key is the trust boundary: access is allowed only when this
    # exact `(company_id, logical_model_id)` pair exists and is active.
    desired_assignments = {
        (tenant["company_id"], enabled_model["logical_model_id"]): enabled_model["route_target"]
        for tenant in tenants
        for enabled_model in tenant["enabled_models"]
    }

    db = GatewayDB(database_url)
    db.open()
    try:
        db.init()

        for tenant in tenants:
            # Upsert active tenant rows and create missing wallets. Existing wallets
            # are not topped up by sync; only a newly-created wallet gets the initial seed.
            db.upsert_company_mapping(
                openwebui_api_key=tenant["openwebui_api_key"],
                company_id=tenant["company_id"],
                company_name=tenant["company_name"],
                team_id=tenant["team_id"],
                litellm_url=tenant["litellm_url"],
                default_max_tokens=tenant["default_max_tokens"],
                is_active=1,
            )
            wallet_created = db.ensure_wallet_account(
                tenant["company_id"],
                low_balance_threshold=tenant["low_balance_threshold"],
            )
            if wallet_created and tenant["initial_balance_tokens"] > 0:
                db.adjust_wallet_balance(
                    tenant["company_id"],
                    tenant["initial_balance_tokens"],
                    "purchase",
                    metadata={"source": "control_plane_sync"},
                    idempotency_key=f"control-plane-sync:wallet-seed:{tenant['company_id']}",
                )

        if sync_prune_mode == "deactivate":
            # Deactivation keeps historical rows while removing them from active auth.
            desired_keys = {tenant["openwebui_api_key"] for tenant in tenants}
            for existing_key in db.list_company_mapping_keys():
                if existing_key not in desired_keys:
                    db.deactivate_company_mapping(existing_key)

        for model in models:
            # Model registry rows define tokenizer identity, backend model name, and
            # optional default route for a logical model.
            db.upsert_model_registry_entry(
                logical_model_id=model["logical_model_id"],
                backend_model=model["backend_model"],
                tokenizer_repo=model["tokenizer_repo"],
                tokenizer_revision=model["tokenizer_revision"],
                model_policy_cap=model["model_policy_cap"],
                route_target=model["route_target"],
                is_active=1,
            )

        if sync_prune_mode == "deactivate":
            # Models removed from desired state stop being assignable without erasing history.
            for logical_model_id in db.list_model_registry_ids():
                if logical_model_id not in model_ids:
                    db.deactivate_model_registry_entry(logical_model_id)

        for company_id, logical_model_id in sorted(desired_assignments):
            # Assignment rows can override route_target per company/model pair, which
            # is how one logical model can route to tenant-specific vLLM services.
            db.upsert_tenant_model_assignment(
                company_id,
                logical_model_id,
                route_target=desired_assignments[(company_id, logical_model_id)],
                is_active=1,
            )

        if sync_prune_mode == "deactivate":
            # Removed assignments immediately stop access for that company/model pair.
            for company_id, logical_model_id in db.list_tenant_model_assignment_keys():
                if (company_id, logical_model_id) not in desired_assignments:
                    db.deactivate_tenant_model_assignment(company_id, logical_model_id)

        desired_state_sha256 = hashlib.sha256(
            json.dumps({"models": models, "tenants": tenants, "sync_prune_mode": sync_prune_mode}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        # Store a fingerprint of the normalized input so operators can tell exactly
        # which desired-state version last synced.
        db.record_control_plane_sync_run(
            desired_state_sha256=desired_state_sha256,
            sync_prune_mode=sync_prune_mode,
            tenant_count=len(tenants),
            model_count=len(models),
            assignment_count=len(desired_assignments),
            metadata={
                "company_ids": [tenant["company_id"] for tenant in tenants],
                "logical_model_ids": sorted(model_ids),
            },
        )
    finally:
        db.close()

    print(
        f"control plane sync complete; tenants={len(tenants)} models={len(models)} assignments={len(desired_assignments)}"
    )


if __name__ == "__main__":
    main()
