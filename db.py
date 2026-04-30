import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any

from fastapi import HTTPException

from config import (
    LEDGER_STATUS_COMMITTED,
    LEDGER_STATUS_RECONCILED,
    ModelConfig,
    USAGE_STATUS_PENDING,
    bool_to_int,
    db_path_from_url,
    safe_json_dumps,
    utcnow,
)


# OpenAI-style clients expect structured failures under a top-level `error`
# object. The gateway uses this helper so quota failures remain readable for
# OpenAI-compatible clients.
def build_openai_error_detail(message: str, *, error_type: str, error_code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": error_code,
        }
    }


# Reservation can make a wallet look empty while requests are still in flight.
# This helper chooses a clearer error code when retrying may succeed after
# another request settles or releases its reserved tokens.
def build_prepaid_quota_error_detail(
    *,
    required_tokens: int,
    available_tokens: int,
    reserved_tokens: int,
    exhausted_message: str,
) -> dict[str, Any]:
    if reserved_tokens > 0 and (available_tokens + reserved_tokens) >= required_tokens:
        return build_openai_error_detail(
            "Allowance is temporarily reserved by in-flight requests. Please retry in a few seconds.",
            error_type="insufficient_quota",
            error_code="temporarily_reserved",
        )
    return build_openai_error_detail(
        exhausted_message,
        error_type="insufficient_quota",
        error_code="insufficient_prepaid_balance",
    )


# `GatewayDB` is the only object that mutates gateway-owned tables. Route code
# calls this class for tenant mappings, model assignments, wallets, ledger rows,
# and usage events, while this class hides Postgres-vs-SQLite driver differences.
class GatewayDB:

    def __init__(self, database_url: str) -> None:
        # `DATABASE_URL` decides which database driver is used. Production uses
        # Postgres through psycopg; tests can use Python's built-in sqlite3.
        self.database_url = database_url
        self.backend = self._detect_backend(database_url)
        self.db_path = db_path_from_url(database_url) if self.backend == "sqlite" else None
        self._pg = None
        self._pg_dict_row = None

        if self.backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("psycopg is required for postgresql DATABASE_URL") from exc
            self._pg = psycopg
            self._pg_dict_row = dict_row

    def _detect_backend(self, database_url: str) -> str:
        # Keep backend detection explicit so unsupported URLs fail during startup.
        lowered = database_url.lower()
        if lowered.startswith("postgresql://") or lowered.startswith("postgres://"):
            return "postgres"
        if lowered.startswith("sqlite:///"):
            return "sqlite"
        raise ValueError("DATABASE_URL must start with postgresql://, postgres://, or sqlite:///")

    def _sql(self, query: str) -> str:
        # Query text is written with SQLite-style `?` placeholders. psycopg uses
        # `%s`, so Postgres queries are translated at the boundary.
        if self.backend == "postgres":
            return query.replace("?", "%s")
        return query

    @contextmanager
    def _connect(self):
        # `@contextmanager` turns this generator into a `with` block helper.
        # Callers receive a connection object and this helper closes it afterward.
        if self.backend == "postgres":
            assert self._pg is not None and self._pg_dict_row is not None
            conn = self._pg.connect(self.database_url, row_factory=self._pg_dict_row)
            try:
                yield conn
            finally:
                conn.close()
            return

        assert self.db_path is not None
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self):
        # All wallet mutations go through transactions. SQLite needs
        # `BEGIN IMMEDIATE` to take a write lock; Postgres relies on normal
        # transactional behavior plus row locks where needed.
        with self._connect() as conn:
            try:
                if self.backend == "sqlite":
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        # Return one database row as a plain dict so callers do not depend on a driver-specific row type.
        with self._connect() as conn:
            cur = conn.execute(self._sql(query), params)
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        # Return all matching rows as plain dicts for API responses and metrics.
        with self._connect() as conn:
            cur = conn.execute(self._sql(query), params)
            return [dict(row) for row in cur.fetchall()]

    def _execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        # Use this for simple writes that only need "execute inside one transaction".
        with self._transaction() as conn:
            conn.execute(self._sql(query), params)

    def init(self) -> None:
        # Create the gateway-owned schema if it is missing. These tables hold
        # active control-plane rows, wallet state, append-only ledger movement,
        # usage audit events, and sync history.
        if self.backend == "sqlite":
            assert self.db_path is not None
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.commit()

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS company_team_map (
                openwebui_api_key TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                team_id TEXT NOT NULL,
                litellm_url TEXT,
                default_max_tokens INTEGER,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_column("company_team_map", "company_id", "TEXT")
        self._ensure_column("company_team_map", "litellm_url", "TEXT")
        self._execute("CREATE INDEX IF NOT EXISTS idx_company_team_map_company ON company_team_map (company_id)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_company_team_map_active ON company_team_map (is_active)")

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry (
                logical_model_id TEXT PRIMARY KEY,
                backend_model TEXT NOT NULL,
                tokenizer_repo TEXT NOT NULL,
                tokenizer_revision TEXT NOT NULL,
                model_policy_cap BIGINT NOT NULL,
                route_target TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._execute("CREATE INDEX IF NOT EXISTS idx_model_registry_active ON model_registry (is_active)")

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_model_assignments (
                company_id TEXT NOT NULL,
                logical_model_id TEXT NOT NULL,
                route_target TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (company_id, logical_model_id)
            )
            """
        )
        self._ensure_column("tenant_model_assignments", "route_target", "TEXT")
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_tenant_model_assignments_active ON tenant_model_assignments (company_id, is_active)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_sync_runs (
                id TEXT PRIMARY KEY,
                desired_state_sha256 TEXT NOT NULL,
                sync_prune_mode TEXT NOT NULL,
                tenant_count BIGINT NOT NULL,
                model_count BIGINT NOT NULL,
                assignment_count BIGINT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_control_plane_sync_runs_created ON control_plane_sync_runs (created_at)"
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS wallet_accounts (
                company_id TEXT PRIMARY KEY,
                available_tokens BIGINT NOT NULL,
                reserved_tokens BIGINT NOT NULL,
                status TEXT NOT NULL,
                low_balance_threshold BIGINT NOT NULL DEFAULT 0,
                version BIGINT NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (available_tokens >= 0),
                CHECK (reserved_tokens >= 0)
            )
            """
        )

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS wallet_ledger (
                id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                request_id TEXT,
                entry_type TEXT NOT NULL,
                delta_tokens BIGINT NOT NULL,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wallet_ledger_idempotency ON wallet_ledger (idempotency_key)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_wallet_ledger_company_created ON wallet_ledger (company_id, created_at)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_wallet_ledger_request ON wallet_ledger (request_id)")

        self._execute(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                request_id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                user_id TEXT,
                logical_model_id TEXT NOT NULL,
                backend_model TEXT NOT NULL,
                tokenizer_repo TEXT NOT NULL,
                tokenizer_revision TEXT,
                estimated_prompt_tokens BIGINT,
                estimation_method TEXT,
                reserved_completion_tokens BIGINT,
                actual_prompt_tokens BIGINT,
                actual_completion_tokens BIGINT,
                actual_total_tokens BIGINT,
                latency_ms BIGINT,
                status TEXT NOT NULL,
                stream INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_column("usage_events", "estimation_method", "TEXT")
        self._execute("CREATE INDEX IF NOT EXISTS idx_usage_events_company_created ON usage_events (company_id, created_at)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_usage_events_status ON usage_events (status)")

    def _ensure_column(self, table_name: str, column_name: str, column_definition: str) -> None:
        # Small compatibility migration for databases created by older gateway
        # versions. Postgres and SQLite expose schema metadata through different
        # mechanisms, so each backend gets its own check.
        if self.backend == "postgres":
            row = self._fetchone(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = ? AND column_name = ?
                """,
                (table_name, column_name),
            )
        else:
            columns = self._fetchall(f"PRAGMA table_info({table_name})")
            row = next((column for column in columns if column.get("name") == column_name), None)
        if row:
            return
        self._execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def list_company_mappings(self) -> list[dict[str, Any]]:
        # Admin read path for tenant/company mappings. The API masks keys before returning them.
        return self._fetchall(
            """
            SELECT openwebui_api_key, company_id, company_name, team_id, litellm_url, default_max_tokens, is_active, created_at, updated_at
            FROM company_team_map
            ORDER BY updated_at DESC
            """
        )

    def get_company_mapping_by_openwebui_key(self, openwebui_api_key: str) -> dict[str, Any] | None:
        # Authentication starts here: the inbound gateway key maps to one active company row.
        row = self._fetchone(
            """
            SELECT openwebui_api_key, company_id, company_name, team_id, litellm_url, default_max_tokens, is_active, created_at, updated_at
            FROM company_team_map
            WHERE openwebui_api_key = ? AND is_active = 1
            """,
            (openwebui_api_key,),
        )
        if row and not row.get("company_id"):
            row["company_id"] = row["company_name"]
        return row

    def upsert_company_mapping(
        self,
        openwebui_api_key: str,
        company_id: str,
        company_name: str,
        team_id: str,
        litellm_url: str | None,
        default_max_tokens: int | None,
        is_active: int = 1,
    ) -> None:
        # Upsert makes sync idempotent: the same desired tenant can be applied
        # repeatedly without creating duplicate rows.
        now = utcnow()
        self._execute(
            """
            INSERT INTO company_team_map (
                openwebui_api_key, company_id, company_name, team_id, litellm_url, default_max_tokens, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(openwebui_api_key) DO UPDATE SET
                company_id = excluded.company_id,
                company_name = excluded.company_name,
                team_id = excluded.team_id,
                litellm_url = excluded.litellm_url,
                default_max_tokens = excluded.default_max_tokens,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (openwebui_api_key, company_id, company_name, team_id, litellm_url, default_max_tokens, is_active, now, now),
        )

    def list_company_mapping_keys(self) -> list[str]:
        return [
            row["openwebui_api_key"]
            for row in self._fetchall("SELECT openwebui_api_key FROM company_team_map")
            if row.get("openwebui_api_key")
        ]

    def deactivate_company_mapping(self, openwebui_api_key: str) -> None:
        # Deactivation preserves historical rows while preventing future authentication.
        self._execute(
            """
            UPDATE company_team_map
            SET is_active = 0, updated_at = ?
            WHERE openwebui_api_key = ?
            """,
            (utcnow(), openwebui_api_key),
        )

    def list_model_registry(self) -> list[dict[str, Any]]:
        return self._fetchall(
            """
            SELECT logical_model_id, backend_model, tokenizer_repo, tokenizer_revision, model_policy_cap, route_target, is_active, created_at, updated_at
            FROM model_registry
            ORDER BY logical_model_id ASC
            """
        )

    def list_model_registry_ids(self) -> list[str]:
        return [
            row["logical_model_id"]
            for row in self._fetchall("SELECT logical_model_id FROM model_registry")
            if row.get("logical_model_id")
        ]

    def upsert_model_registry_entry(
        self,
        logical_model_id: str,
        backend_model: str,
        tokenizer_repo: str,
        tokenizer_revision: str,
        model_policy_cap: int,
        route_target: str | None = None,
        is_active: int = 1,
    ) -> None:
        now = utcnow()
        self._execute(
            """
            INSERT INTO model_registry (
                logical_model_id, backend_model, tokenizer_repo, tokenizer_revision, model_policy_cap, route_target, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(logical_model_id) DO UPDATE SET
                backend_model = excluded.backend_model,
                tokenizer_repo = excluded.tokenizer_repo,
                tokenizer_revision = excluded.tokenizer_revision,
                model_policy_cap = excluded.model_policy_cap,
                route_target = excluded.route_target,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (
                logical_model_id,
                backend_model,
                tokenizer_repo,
                tokenizer_revision,
                model_policy_cap,
                route_target,
                is_active,
                now,
                now,
            ),
        )

    def deactivate_model_registry_entry(self, logical_model_id: str) -> None:
        self._execute(
            """
            UPDATE model_registry
            SET is_active = 0, updated_at = ?
            WHERE logical_model_id = ?
            """,
            (utcnow(), logical_model_id),
        )

    def list_tenant_model_assignments(self) -> list[dict[str, Any]]:
        return self._fetchall(
            """
            SELECT company_id, logical_model_id, route_target, is_active, created_at, updated_at
            FROM tenant_model_assignments
            ORDER BY company_id ASC, logical_model_id ASC
            """
        )

    def list_tenant_model_assignment_keys(self) -> list[tuple[str, str]]:
        return [
            (row["company_id"], row["logical_model_id"])
            for row in self._fetchall("SELECT company_id, logical_model_id FROM tenant_model_assignments")
            if row.get("company_id") and row.get("logical_model_id")
        ]

    def upsert_tenant_model_assignment(
        self,
        company_id: str,
        logical_model_id: str,
        route_target: str | None = None,
        is_active: int = 1,
    ) -> None:
        now = utcnow()
        self._execute(
            """
            INSERT INTO tenant_model_assignments (company_id, logical_model_id, route_target, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(company_id, logical_model_id) DO UPDATE SET
                route_target = excluded.route_target,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (company_id, logical_model_id, route_target, is_active, now, now),
        )

    def deactivate_tenant_model_assignment(self, company_id: str, logical_model_id: str) -> None:
        self._execute(
            """
            UPDATE tenant_model_assignments
            SET is_active = 0, updated_at = ?
            WHERE company_id = ? AND logical_model_id = ?
            """,
            (utcnow(), company_id, logical_model_id),
        )

    def get_model_config(self, logical_model_id: str) -> ModelConfig | None:
        row = self._fetchone(
            """
            SELECT logical_model_id, backend_model, tokenizer_repo, tokenizer_revision, model_policy_cap, route_target
            FROM model_registry
            WHERE logical_model_id = ? AND is_active = 1
            """,
            (logical_model_id,),
        )
        if row is None:
            return None
        return ModelConfig(
            logical_model_id=row["logical_model_id"],
            backend_model=row["backend_model"],
            tokenizer_repo=row["tokenizer_repo"],
            tokenizer_revision=row["tokenizer_revision"],
            route=row.get("route_target"),
            model_policy_cap=int(row["model_policy_cap"]),
            preload=False,
        )

    def get_company_model_config(self, company_id: str, logical_model_id: str) -> ModelConfig | None:
        row = self._fetchone(
            """
            SELECT mr.logical_model_id, mr.backend_model, mr.tokenizer_repo, mr.tokenizer_revision, mr.model_policy_cap, tma.route_target
            FROM model_registry mr
            INNER JOIN tenant_model_assignments tma
                ON tma.logical_model_id = mr.logical_model_id
            WHERE tma.company_id = ?
              AND tma.logical_model_id = ?
              AND mr.is_active = 1
              AND tma.is_active = 1
            """,
            (company_id, logical_model_id),
        )
        if row is None:
            return None
        return ModelConfig(
            logical_model_id=row["logical_model_id"],
            backend_model=row["backend_model"],
            tokenizer_repo=row["tokenizer_repo"],
            tokenizer_revision=row["tokenizer_revision"],
            route=row.get("route_target"),
            model_policy_cap=int(row["model_policy_cap"]),
            preload=False,
        )

    def list_openai_models_for_company(self, company_id: str) -> dict[str, Any]:
        rows = self._fetchall(
            """
            SELECT mr.logical_model_id, mr.backend_model, mr.tokenizer_repo, mr.tokenizer_revision, tma.route_target, mr.model_policy_cap
            FROM model_registry mr
            INNER JOIN tenant_model_assignments tma
                ON tma.logical_model_id = mr.logical_model_id
            WHERE tma.company_id = ? AND mr.is_active = 1 AND tma.is_active = 1
            ORDER BY mr.logical_model_id ASC
            """,
            (company_id,),
        )
        return {
            "object": "list",
            "data": [
                {
                    "id": row["logical_model_id"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "gateway",
                    "metadata": {
                        "backend_model": row["backend_model"],
                        "tokenizer_repo": row["tokenizer_repo"],
                        "tokenizer_revision": row["tokenizer_revision"],
                        "route": row.get("route_target"),
                        "model_policy_cap": int(row["model_policy_cap"]),
                    },
                }
                for row in rows
            ],
        }

    def record_control_plane_sync_run(
        self,
        *,
        desired_state_sha256: str,
        sync_prune_mode: str,
        tenant_count: int,
        model_count: int,
        assignment_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO control_plane_sync_runs (
                id, desired_state_sha256, sync_prune_mode, tenant_count, model_count, assignment_count, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                desired_state_sha256,
                sync_prune_mode,
                tenant_count,
                model_count,
                assignment_count,
                safe_json_dumps(metadata or {}),
                utcnow(),
            ),
        )

    def ensure_wallet_account(self, company_id: str, low_balance_threshold: int | None = None) -> bool:
        # A company mapping and a wallet are separate records. This method creates
        # a missing wallet with zero balance and optionally updates the alert threshold.
        now = utcnow()
        threshold_value = 0 if low_balance_threshold is None else low_balance_threshold
        with self._transaction() as conn:
            existing = conn.execute(
                self._sql(
                    """
                    SELECT company_id
                    FROM wallet_accounts
                    WHERE company_id = ?
                    """
                ),
                (company_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO wallet_accounts (
                            company_id, available_tokens, reserved_tokens, status, low_balance_threshold, version, created_at, updated_at
                        )
                        VALUES (?, 0, 0, 'active', ?, 0, ?, ?)
                        """
                    ),
                    (company_id, threshold_value, now, now),
                )
                return True

            if low_balance_threshold is not None:
                conn.execute(
                    self._sql(
                        """
                        UPDATE wallet_accounts
                        SET low_balance_threshold = ?, updated_at = ?
                        WHERE company_id = ?
                        """
                    ),
                    (low_balance_threshold, now, company_id),
                )
            return False

    def _lock_wallet_account(self, conn: Any, company_id: str) -> dict[str, Any]:
        # Wallet balance changes must be serialized per company. Postgres uses
        # `FOR UPDATE` to lock the row; SQLite already holds a write lock from
        # `BEGIN IMMEDIATE` in `_transaction`.
        if self.backend == "postgres":
            cur = conn.execute(
                self._sql(
                    """
                    SELECT company_id, available_tokens, reserved_tokens, status, low_balance_threshold, version, created_at, updated_at
                    FROM wallet_accounts
                    WHERE company_id = ?
                    FOR UPDATE
                    """
                ),
                (company_id,),
            )
        else:
            cur = conn.execute(
                self._sql(
                    """
                    SELECT company_id, available_tokens, reserved_tokens, status, low_balance_threshold, version, created_at, updated_at
                    FROM wallet_accounts
                    WHERE company_id = ?
                    """
                ),
                (company_id,),
            )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Wallet not found")
        return dict(row)

    def adjust_wallet_balance(
        self,
        company_id: str,
        delta_tokens: int,
        entry_type: str,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        # Operator-driven balance changes update the wallet and append a ledger
        # row in the same transaction. The optional idempotency key lets sync
        # safely seed a newly-created wallet without duplicating the seed entry.
        if delta_tokens == 0:
            return
        metadata_json = safe_json_dumps(metadata or {})
        now = utcnow()
        with self._transaction() as conn:
            if idempotency_key:
                existing = conn.execute(
                    self._sql("SELECT idempotency_key FROM wallet_ledger WHERE idempotency_key = ?"),
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return
            # Read and lock the wallet before checking for negative balances.
            wallet = self._lock_wallet_account(conn, company_id)
            available_tokens = int(wallet["available_tokens"])
            new_available = available_tokens + delta_tokens
            if new_available < 0:
                raise HTTPException(status_code=409, detail="Wallet balance cannot go negative")
            conn.execute(
                self._sql(
                    """
                    UPDATE wallet_accounts
                    SET available_tokens = ?, version = version + 1, updated_at = ?
                    WHERE company_id = ?
                    """
                ),
                (new_available, now, company_id),
            )
            conn.execute(
                self._sql(
                    """
                    INSERT INTO wallet_ledger (id, company_id, request_id, entry_type, delta_tokens, status, idempotency_key, metadata_json, created_at)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)
                    """
                ),
                (str(uuid.uuid4()), company_id, entry_type, delta_tokens, LEDGER_STATUS_COMMITTED, idempotency_key, metadata_json, now),
            )

    def list_wallets(self) -> list[dict[str, Any]]:
        # Admin and metrics read path for current wallet balances.
        return self._fetchall(
            """
            SELECT company_id, available_tokens, reserved_tokens, status, low_balance_threshold, version, created_at, updated_at
            FROM wallet_accounts
            ORDER BY updated_at DESC
            """
        )

    def get_wallet(self, company_id: str) -> dict[str, Any] | None:
        # Request admission reads this snapshot before attempting a reservation.
        return self._fetchone(
            """
            SELECT company_id, available_tokens, reserved_tokens, status, low_balance_threshold, version, created_at, updated_at
            FROM wallet_accounts
            WHERE company_id = ?
            """,
            (company_id,),
        )

    def create_pending_usage_event(
        self,
        request_id: str,
        company_id: str,
        user_id: str | None,
        model_config: ModelConfig,
        estimated_prompt_tokens: int,
        estimation_method: str,
        reserved_completion_tokens: int,
        stream: bool,
    ) -> None:
        # The usage row is created immediately after reservation and before the
        # upstream call. That guarantees every admitted request has an audit row
        # even if the model runtime later fails.
        now = utcnow()
        self._execute(
            """
            INSERT INTO usage_events (
                request_id, company_id, user_id, logical_model_id, backend_model, tokenizer_repo, tokenizer_revision,
                estimated_prompt_tokens, estimation_method, reserved_completion_tokens, actual_prompt_tokens, actual_completion_tokens,
                actual_total_tokens, latency_ms, status, stream, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO NOTHING
            """,
            (
                request_id,
                company_id,
                user_id,
                model_config.logical_model_id,
                model_config.backend_model,
                model_config.tokenizer_repo,
                model_config.tokenizer_revision,
                estimated_prompt_tokens,
                estimation_method,
                reserved_completion_tokens,
                USAGE_STATUS_PENDING,
                bool_to_int(stream),
                now,
                now,
            ),
        )

    def update_usage_event(
        self,
        request_id: str,
        *,
        actual_prompt_tokens: int | None = None,
        actual_completion_tokens: int | None = None,
        actual_total_tokens: int | None = None,
        latency_ms: int | None = None,
        status: str | None = None,
    ) -> None:
        # Usage events are filled in over time: pending at reservation, then
        # completed/reconciled/failed later. `None` means "keep the current value".
        current = self._fetchone(
            """
            SELECT request_id, actual_prompt_tokens, actual_completion_tokens, actual_total_tokens, latency_ms, status
            FROM usage_events
            WHERE request_id = ?
            """,
            (request_id,),
        )
        if current is None:
            return
        self._execute(
            """
            UPDATE usage_events
            SET actual_prompt_tokens = ?,
                actual_completion_tokens = ?,
                actual_total_tokens = ?,
                latency_ms = ?,
                status = ?,
                updated_at = ?
            WHERE request_id = ?
            """,
            (
                actual_prompt_tokens if actual_prompt_tokens is not None else current.get("actual_prompt_tokens"),
                actual_completion_tokens if actual_completion_tokens is not None else current.get("actual_completion_tokens"),
                actual_total_tokens if actual_total_tokens is not None else current.get("actual_total_tokens"),
                latency_ms if latency_ms is not None else current.get("latency_ms"),
                status or current.get("status"),
                utcnow(),
                request_id,
            ),
        )

    def get_usage_event(self, request_id: str) -> dict[str, Any] | None:
        # Fetch one request audit row by the gateway-generated request id.
        return self._fetchone("SELECT * FROM usage_events WHERE request_id = ?", (request_id,))

    def list_usage_events(self) -> list[dict[str, Any]]:
        # Admin read path for recent request audit and reconciliation state.
        return self._fetchall("SELECT * FROM usage_events ORDER BY created_at DESC")

    def summarize_wallet_ledger(self) -> list[dict[str, Any]]:
        # Metrics need aggregate ledger counts, but billing decisions still use
        # the raw wallet and usage rows.
        return self._fetchall(
            """
            SELECT company_id, entry_type, COUNT(*) AS entry_count, COALESCE(SUM(ABS(delta_tokens)), 0) AS token_volume
            FROM wallet_ledger
            GROUP BY company_id, entry_type
            ORDER BY company_id, entry_type
            """
        )

    def reserve_tokens(
        self,
        company_id: str,
        request_id: str,
        estimated_prompt_tokens: int,
        reserved_completion_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        # Admission control happens by moving estimated prompt plus allowed
        # completion tokens from `available_tokens` to `reserved_tokens`.
        # Final settlement later compares this reservation with vLLM actual usage.
        reserved_total_tokens = estimated_prompt_tokens + reserved_completion_tokens
        if reserved_total_tokens <= 0:
            raise HTTPException(status_code=400, detail="Reservation must be positive")

        now = utcnow()
        metadata_json = safe_json_dumps(metadata or {})
        idempotency_key = f"reserve:{request_id}"

        with self._transaction() as conn:
            # The ledger idempotency key prevents a retry from reserving the same
            # request twice.
            existing = conn.execute(
                self._sql("SELECT idempotency_key FROM wallet_ledger WHERE idempotency_key = ?"),
                (idempotency_key,),
            ).fetchone()
            if existing:
                return reserved_total_tokens

            wallet = self._lock_wallet_account(conn, company_id)
            available_tokens = int(wallet["available_tokens"])
            reserved_tokens = int(wallet["reserved_tokens"])
            if available_tokens < reserved_total_tokens:
                # Do not call the model runtime if the wallet cannot cover the reservation.
                raise HTTPException(
                    status_code=402,
                    detail=build_prepaid_quota_error_detail(
                        required_tokens=reserved_total_tokens,
                        available_tokens=available_tokens,
                        reserved_tokens=reserved_tokens,
                        exhausted_message="Insufficient prepaid balance for reservation",
                    ),
                )

            conn.execute(
                self._sql(
                    """
                    UPDATE wallet_accounts
                    SET available_tokens = ?, reserved_tokens = ?, version = version + 1, updated_at = ?
                    WHERE company_id = ?
                    """
                ),
                (
                    available_tokens - reserved_total_tokens,
                    reserved_tokens + reserved_total_tokens,
                    now,
                    company_id,
                ),
            )
            conn.execute(
                self._sql(
                    """
                    INSERT INTO wallet_ledger (id, company_id, request_id, entry_type, delta_tokens, status, idempotency_key, metadata_json, created_at)
                    VALUES (?, ?, ?, 'reserve', ?, ?, ?, ?, ?)
                    """
                ),
                (
                    str(uuid.uuid4()),
                    company_id,
                    request_id,
                    -reserved_total_tokens,
                    LEDGER_STATUS_COMMITTED,
                    idempotency_key,
                    metadata_json,
                    now,
                ),
            )

        return reserved_total_tokens

    def release_full_reservation(self, company_id: str, request_id: str, reserved_total_tokens: int, metadata: dict[str, Any] | None = None) -> None:
        # If no billable inference completed, the full reservation returns to
        # available balance and a release ledger row records the movement.
        if reserved_total_tokens <= 0:
            return
        idempotency_key = f"release:{request_id}:full"
        now = utcnow()
        metadata_json = safe_json_dumps(metadata or {})

        with self._transaction() as conn:
            # Do not release the same reservation twice if error handling is retried.
            existing = conn.execute(
                self._sql("SELECT idempotency_key FROM wallet_ledger WHERE idempotency_key = ?"),
                (idempotency_key,),
            ).fetchone()
            if existing:
                return
            wallet = self._lock_wallet_account(conn, company_id)
            available_tokens = int(wallet["available_tokens"])
            reserved_tokens = int(wallet["reserved_tokens"])
            if reserved_tokens < reserved_total_tokens:
                raise HTTPException(status_code=409, detail="Reserved balance is inconsistent")

            conn.execute(
                self._sql(
                    """
                    UPDATE wallet_accounts
                    SET available_tokens = ?, reserved_tokens = ?, version = version + 1, updated_at = ?
                    WHERE company_id = ?
                    """
                ),
                (available_tokens + reserved_total_tokens, reserved_tokens - reserved_total_tokens, now, company_id),
            )
            conn.execute(
                self._sql(
                    """
                    INSERT INTO wallet_ledger (id, company_id, request_id, entry_type, delta_tokens, status, idempotency_key, metadata_json, created_at)
                    VALUES (?, ?, ?, 'release', ?, ?, ?, ?, ?)
                    """
                ),
                (str(uuid.uuid4()), company_id, request_id, reserved_total_tokens, LEDGER_STATUS_COMMITTED, idempotency_key, metadata_json, now),
            )

    def finalize_request(
        self,
        company_id: str,
        request_id: str,
        reserved_total_tokens: int,
        actual_total_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # Final settlement is based on vLLM-reported actual usage. If actual
        # usage is below the reservation, unused tokens return to available
        # balance. If actual usage is above the reservation, the extra is charged
        # from still-available balance.
        if actual_total_tokens < 0:
            raise HTTPException(status_code=500, detail="Actual usage cannot be negative")
        release_tokens = max(reserved_total_tokens - actual_total_tokens, 0)
        extra_charge_tokens = max(actual_total_tokens - reserved_total_tokens, 0)
        finalize_idempotency = f"finalize:{request_id}"
        release_idempotency = f"release:{request_id}:delta"
        now = utcnow()
        metadata_json = safe_json_dumps(metadata or {})

        with self._transaction() as conn:
            # A finalized request should never be charged twice.
            existing = conn.execute(
                self._sql("SELECT idempotency_key FROM wallet_ledger WHERE idempotency_key = ?"),
                (finalize_idempotency,),
            ).fetchone()
            if existing:
                return

            wallet = self._lock_wallet_account(conn, company_id)
            available_tokens = int(wallet["available_tokens"])
            reserved_tokens = int(wallet["reserved_tokens"])
            if reserved_tokens < reserved_total_tokens:
                raise HTTPException(status_code=409, detail="Reserved balance is inconsistent")
            if available_tokens < extra_charge_tokens:
                # The gateway never lets final settlement make a prepaid wallet negative.
                raise HTTPException(status_code=409, detail="Actual usage exceeded reserved balance and remaining prepaid balance")

            conn.execute(
                self._sql(
                    """
                    UPDATE wallet_accounts
                    SET available_tokens = ?, reserved_tokens = ?, version = version + 1, updated_at = ?
                    WHERE company_id = ?
                    """
                ),
                (
                    available_tokens - extra_charge_tokens + release_tokens,
                    reserved_tokens - reserved_total_tokens,
                    now,
                    company_id,
                ),
            )
            conn.execute(
                self._sql(
                    """
                    INSERT INTO wallet_ledger (id, company_id, request_id, entry_type, delta_tokens, status, idempotency_key, metadata_json, created_at)
                    VALUES (?, ?, ?, 'finalize', ?, ?, ?, ?, ?)
                    """
                ),
                (
                    str(uuid.uuid4()),
                    company_id,
                    request_id,
                    -extra_charge_tokens,
                    LEDGER_STATUS_RECONCILED,
                    finalize_idempotency,
                    metadata_json,
                    now,
                ),
            )
            if release_tokens > 0:
                # A separate release row makes unused-reservation returns visible
                # in the append-only ledger.
                conn.execute(
                    self._sql(
                        """
                        INSERT INTO wallet_ledger (id, company_id, request_id, entry_type, delta_tokens, status, idempotency_key, metadata_json, created_at)
                        VALUES (?, ?, ?, 'release', ?, ?, ?, ?, ?)
                        """
                    ),
                    (str(uuid.uuid4()), company_id, request_id, release_tokens, LEDGER_STATUS_RECONCILED, release_idempotency, metadata_json, now),
                )
