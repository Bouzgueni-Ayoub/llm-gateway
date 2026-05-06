from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class GatewayMetrics:
    """Collects gateway counters in memory and renders Prometheus text output."""

    LATENCY_BUCKETS_MS = (100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(self) -> None:
        self._lock = Lock()
        self._request_totals: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        self._latency_buckets: dict[tuple[str, str, str, str, str, str], int] = defaultdict(int)
        self._latency_sum_ms: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        self._latency_count: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
        self._estimation_totals: dict[tuple[str, str, str], int] = defaultdict(int)

    def observe_estimation(self, company_id: str, logical_model_id: str, method: str) -> None:
        with self._lock:
            self._estimation_totals[(company_id, logical_model_id, method)] += 1

    def observe_request(
        self,
        company_id: str,
        logical_model_id: str,
        path: str,
        stream: bool,
        status: str,
        latency_ms: int,
    ) -> None:
        labels = (company_id, logical_model_id, path, "true" if stream else "false", status)
        with self._lock:
            self._request_totals[labels] += 1
            self._latency_sum_ms[labels] += max(latency_ms, 0)
            self._latency_count[labels] += 1
            for bucket in self.LATENCY_BUCKETS_MS:
                if latency_ms <= bucket:
                    self._latency_buckets[labels + (str(bucket),)] += 1
            self._latency_buckets[labels + ("+Inf",)] += 1

    def render(self, db: Any) -> str:
        with self._lock:
            request_totals = dict(self._request_totals)
            latency_buckets = dict(self._latency_buckets)
            latency_sum_ms = dict(self._latency_sum_ms)
            latency_count = dict(self._latency_count)
            estimation_totals = dict(self._estimation_totals)

        wallet_rows = db.list_wallets()
        usage_rows = db.list_usage_events()
        ledger_rows = db.summarize_wallet_ledger()

        lines: list[str] = []

        self._append_header(
            lines,
            "gateway_requests_total",
            "counter",
            "Gateway requests observed after final billing or release status is known.",
        )
        for labels, value in sorted(request_totals.items()):
            lines.append(
                self._metric_line(
                    "gateway_requests_total",
                    {
                        "company_id": labels[0],
                        "logical_model_id": labels[1],
                        "path": labels[2],
                        "stream": labels[3],
                        "status": labels[4],
                    },
                    value,
                )
            )

        self._append_header(
            lines,
            "gateway_request_latency_ms",
            "histogram",
            "Gateway request latency in milliseconds for completed request lifecycles.",
        )
        for labels in sorted(latency_sum_ms):
            metric_labels = {
                "company_id": labels[0],
                "logical_model_id": labels[1],
                "path": labels[2],
                "stream": labels[3],
                "status": labels[4],
            }
            for bucket in (*[str(item) for item in self.LATENCY_BUCKETS_MS], "+Inf"):
                lines.append(
                    self._metric_line(
                        "gateway_request_latency_ms_bucket",
                        {**metric_labels, "le": bucket},
                        latency_buckets.get(labels + (bucket,), 0),
                    )
                )
            lines.append(self._metric_line("gateway_request_latency_ms_sum", metric_labels, latency_sum_ms[labels]))
            lines.append(self._metric_line("gateway_request_latency_ms_count", metric_labels, latency_count[labels]))

        self._append_header(
            lines,
            "gateway_estimation_method_total",
            "counter",
            "Prompt estimation method usage by company and logical model.",
        )
        for labels, value in sorted(estimation_totals.items()):
            lines.append(
                self._metric_line(
                    "gateway_estimation_method_total",
                    {
                        "company_id": labels[0],
                        "logical_model_id": labels[1],
                        "method": labels[2],
                    },
                    value,
                )
            )

        self._append_header(lines, "gateway_wallet_available_tokens", "gauge", "Current spendable prepaid balance.")
        self._append_header(lines, "gateway_wallet_reserved_tokens", "gauge", "Current reserved token balance.")
        self._append_header(lines, "gateway_wallet_low_balance_threshold", "gauge", "Configured wallet alert threshold.")
        self._append_header(lines, "gateway_wallet_low_balance", "gauge", "Whether available balance is at or below threshold.")
        for row in wallet_rows:
            company_id = str(row["company_id"])
            labels = {"company_id": company_id}
            available_tokens = int(row["available_tokens"])
            reserved_tokens = int(row["reserved_tokens"])
            threshold = int(row.get("low_balance_threshold") or 0)
            lines.append(self._metric_line("gateway_wallet_available_tokens", labels, available_tokens))
            lines.append(self._metric_line("gateway_wallet_reserved_tokens", labels, reserved_tokens))
            lines.append(self._metric_line("gateway_wallet_low_balance_threshold", labels, threshold))
            is_low_balance = 1 if threshold > 0 and available_tokens <= threshold else 0
            lines.append(self._metric_line("gateway_wallet_low_balance", labels, is_low_balance))

        usage_counts: dict[tuple[str, str], int] = defaultdict(int)
        oldest_pending_by_company: dict[str, int] = {}
        now = datetime.now(timezone.utc)
        for row in usage_rows:
            company_id = str(row["company_id"])
            status = str(row["status"])
            usage_counts[(company_id, status)] += 1
            if status != "reconciliation_pending":
                continue
            updated_at = row.get("updated_at") or row.get("created_at")
            try:
                age_seconds = max(int((now - datetime.fromisoformat(str(updated_at))).total_seconds()), 0)
            except Exception:
                age_seconds = 0
            current = oldest_pending_by_company.get(company_id, 0)
            oldest_pending_by_company[company_id] = max(current, age_seconds)

        self._append_header(
            lines,
            "gateway_usage_events_current",
            "gauge",
            "Current usage event count by company and lifecycle status.",
        )
        for labels, value in sorted(usage_counts.items()):
            lines.append(
                self._metric_line(
                    "gateway_usage_events_current",
                    {"company_id": labels[0], "status": labels[1]},
                    value,
                )
            )

        self._append_header(
            lines,
            "gateway_reconciliation_pending_oldest_age_seconds",
            "gauge",
            "Age in seconds of the oldest reconciliation_pending request per company.",
        )
        for company_id, value in sorted(oldest_pending_by_company.items()):
            lines.append(
                self._metric_line(
                    "gateway_reconciliation_pending_oldest_age_seconds",
                    {"company_id": company_id},
                    value,
                )
            )

        self._append_header(
            lines,
            "gateway_wallet_ledger_entries_total",
            "gauge",
            "Current append-only wallet ledger row count by company and entry type.",
        )
        self._append_header(
            lines,
            "gateway_wallet_ledger_tokens_total",
            "gauge",
            "Absolute token movement captured in wallet ledger rows by company and entry type.",
        )
        for row in ledger_rows:
            labels = {"company_id": str(row["company_id"]), "entry_type": str(row["entry_type"])}
            lines.append(self._metric_line("gateway_wallet_ledger_entries_total", labels, int(row["entry_count"])))
            lines.append(self._metric_line("gateway_wallet_ledger_tokens_total", labels, int(row["token_volume"])))

        return "\n".join(lines) + "\n"

    def _append_header(self, lines: list[str], name: str, metric_type: str, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")

    def _metric_line(self, name: str, labels: dict[str, str], value: int | float) -> str:
        if not labels:
            return f"{name} {value}"
        rendered = ",".join(f'{key}="{self._escape(value_text)}"' for key, value_text in sorted(labels.items()))
        return f"{name}{{{rendered}}} {value}"

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
