"""
Usage logging to BigQuery for RoboSmartInvestment.

Owns the ``usage_logs`` table name, schema, and the async, fault-tolerant write
path. One row is written per ``/chat`` request (success or error).

Design: docs/superpowers/specs/2026-06-07-usage-logging-bigquery-design.md
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List

from google.cloud import bigquery

logger = logging.getLogger(__name__)

PROJECT_ID = "pro-visitor-429015-f5"
USAGE_TABLE = f"{PROJECT_ID}.StockData.usage_logs"

# Table schema — keep in sync with the design doc.
USAGE_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("mode", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("status", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("latency_ms", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("model_used", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("overloaded", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("prompt_tokens", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("completion_tokens", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("total_tokens", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("llm_calls", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("risk", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("budget", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("currency", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("top_k", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("stocks_delivered", "INT64", mode="NULLABLE"),
    bigquery.SchemaField(
        "holdings",
        "RECORD",
        mode="REPEATED",
        fields=[
            bigquery.SchemaField("ticker", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("weight", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("fundamental_score", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("technical_score", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("label", "STRING", mode="NULLABLE"),
        ],
    ),
]

_RETRIES = 3
_BASE_BACKOFF = 0.5  # seconds; doubles each attempt (0.5s, 1s, 2s)

_table_ready = False
_table_lock = threading.Lock()


def ensure_table(client: bigquery.Client) -> bool:
    """
    Create the ``usage_logs`` table if it does not exist.

    Idempotent and runs the create call at most once per process (guarded by a
    flag + lock). Never raises — returns True when the table is ready, False if
    BigQuery was unavailable.
    """
    global _table_ready
    if _table_ready:
        return True
    with _table_lock:
        if _table_ready:
            return True
        try:
            table = bigquery.Table(USAGE_TABLE, schema=USAGE_SCHEMA)
            client.create_table(table, exists_ok=True)
            _table_ready = True
            logger.info("usage_logs table ready: %s", USAGE_TABLE)
            return True
        except Exception as exc:
            logger.error("Failed to ensure usage_logs table: %s", exc)
            return False


def log_usage(client: bigquery.Client, record: Dict[str, Any]) -> bool:
    """
    Stream a single usage record into BigQuery with retry + exponential backoff.

    Returns True on success. Never raises — logs and drops the record on final
    failure. Treats both raised exceptions and returned row-level errors as
    retryable failures.
    """
    if not ensure_table(client):
        logger.error(
            "usage_logs unavailable — dropping record %s", record.get("event_id")
        )
        return False

    for attempt in range(_RETRIES):
        try:
            errors = client.insert_rows_json(USAGE_TABLE, [record])
            if not errors:
                return True
            logger.warning(
                "usage_logs insert returned errors (attempt %d/%d): %s",
                attempt + 1, _RETRIES, errors,
            )
        except Exception as exc:
            logger.warning(
                "usage_logs insert raised (attempt %d/%d): %s",
                attempt + 1, _RETRIES, exc,
            )

        if attempt < _RETRIES - 1:
            time.sleep(_BASE_BACKOFF * (2 ** attempt))

    logger.error(
        "usage_logs insert failed after %d attempts — dropping record %s",
        _RETRIES, record.get("event_id"),
    )
    return False


def log_usage_async(
    get_client: Callable[[], bigquery.Client],
    record: Dict[str, Any],
) -> None:
    """
    Fire-and-forget write: run the insert in a daemon thread so the request path
    incurs zero added latency. The BigQuery client is resolved lazily inside the
    worker via ``get_client`` so a slow/unavailable client never blocks callers.
    """
    def _worker() -> None:
        try:
            client = get_client()
            log_usage(client, record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("usage logging worker failed: %s", exc)

    threading.Thread(target=_worker, name="usage-logger", daemon=True).start()
