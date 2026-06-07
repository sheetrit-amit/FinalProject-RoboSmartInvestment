"""
Usage logging to BigQuery for RoboSmartInvestment.

Owns the telemetry tables, their schemas, and the async, fault-tolerant write
path. Two tables, linked 1:1 by ``event_id``:

  * ``usage_logs``    — one structured row per /chat request (params, portfolio,
                        latency, token usage). No free text.
  * ``usage_outputs`` — the raw model response text plus the requested risk /
                        money / stock-count for that same request.

Design: docs/superpowers/specs/2026-06-07-usage-logging-bigquery-design.md
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Set

from google.cloud import bigquery

logger = logging.getLogger(__name__)

PROJECT_ID = "pro-visitor-429015-f5"
USAGE_TABLE = f"{PROJECT_ID}.StockData.usage_logs"
OUTPUTS_TABLE = f"{PROJECT_ID}.StockData.usage_outputs"

# Structured per-request telemetry — keep in sync with the design doc.
USAGE_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("user_message", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("response_text", "STRING", mode="NULLABLE"),
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
            bigquery.SchemaField("explanation", "STRING", mode="NULLABLE"),
        ],
    ),
]

# Raw model output text + the requested parameters, linked to usage_logs by event_id.
OUTPUTS_SCHEMA: List[bigquery.SchemaField] = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("event_time", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("response_text", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("requested_risk", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("requested_budget", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("requested_currency", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("requested_top_k", "INT64", mode="NULLABLE"),
]

# Table id -> schema. Drives ensure_table / create-if-not-exists.
_SCHEMAS: Dict[str, List[bigquery.SchemaField]] = {
    USAGE_TABLE: USAGE_SCHEMA,
    OUTPUTS_TABLE: OUTPUTS_SCHEMA,
}

_RETRIES = 3
_BASE_BACKOFF = 0.5  # seconds; doubles each attempt (0.5s, 1s, 2s)

_ready: Set[str] = set()
_ready_lock = threading.Lock()


def ensure_table(client: bigquery.Client, table: str = USAGE_TABLE) -> bool:
    """
    Create ``table`` if it does not exist (idempotent; the create call runs at
    most once per process per table). Never raises — returns True when ready,
    False if BigQuery was unavailable.
    """
    if table in _ready:
        return True
    with _ready_lock:
        if table in _ready:
            return True
        try:
            client.create_table(
                bigquery.Table(table, schema=_SCHEMAS[table]), exists_ok=True
            )
            _ready.add(table)
            logger.info("telemetry table ready: %s", table)
            return True
        except Exception as exc:
            logger.error("Failed to ensure table %s: %s", table, exc)
            return False


def _insert(client: bigquery.Client, table: str, record: Dict[str, Any]) -> bool:
    """
    Stream one record into ``table`` with retry + exponential backoff. Returns
    True on success. Never raises — logs and drops the record on final failure.
    """
    if not ensure_table(client, table):
        logger.error("%s unavailable — dropping record %s", table, record.get("event_id"))
        return False

    for attempt in range(_RETRIES):
        try:
            errors = client.insert_rows_json(table, [record])
            if not errors:
                return True
            logger.warning(
                "%s insert returned errors (attempt %d/%d): %s",
                table, attempt + 1, _RETRIES, errors,
            )
        except Exception as exc:
            logger.warning(
                "%s insert raised (attempt %d/%d): %s",
                table, attempt + 1, _RETRIES, exc,
            )

        if attempt < _RETRIES - 1:
            time.sleep(_BASE_BACKOFF * (2 ** attempt))

    logger.error(
        "%s insert failed after %d attempts — dropping record %s",
        table, _RETRIES, record.get("event_id"),
    )
    return False


def log_usage(client: bigquery.Client, record: Dict[str, Any]) -> bool:
    """Write a structured row to ``usage_logs``."""
    return _insert(client, USAGE_TABLE, record)


def log_output(client: bigquery.Client, record: Dict[str, Any]) -> bool:
    """Write a raw-output row to ``usage_outputs`` (linked by event_id)."""
    return _insert(client, OUTPUTS_TABLE, record)


def _log_async(
    get_client: Callable[[], bigquery.Client],
    table: str,
    record: Dict[str, Any],
) -> None:
    """Fire-and-forget write in a daemon thread — zero added request latency."""
    def _worker() -> None:
        try:
            _insert(get_client(), table, record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("telemetry worker failed for %s: %s", table, exc)

    threading.Thread(target=_worker, name="usage-logger", daemon=True).start()


def log_usage_async(
    get_client: Callable[[], bigquery.Client], record: Dict[str, Any]
) -> None:
    """Async variant of :func:`log_usage`."""
    _log_async(get_client, USAGE_TABLE, record)


def log_output_async(
    get_client: Callable[[], bigquery.Client], record: Dict[str, Any]
) -> None:
    """Async variant of :func:`log_output`."""
    _log_async(get_client, OUTPUTS_TABLE, record)
