"""
Idempotent BigQuery schema migration for the usage-telemetry tables.

Why this exists
---------------
``usage_logger.ensure_table()`` only runs ``create_table(exists_ok=True)``: it
creates a *missing* table but never alters an *existing* one. So when new columns
were added to ``USAGE_SCHEMA`` in code (commit 77f7902 added ``user_message``,
``response_text`` and ``holdings.explanation``), the already-existing
``usage_logs`` table stayed on the old schema. Streaming inserts of the new record
shape were then rejected by BigQuery with "no such field: ..." and dropped after
the retry budget — which is why real build / conversational events stopped landing
in ``usage_logs`` while ``usage_outputs`` (whose record still matched its table)
kept filling.

What it does
------------
Reconciles every live telemetry table with the schema declared in
``usage_logger``, adding any missing fields (including nested RECORD subfields).
Additive only and idempotent: it never drops, reorders, or retypes a column, and
re-running it is a safe no-op once the tables are in sync. Insert mapping is by
field *name*, so appended columns work regardless of position.

Note: BigQuery's streaming-insert path caches a table's schema for a few minutes,
so immediately after a migration the new fields (nested ones especially) may still
be rejected for a short window before the cache refreshes.

Usage:
    python scripts/migrate_usage_schema.py
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
)

from google.cloud import bigquery  # noqa: E402

import usage_logger as ul  # noqa: E402


def _paths(fields, prefix=""):
    """Flatten a schema into dotted field paths (for change detection)."""
    out = []
    for f in fields:
        p = prefix + f.name
        out.append(p)
        if f.field_type == "RECORD":
            out.extend(_paths(f.fields, p + "."))
    return out


def _reconcile(live, declared):
    """Return live schema + any declared fields missing by name, recursing into
    RECORD subfields. Additive only; REQUIRED additions are demoted to NULLABLE
    because BigQuery cannot add a REQUIRED column to an existing table."""
    live_by_name = {f.name: f for f in live}
    merged = []
    for lf in live:
        df = next((d for d in declared if d.name == lf.name), None)
        if df is not None and lf.field_type == "RECORD":
            merged.append(
                bigquery.SchemaField(
                    lf.name, lf.field_type, mode=lf.mode,
                    fields=tuple(_reconcile(lf.fields, df.fields)),
                )
            )
        else:
            merged.append(lf)
    for df in declared:
        if df.name not in live_by_name:
            mode = "NULLABLE" if df.mode == "REQUIRED" else df.mode
            merged.append(
                bigquery.SchemaField(df.name, df.field_type, mode=mode, fields=df.fields)
            )
    return merged


def migrate(client, table_id, declared_schema):
    tbl = client.get_table(table_id)
    before = set(_paths(tbl.schema))
    new_schema = _reconcile(tbl.schema, declared_schema)
    added = sorted(set(_paths(new_schema)) - before)
    if not added:
        print(f"  {table_id}: up to date ({len(before)} fields)")
        return
    tbl.schema = new_schema
    client.update_table(tbl, ["schema"])
    print(f"  {table_id}: added {added}")


def main():
    client = bigquery.Client(project=ul.PROJECT_ID)
    print("Reconciling usage-telemetry tables to declared schemas:")
    migrate(client, ul.USAGE_TABLE, ul.USAGE_SCHEMA)
    migrate(client, ul.OUTPUTS_TABLE, ul.OUTPUTS_SCHEMA)
    print("Done.")


if __name__ == "__main__":
    main()
