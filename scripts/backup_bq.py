"""Local backup of the StockData BigQuery dataset.

Exports every table to db_backup/StockData/<table>.parquet plus a <table>.schema.json
(BigQuery column types/modes) and a manifest.json. Parquet + schema gives a clean
round-trip restore. The output dir is git-ignored — it can be large.

Run:  python scripts/backup_bq.py
Needs Application Default Credentials (gcloud auth application-default login).
"""

import datetime
import json
import os

from google.cloud import bigquery

PROJECT = "pro-visitor-429015-f5"
DATASET = "StockData"
LOCATION = "EU"

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db_backup", DATASET
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = bigquery.Client(project=PROJECT, location=LOCATION)

    manifest = {
        "project": PROJECT,
        "dataset": DATASET,
        "location": LOCATION,
        "exported_at": datetime.datetime.utcnow().isoformat() + "Z",
        "tables": {},
    }

    for item in client.list_tables(f"{PROJECT}.{DATASET}"):
        name = item.table_id
        table = client.get_table(item.reference)

        with open(os.path.join(OUT_DIR, f"{name}.schema.json"), "w") as f:
            json.dump(
                [{"name": s.name, "field_type": s.field_type, "mode": s.mode} for s in table.schema],
                f,
                indent=2,
            )

        df = client.query(f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`").to_dataframe()
        path = os.path.join(OUT_DIR, f"{name}.parquet")
        df.to_parquet(path, index=False)

        manifest["tables"][name] = {"rows": len(df), "file": os.path.basename(path)}
        print(f"{name:28s} {len(df):>8} rows -> {os.path.basename(path)}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBackup written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
