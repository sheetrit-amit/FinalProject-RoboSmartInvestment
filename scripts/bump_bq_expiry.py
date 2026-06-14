"""Reset the 60-day expiration timer on every StockData table.

The project is a BigQuery Sandbox: every table auto-deletes 60 days after creation.
This pushes each table's expiration to now + 60 days (in place — no data is moved,
dropped, or reloaded), buying another full cycle. Re-run before the new expiry.

The permanent fix is to attach a billing account (removes forced expiration); this
script is the stopgap until then.

Run:  python scripts/bump_bq_expiry.py
Needs Application Default Credentials.
"""

import datetime

from google.cloud import bigquery

PROJECT = "pro-visitor-429015-f5"
DATASET = "StockData"
LOCATION = "EU"
EXTEND_DAYS = 60  # sandbox max


def main():
    client = bigquery.Client(project=PROJECT, location=LOCATION)
    new_expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=EXTEND_DAYS)

    for item in client.list_tables(f"{PROJECT}.{DATASET}"):
        table = client.get_table(item.reference)
        old = table.expires
        table.expires = new_expiry
        client.update_table(table, ["expires"])
        print(f"{item.table_id:28s} {str(old):35s} -> {client.get_table(item.reference).expires}")

    print(f"\nAll tables now expire ~{new_expiry.date()}.")


if __name__ == "__main__":
    main()
