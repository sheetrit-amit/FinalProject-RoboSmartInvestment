"""bigquery helper layer, all table names live here"""

import logging
import os
from typing import Any, Dict, List

from google.cloud import bigquery
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

PROJECT_ID = "pro-visitor-429015-f5"
BQ_LOCATION = "EU"

_TICKER_GRADES = f"{PROJECT_ID}.StockData.ticker_grades"


def get_bigquery_client() -> bigquery.Client:
    # resolution order: inline json env, sa file env, application default creds
    import json as _json
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            info = _json.loads(sa_json)
            creds = service_account.Credentials.from_service_account_info(info)
            logger.info("BigQuery: using inline GCP_SERVICE_ACCOUNT_JSON")
            return bigquery.Client(credentials=creds, project=PROJECT_ID, location=BQ_LOCATION)
        except Exception:
            pass

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(sa_path)
        logger.info("BigQuery: using service-account file %s", sa_path)
        return bigquery.Client(credentials=creds, project=PROJECT_ID, location=BQ_LOCATION)

    logger.info("BigQuery: using Application Default Credentials")
    return bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)


def get_tickers_by_risk(risk_level: str, client: bigquery.Client) -> List[str]:
    # graded tickers matching the requested risk level, ordered for reproducibility
    query = """
        SELECT tg.ticker
        FROM `pro-visitor-429015-f5.StockData.ticker_grades`  AS tg
        JOIN  `pro-visitor-429015-f5.StockData.companies_risk_ratings` AS rr
              ON tg.ticker = rr.ticker
        WHERE tg.mark IS NOT NULL
          AND rr.risk_level = @risk
        ORDER BY tg.ticker
    """
    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("risk", "STRING", risk_level)
        ]
    )
    rows = client.query(query, job_config=job_cfg).result()
    tickers = [row.ticker for row in rows]
    logger.info("Risk=%s → %d tickers", risk_level, len(tickers))
    return tickers


def get_fundamental_scores(
    tickers: List[str],
    client: bigquery.Client,
) -> List[Dict[str, Any]]:
    # mark + explanation for the given tickers, empty list when none
    if not tickers:
        return []

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )
    query = f"""
        SELECT ticker, mark, explanation
        FROM `{_TICKER_GRADES}`
        WHERE ticker IN UNNEST(@tickers)
    """
    rows = client.query(query, job_config=job_cfg).result()
    return [
        {
            "ticker": row.ticker,
            "mark": float(row.mark) if row.mark is not None else None,
            "explanation": str(row.explanation or ""),
        }
        for row in rows
    ]
