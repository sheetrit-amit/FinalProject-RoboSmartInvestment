"""
BigQuery helper layer for RoboSmartInvestment.
All project-specific table names live here.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from google.cloud import bigquery
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

PROJECT_ID = "pro-visitor-429015-f5"
BQ_LOCATION = "EU"   # StockData dataset lives in EU region

# Table references
_TICKER_GRADES   = f"{PROJECT_ID}.StockData.ticker_grades"
_RISK_RATINGS    = f"{PROJECT_ID}.StockData.companies_risk_ratings"
_DAILY_PRICES    = f"{PROJECT_ID}.StockData.daily_prices"


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_bigquery_client() -> bigquery.Client:
    """
    Build a BigQuery client.

    Resolution order:
    1. GOOGLE_APPLICATION_CREDENTIALS env-var (path to a service-account JSON)
    2. Application Default Credentials (``gcloud auth application-default login``)
    """
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(sa_path)
        logger.info("BigQuery: using service-account file %s", sa_path)
        return bigquery.Client(credentials=creds, project=PROJECT_ID, location=BQ_LOCATION)

    logger.info("BigQuery: using Application Default Credentials")
    return bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_tickers_by_risk(risk_level: str, client: bigquery.Client) -> List[str]:
    """
    Return all tickers that have a grade AND match the requested risk level.
    Ordered by ticker for reproducibility.
    """
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
    """
    Fetch ticker_grades (mark + explanation) for the given tickers.
    Returns an empty list when tickers is empty.
    """
    if not tickers:
        return []

    # Build SQL IN clause safely (tickers are uppercase stock symbols)
    in_clause = "(" + ", ".join(f"'{t}'" for t in tickers) + ")"
    query = f"""
        SELECT ticker, mark, explanation
        FROM `{_TICKER_GRADES}`
        WHERE ticker IN {in_clause}
    """
    rows = client.query(query).result()
    return [
        {
            "ticker": row.ticker,
            "mark": float(row.mark) if row.mark is not None else None,
            "explanation": str(row.explanation or ""),
        }
        for row in rows
    ]


def get_price_history(
    tickers: List[str],
    client: bigquery.Client,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch close-price history for a list of tickers (ordered by date asc).
    Optionally cap at *limit* most recent rows per ticker.
    """
    if not tickers:
        return []

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )
    if limit:
        query = f"""
            SELECT date, ticker, close
            FROM `{_DAILY_PRICES}`
            WHERE ticker IN UNNEST(@tickers)
            QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) <= {limit}
            ORDER BY date ASC
        """
    else:
        query = f"""
            SELECT date, ticker, close
            FROM `{_DAILY_PRICES}`
            WHERE ticker IN UNNEST(@tickers)
            ORDER BY date ASC
        """
    rows = client.query(query, job_config=job_cfg).result()
    return [{"date": str(row.date), "ticker": row.ticker, "close": row.close} for row in rows]
