"""
FastAPI service that runs the Markowitz model for a supplied list of tickers.
- Input: comma-separated tickers string.
- Output: list of {ticker, weight} representing the optimized portfolio.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import warnings
from fastapi import FastAPI, HTTPException
from google.cloud import bigquery
from google.oauth2 import service_account
from pydantic import BaseModel, Field
import uvicorn

# Surface the local path in logs
print(f"[markowitz] Running from: {Path(__file__).resolve()}")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DAILY_PRICES_TABLE = "pro-visitor-429015-f5.StockData.daily_prices"

# Inline service account info supplied by user (consider moving to env/secret store in production).
# SERVICE_ACCOUNT_INFO = {}



def get_bigquery_client() -> bigquery.Client:
    """Build a BigQuery client from inline service account JSON."""
    credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
    return bigquery.Client(credentials=credentials, project=credentials.project_id)


# ---------------------------------------------------------------------------
# Markowitz core
# ---------------------------------------------------------------------------

def markowitz_solver(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.Series:
    """Solve for optimal weights maximizing Sharpe ratio."""
    from scipy.optimize import minimize

    num_assets = len(expected_returns)
    asset_names = expected_returns.index

    def portfolio_return(weights):
        return np.dot(weights, expected_returns)

    def portfolio_volatility(weights):
        var = weights.T @ cov_matrix @ weights
        return np.sqrt(max(var, 1e-8))

    def negative_sharpe(weights):
        p_ret = portfolio_return(weights)
        p_vol = portfolio_volatility(weights)
        if p_vol < 1e-5:
            return 0
        return -(p_ret - risk_free_rate) / p_vol

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1 / num_assets] * num_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-6,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    optimal_weights = pd.Series(result.x, index=asset_names)
    return optimal_weights[optimal_weights > 0.001].round(4).sort_values(ascending=False)


def fetch_prices(tickers: List[str]) -> pd.DataFrame:
    """Load price history for the requested tickers from BigQuery."""
    if not tickers:
        raise ValueError("Ticker list is empty.")

    client = get_bigquery_client()
    query = f"""
        SELECT date, ticker, close
        FROM `{DAILY_PRICES_TABLE}`
        WHERE ticker IN UNNEST(@tickers)
        ORDER BY date ASC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", tickers)]
    )
    df = client.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        raise ValueError("No price data returned for the requested tickers.")
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_markowitz(tickers: List[str]) -> pd.Series:
    """
    Execute the full data preparation + optimization pipeline for given tickers.
    """
    raw = fetch_prices(tickers)

    prices = (
        raw.pivot_table(index="date", columns="ticker", values="close", aggfunc="mean")
        .ffill()
        .dropna()
    )

    orig_rows, orig_cols = prices.shape
    if orig_cols == 0 or orig_rows == 0:
        raise ValueError("Insufficient price data after initial preparation.")

    returns = prices.pct_change()
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(axis=0, how="any")

    clean_rows, clean_cols = returns.shape
    if clean_cols == 0 or clean_rows == 0:
        raise ValueError("No usable returns after cleaning.")

    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    if expected_returns.isnull().any() or cov_matrix.isnull().values.any():
        expected_returns = expected_returns.fillna(0)
        cov_matrix = cov_matrix.fillna(0)

    optimal_portfolio = markowitz_solver(
        expected_returns=expected_returns, cov_matrix=cov_matrix, risk_free_rate=0.02
    )
    return optimal_portfolio


# ---------------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Markowitz Optimizer", version="1.0.0")


class TickersRequest(BaseModel):
    tickers: str = Field(..., example="AAPL,MSFT,NVDA")


@app.post("/optimize")
def optimize_portfolio(body: TickersRequest):
    tickers = [t.strip().upper() for t in body.tickers.split(",") if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="Provide at least one ticker, comma-separated.")

    try:
        weights = run_markowitz(tickers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to optimize: {exc}") from exc

    return [{"ticker": t, "weight": float(w)} for t, w in weights.items()]


if __name__ == "__main__":
    uvicorn.run("markovich:app", host="0.0.0.0", port=8000, reload=False)
