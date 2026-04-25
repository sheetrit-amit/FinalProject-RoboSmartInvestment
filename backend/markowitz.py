"""
Markowitz Portfolio Optimizer

Fetches price history from BigQuery and returns optimal weights that
maximise the Sharpe ratio (mean-variance optimisation on the efficient frontier).
"""

import logging
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from google.cloud import bigquery

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

DAILY_PRICES_TABLE = "pro-visitor-429015-f5.StockData.daily_prices"
RISK_FREE_RATE = 0.045          # ~current US T-bill yield
MAX_TICKERS_FOR_OPTIMIZER = 150  # cap for performance
BQ_LOCATION = "EU"              # StockData dataset lives in EU region


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_prices(tickers: List[str], client: bigquery.Client) -> pd.DataFrame:
    """Load OHLCV close prices for *tickers* from BigQuery."""
    if not tickers:
        raise ValueError("Ticker list is empty.")

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("tickers", "STRING", tickers)
        ]
    )
    query = f"""
        SELECT date, ticker, close
        FROM `{DAILY_PRICES_TABLE}`
        WHERE ticker IN UNNEST(@tickers)
        ORDER BY date ASC
    """
    df = client.query(query, job_config=job_cfg, location=BQ_LOCATION).to_dataframe()
    if df.empty:
        raise ValueError("No price data returned for the requested tickers.")

    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def _negative_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> float:
    p_ret = weights @ expected_returns
    p_var = weights @ cov @ weights
    p_vol = np.sqrt(max(p_var, 1e-10))
    return 0.0 if p_vol < 1e-6 else -(p_ret - rf) / p_vol


def markowitz_solver(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.Series:
    """
    Solve for the portfolio weights on the efficient frontier that maximise
    the Sharpe ratio subject to: sum(w)=1, 0 ≤ w_i ≤ 1.

    Returns a Series of non-trivial weights (> 0.1 %) sorted descending.
    """
    n = len(expected_returns)
    mu = expected_returns.values
    sigma = cov_matrix.values

    result = minimize(
        _negative_sharpe,
        np.full(n, 1 / n),
        args=(mu, sigma, risk_free_rate),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        tol=1e-8,
        options={"maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")

    weights = pd.Series(result.x, index=expected_returns.index)
    return weights[weights > 0.001].round(4).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_markowitz(tickers: List[str], client: bigquery.Client) -> List[Dict]:
    """
    Full pipeline: load prices → clean → compute returns → optimise.

    Returns a list of {"ticker": str, "weight": float} dicts, sorted by
    weight descending.  Only tickers with weight > 0.1 % are included.
    """
    # Keep at most MAX_TICKERS_FOR_OPTIMIZER to bound compute time
    tickers = tickers[:MAX_TICKERS_FOR_OPTIMIZER]

    raw = fetch_prices(tickers, client)

    # Pivot to wide format
    prices = (
        raw.pivot_table(index="date", columns="ticker", values="close", aggfunc="mean")
        .ffill()
        .dropna()
    )

    if prices.shape[0] < 60:
        raise ValueError(
            f"Only {prices.shape[0]} trading days available after cleaning — "
            "need at least 60."
        )

    # Daily returns
    returns = prices.pct_change()
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns.dropna(axis=1, how="all", inplace=True)
    returns.dropna(axis=0, how="any", inplace=True)

    if returns.shape[1] < 2:
        raise ValueError("Fewer than 2 tickers have usable return data.")

    # Pre-filter to top-N by individual Sharpe (keeps the problem manageable)
    if returns.shape[1] > 80:
        sharpe = returns.mean() / returns.std().replace(0, np.nan)
        top = sharpe.nlargest(80).index
        returns = returns[top]
        logger.info("Pre-filtered to %d tickers by individual Sharpe", len(top))

    # Annualised statistics
    exp_ret = returns.mean() * 252
    cov = returns.cov() * 252

    exp_ret.fillna(0, inplace=True)
    cov.fillna(0, inplace=True)

    logger.info("Running Markowitz on %d assets …", len(exp_ret))
    optimal = markowitz_solver(exp_ret, cov)
    logger.info("Optimal basket: %d positions", len(optimal))

    return [{"ticker": t, "weight": float(w)} for t, w in optimal.items()]
