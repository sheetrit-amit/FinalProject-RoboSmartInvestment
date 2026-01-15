"""
Markowitz Portfolio Optimizer

This module implements the Markowitz Mean-Variance Portfolio Optimization model.
It finds the optimal portfolio weights that maximize the Sharpe Ratio for a given
set of stocks filtered by risk level.

Key Features:
- Fetches stock prices from BigQuery
- Calculates expected returns and covariance matrix
- Optimizes portfolio using Sharpe Ratio maximization
- Supports filtering by risk level from Decision Tree classifier

Usage:
    python portfolio_optimizer.py
    
    # Or import and use the function:
    from markowitz.portfolio_optimizer import optimize_portfolio_by_risk
    portfolio = optimize_portfolio_by_risk('Medium')
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
from google.cloud import bigquery


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific RuntimeWarnings from numpy and pandas
warnings.filterwarnings("ignore", category=RuntimeWarning, module='numpy')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='pandas')


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ID = "pro-visitor-429015-f5"

TABLES = {
    'prices': f"{PROJECT_ID}.StockData.daily_prices",
    'balance': f"{PROJECT_ID}.StockData.balance_sheets",
    'income': f"{PROJECT_ID}.StockData.income_statements",
    'risk_ratings': f"{PROJECT_ID}.StockData.companies_risk_ratings"
}

DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate


# ==============================================================================
# BigQuery Connection
# ==============================================================================

def get_bigquery_client():
    """Get BigQuery client using Application Default Credentials."""
    client = bigquery.Client(project=PROJECT_ID)
    logger.info(f"✅ Connected to project: {PROJECT_ID}")
    return client


# ==============================================================================
# Markowitz Solver
# ==============================================================================

def markowitz_solver(expected_returns, cov_matrix, risk_free_rate=DEFAULT_RISK_FREE_RATE):
    """
    Solve for optimal portfolio weights using Markowitz Mean-Variance Optimization.
    
    Maximizes the Sharpe Ratio: (Portfolio Return - Risk Free Rate) / Portfolio Volatility
    
    Args:
        expected_returns: pd.Series of annualized expected returns for each asset
        cov_matrix: pd.DataFrame of annualized covariance matrix
        risk_free_rate: Annual risk-free rate (default: 2%)
    
    Returns:
        pd.Series: Optimal weights for each asset (only assets with weight > 0.1%)
    """
    num_assets = len(expected_returns)
    assets_names = expected_returns.index

    def portfolio_return(weights):
        """Calculate portfolio expected return."""
        return np.dot(weights, expected_returns)

    def portfolio_volatility(weights):
        """Calculate portfolio volatility (standard deviation)."""
        var = weights.T @ cov_matrix @ weights
        # Prevent negative value in sqrt due to floating point error
        return np.sqrt(max(var, 1e-8))

    def negative_sharpe(weights):
        """Negative Sharpe Ratio (for minimization)."""
        p_ret = portfolio_return(weights)
        p_vol = portfolio_volatility(weights)
        # Prevent division by zero
        if p_vol < 1e-5:
            return 0
        return -(p_ret - risk_free_rate) / p_vol

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: each weight between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/num_assets] * num_assets)

    # Run optimization
    result = minimize(
        negative_sharpe,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-6
    )

    if not result.success:
        logger.warning(f"Optimization warning: {result.message}")
        return None

    optimal_weights = pd.Series(result.x, index=assets_names)
    
    # Return only assets with meaningful weights (> 0.1%)
    return optimal_weights[optimal_weights > 0.001].round(4).sort_values(ascending=False)


# ==============================================================================
# Portfolio Optimization Functions
# ==============================================================================

def get_all_prices(client):
    """
    Fetch all daily prices from BigQuery.
    
    Returns:
        pd.DataFrame: Pivot table with dates as index and tickers as columns
    """
    query = f"""
        SELECT date, ticker, close
        FROM `{TABLES['prices']}`
        ORDER BY date ASC
    """
    
    logger.info("Fetching price data from BigQuery...")
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    
    # Pivot to get prices matrix
    prices = df.pivot_table(index='date', columns='ticker', values='close', aggfunc='mean')
    prices = prices.ffill().dropna()
    
    logger.info(f"Price matrix shape: {prices.shape}")
    return prices


def filter_top_stocks_by_sharpe(prices, top_n=250):
    """
    Filter stocks by Sharpe Ratio to select the best performers.
    
    Args:
        prices: pd.DataFrame of prices (dates x tickers)
        top_n: Number of top stocks to select
    
    Returns:
        pd.DataFrame: Filtered prices for top stocks
    """
    returns = prices.pct_change().dropna()
    
    avg_returns = returns.mean()
    std_dev = returns.std()
    sharpe_ratios = avg_returns / std_dev
    
    top_tickers = sharpe_ratios.sort_values(ascending=False).head(top_n).index
    
    logger.info(f"Selected top {len(top_tickers)} stocks by Sharpe Ratio")
    return prices[top_tickers]


def calculate_optimization_inputs(prices):
    """
    Calculate expected returns and covariance matrix from prices.
    
    Args:
        prices: pd.DataFrame of prices (dates x tickers)
    
    Returns:
        tuple: (expected_returns, cov_matrix) - both annualized
    """
    returns = prices.pct_change().dropna()
    
    # Clean data
    returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    returns = returns.dropna(axis=1, how='all')
    returns = returns.dropna(axis=0, how='any')
    
    # Annualize (252 trading days)
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Safety check for NaNs
    if expected_returns.isnull().any() or cov_matrix.isnull().values.any():
        logger.warning("NaNs found in matrices. Replacing with 0.")
        expected_returns = expected_returns.fillna(0)
        cov_matrix = cov_matrix.fillna(0)
    
    logger.info(f"Matrices ready. Shape: {cov_matrix.shape}")
    return expected_returns, cov_matrix


def optimize_portfolio_by_risk(target_risk, client=None):
    """
    Optimize portfolio for stocks matching a specific risk level.
    
    This is the main function to be used by n8n or other systems.
    
    Args:
        target_risk: Risk level from Decision Tree classifier
                    Options: 'High', 'Med-High', 'Medium', 'Med-Low', 'Low', 'Unknown'
        client: BigQuery client (optional, will create if not provided)
    
    Returns:
        pd.Series: Optimal portfolio weights for each selected stock
    """
    if client is None:
        client = get_bigquery_client()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing portfolio for Risk Level: '{target_risk}'")
    logger.info('='*60)
    
    # Query stocks matching the risk level
    query = f"""
        SELECT t.date, t.ticker, t.close
        FROM `{TABLES['prices']}` t
        INNER JOIN `{TABLES['risk_ratings']}` r ON t.ticker = r.ticker
        WHERE r.risk_level = '{target_risk}'
        ORDER BY t.date ASC
    """
    
    logger.info(f"Fetching stocks with Risk Level: '{target_risk}' from BigQuery...")
    df = client.query(query).to_dataframe()
    
    unique_tickers = df['ticker'].nunique()
    logger.info(f"Found {unique_tickers} stocks matching this risk profile.")
    
    if unique_tickers == 0:
        logger.warning("No stocks found for this risk level!")
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Create price matrix
    prices = df.pivot_table(index='date', columns='ticker', values='close', aggfunc='mean')
    prices = prices.ffill().dropna()
    
    logger.info(f"Data ready for model. Matrix shape: {prices.shape}")
    
    # Calculate returns and optimization inputs
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Safety checks
    if expected_returns.isnull().any() or cov_matrix.isnull().values.any():
        logger.warning("NaNs found. Filling with 0.")
        expected_returns = expected_returns.fillna(0)
        cov_matrix = cov_matrix.fillna(0)
    
    # Run optimization
    optimal_portfolio = markowitz_solver(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=DEFAULT_RISK_FREE_RATE
    )
    
    return optimal_portfolio


def optimize_full_market(client=None, top_n=250):
    """
    Optimize portfolio across the full market (top stocks by Sharpe).
    
    Args:
        client: BigQuery client (optional)
        top_n: Number of top stocks to consider
    
    Returns:
        pd.Series: Optimal portfolio weights
    """
    if client is None:
        client = get_bigquery_client()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing portfolio for Full Market (Top {top_n} stocks)")
    logger.info('='*60)
    
    # Get all prices
    prices = get_all_prices(client)
    
    # Filter top stocks
    prices_filtered = filter_top_stocks_by_sharpe(prices, top_n=top_n)
    
    # Calculate optimization inputs
    expected_returns, cov_matrix = calculate_optimization_inputs(prices_filtered)
    
    # Run optimization
    optimal_portfolio = markowitz_solver(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=DEFAULT_RISK_FREE_RATE
    )
    
    return optimal_portfolio


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main entry point for portfolio optimization."""
    client = get_bigquery_client()
    
    # Example 1: Optimize by risk level
    risk_level = 'Medium'
    logger.info(f"\n🎯 Example 1: Optimizing for '{risk_level}' risk level")
    
    portfolio = optimize_portfolio_by_risk(risk_level, client)
    
    if portfolio is not None:
        logger.info("\n--- Optimal Portfolio (Weights) ---")
        for ticker, weight in portfolio.items():
            logger.info(f"   {ticker}: {weight:.2%}")
        logger.info(f"\nTotal assets in portfolio: {len(portfolio)}")
    else:
        logger.warning("Optimization failed or no stocks found.")
    
    # Example 2: Full market optimization
    logger.info(f"\n🌍 Example 2: Full Market Optimization")
    
    full_portfolio = optimize_full_market(client, top_n=100)
    
    if full_portfolio is not None:
        logger.info("\n--- Full Market Optimal Portfolio ---")
        for ticker, weight in full_portfolio.head(10).items():
            logger.info(f"   {ticker}: {weight:.2%}")
        logger.info(f"   ... and {len(full_portfolio) - 10} more assets")
        logger.info(f"\nTotal assets: {len(full_portfolio)}")


if __name__ == "__main__":
    main()

