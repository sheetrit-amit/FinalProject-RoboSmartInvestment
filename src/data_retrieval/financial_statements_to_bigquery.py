"""
Financial Statements to BigQuery Data Loader

Downloads financial statements (Income Statement, Balance Sheet, Cash Flow)
from Yahoo Finance and loads them into Google BigQuery.

Usage:
    python financial_statements_to_bigquery.py
"""

import logging
from typing import List, Dict

import yfinance as yf
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ID = "pro-visitor-429015-f5"
DATASET_ID = "StockData"
INCOME_TABLE = "income_statements"
BALANCE_TABLE = "balance_sheets"
CASHFLOW_TABLE = "cash_flows"
TICKER_FILE = "data/tickers_top1000.txt"
BATCH_SIZE = 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tickers_from_file(filename: str) -> List[str]:
    """Load ticker symbols from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def normalize_column_name(col: str) -> str:
    """
    Normalize column names for BigQuery compatibility.
    BigQuery column names must contain only letters, numbers, and underscores.
    """
    normalized = str(col).strip()
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('-', '_')
    normalized = normalized.replace('/', '_')
    normalized = normalized.replace('(', '')
    normalized = normalized.replace(')', '')
    normalized = normalized.replace('&', 'and')
    normalized = normalized.replace(',', '')
    normalized = normalized.replace('.', '')
    normalized = normalized.replace("'", '')
    
    if normalized and normalized[0].isdigit():
        normalized = '_' + normalized
    
    normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
    
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    return normalized.strip('_').lower()


def process_financial_statement(df: pd.DataFrame, ticker: str, period_type: str) -> pd.DataFrame:
    """
    Process a financial statement DataFrame for BigQuery insertion.
    Yahoo Finance returns data with dates as columns and metrics as rows - we transpose it.
    """
    if df.empty:
        return pd.DataFrame()
    
    df_transposed = df.T.reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'report_date'})
    
    df_transposed.insert(0, 'ticker', ticker)
    df_transposed['period_type'] = period_type
    df_transposed['report_date'] = pd.to_datetime(df_transposed['report_date']).dt.date
    
    # Normalize column names
    df_transposed.columns = [
        normalize_column_name(col) if col not in ['ticker', 'report_date', 'period_type'] 
        else col 
        for col in df_transposed.columns
    ]
    
    # Convert numeric columns to float
    for col in df_transposed.columns:
        if col not in ['ticker', 'report_date', 'period_type']:
            df_transposed[col] = pd.to_numeric(df_transposed[col], errors='coerce')
    
    return df_transposed


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

def get_bigquery_client():
    """Get BigQuery client."""
    return bigquery.Client(project=PROJECT_ID)


def ensure_dataset_exists(client: bigquery.Client):
    """Create the dataset if it doesn't exist."""
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {DATASET_ID}")


def insert_to_bigquery(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> bool:
    """Insert a DataFrame into BigQuery table."""
    if df.empty:
        return False
    
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    
    try:
        ensure_dataset_exists(client)
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            autodetect=True,
        )
        
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert to {table_ref}: {str(e)}")
        return False


# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================

def download_financial_statements(
    tickers: List[str],
    include_quarterly: bool = True,
    include_annual: bool = True
) -> Dict[str, Dict[str, bool]]:
    """
    Download financial statements from Yahoo Finance and insert into BigQuery.
    
    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols
    include_quarterly : bool
        Whether to include quarterly reports
    include_annual : bool
        Whether to include annual reports
    
    Returns
    -------
    Dict[str, Dict[str, bool]]
        Results: {ticker: {statement_type: success}}
    """
    client = get_bigquery_client()
    results = {}
    
    logger.info(f"Processing {len(tickers)} tickers...")
    
    for ticker in tickers:
        results[ticker] = {'income': False, 'balance': False, 'cashflow': False}
        
        try:
            logger.info(f"Processing {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Income Statement
            dfs = []
            if include_annual and stock.income_stmt is not None and not stock.income_stmt.empty:
                dfs.append(process_financial_statement(stock.income_stmt, ticker, 'annual'))
            if include_quarterly and stock.quarterly_income_stmt is not None and not stock.quarterly_income_stmt.empty:
                dfs.append(process_financial_statement(stock.quarterly_income_stmt, ticker, 'quarterly'))
            if dfs:
                results[ticker]['income'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), INCOME_TABLE)
            
            # Balance Sheet
            dfs = []
            if include_annual and stock.balance_sheet is not None and not stock.balance_sheet.empty:
                dfs.append(process_financial_statement(stock.balance_sheet, ticker, 'annual'))
            if include_quarterly and stock.quarterly_balance_sheet is not None and not stock.quarterly_balance_sheet.empty:
                dfs.append(process_financial_statement(stock.quarterly_balance_sheet, ticker, 'quarterly'))
            if dfs:
                results[ticker]['balance'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), BALANCE_TABLE)
            
            # Cash Flow
            dfs = []
            if include_annual and stock.cashflow is not None and not stock.cashflow.empty:
                dfs.append(process_financial_statement(stock.cashflow, ticker, 'annual'))
            if include_quarterly and stock.quarterly_cashflow is not None and not stock.quarterly_cashflow.empty:
                dfs.append(process_financial_statement(stock.quarterly_cashflow, ticker, 'quarterly'))
            if dfs:
                results[ticker]['cashflow'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), CASHFLOW_TABLE)
            
            success_count = sum(results[ticker].values())
            logger.info(f"  ✓ {ticker}: {success_count}/3 statements loaded")
            
        except Exception as e:
            logger.error(f"  ✗ {ticker} failed: {str(e)}")
    
    return results


def bulk_load(batch_size: int = BATCH_SIZE):
    """Load financial statements for all tickers in batches."""
    
    logger.info("=" * 70)
    logger.info("LOADING FINANCIAL STATEMENTS TO BIGQUERY")
    logger.info("=" * 70)
    
    # Load tickers
    tickers = load_tickers_from_file(TICKER_FILE)
    logger.info(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")
    
    # Process in batches
    all_results = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        logger.info(f"\n--- Batch {batch_num + 1}/{total_batches} ({start_idx + 1}-{end_idx}) ---")
        
        batch_results = download_financial_statements(batch_tickers)
        all_results.update(batch_results)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    total = len(all_results)
    income_ok = sum(1 for t in all_results if all_results[t]['income'])
    balance_ok = sum(1 for t in all_results if all_results[t]['balance'])
    cashflow_ok = sum(1 for t in all_results if all_results[t]['cashflow'])
    
    logger.info(f"Income Statements: {income_ok}/{total}")
    logger.info(f"Balance Sheets: {balance_ok}/{total}")
    logger.info(f"Cash Flows: {cashflow_ok}/{total}")
    logger.info("=" * 70)
    
    return all_results


if __name__ == '__main__':
    bulk_load()
