"""
Upload Training Data to BigQuery

Uploads all stock data (daily prices + financial statements) for 200 training companies
to a NEW BigQuery dataset for decision tree training.

Target Dataset: DecisionTreeTraining
Tables:
    - daily_prices
    - income_statements
    - balance_sheets
    - cash_flows

Usage:
    python upload_training_data_to_bigquery.py
"""

import logging
from datetime import datetime, timedelta
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
DATASET_ID = "DecisionTreeTraining"  # NEW dataset for training data
TICKER_FILE = "data/tickers_training_200.txt"
BATCH_SIZE = 20


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tickers_from_file(filename: str) -> List[str]:
    """Load ticker symbols from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def normalize_column_name(col: str) -> str:
    """Normalize column names for BigQuery compatibility."""
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
    """Process a financial statement DataFrame for BigQuery insertion."""
    if df.empty:
        return pd.DataFrame()
    
    df_transposed = df.T.reset_index()
    df_transposed = df_transposed.rename(columns={'index': 'report_date'})
    
    df_transposed.insert(0, 'ticker', ticker)
    df_transposed['period_type'] = period_type
    df_transposed['report_date'] = pd.to_datetime(df_transposed['report_date']).dt.date
    
    df_transposed.columns = [
        normalize_column_name(col) if col not in ['ticker', 'report_date', 'period_type'] 
        else col 
        for col in df_transposed.columns
    ]
    
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
        logger.info(f"Dataset {DATASET_ID} already exists")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset, timeout=30)
        logger.info(f"✓ Created NEW dataset: {DATASET_ID}")


def create_daily_prices_table(client: bigquery.Client):
    """Create the daily_prices table with proper schema."""
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.daily_prices"
    
    try:
        client.get_table(table_ref)
        logger.info("Table daily_prices already exists")
    except NotFound:
        schema = [
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("open", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("high", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("low", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("volume", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("dividends", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("stock_splits", "FLOAT64", mode="NULLABLE"),
        ]
        
        table = bigquery.Table(table_ref, schema=schema)
        table.clustering_fields = ["ticker", "date"]
        client.create_table(table)
        logger.info("✓ Created table: daily_prices")


def insert_to_bigquery(client: bigquery.Client, df: pd.DataFrame, table_id: str) -> bool:
    """Insert a DataFrame into BigQuery table."""
    if df.empty:
        return False
    
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_id}"
    
    try:
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
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_daily_prices(client: bigquery.Client, ticker: str, start_date: str, end_date: str) -> bool:
    """Download and upload daily prices for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return False
        
        df = df.reset_index()
        
        clean_data = pd.DataFrame({
            'ticker': ticker,
            'date': pd.to_datetime(df['Date']).dt.date,
            'open': df['Open'].round(2),
            'high': df['High'].round(2),
            'low': df['Low'].round(2),
            'close': df['Close'].round(2),
            'volume': df['Volume'].astype('Int64'),
            'dividends': df['Dividends'].round(4),
            'stock_splits': df['Stock Splits'].round(4)
        })
        
        return insert_to_bigquery(client, clean_data, "daily_prices")
        
    except Exception as e:
        logger.error(f"  Daily prices error for {ticker}: {str(e)}")
        return False


def download_financial_statements(client: bigquery.Client, ticker: str) -> Dict[str, bool]:
    """Download and upload all financial statements for a ticker."""
    results = {'income': False, 'balance': False, 'cashflow': False}
    
    try:
        stock = yf.Ticker(ticker)
        
        # Income Statement
        dfs = []
        if stock.income_stmt is not None and not stock.income_stmt.empty:
            dfs.append(process_financial_statement(stock.income_stmt, ticker, 'annual'))
        if stock.quarterly_income_stmt is not None and not stock.quarterly_income_stmt.empty:
            dfs.append(process_financial_statement(stock.quarterly_income_stmt, ticker, 'quarterly'))
        if dfs:
            results['income'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), "income_statements")
        
        # Balance Sheet
        dfs = []
        if stock.balance_sheet is not None and not stock.balance_sheet.empty:
            dfs.append(process_financial_statement(stock.balance_sheet, ticker, 'annual'))
        if stock.quarterly_balance_sheet is not None and not stock.quarterly_balance_sheet.empty:
            dfs.append(process_financial_statement(stock.quarterly_balance_sheet, ticker, 'quarterly'))
        if dfs:
            results['balance'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), "balance_sheets")
        
        # Cash Flow
        dfs = []
        if stock.cashflow is not None and not stock.cashflow.empty:
            dfs.append(process_financial_statement(stock.cashflow, ticker, 'annual'))
        if stock.quarterly_cashflow is not None and not stock.quarterly_cashflow.empty:
            dfs.append(process_financial_statement(stock.quarterly_cashflow, ticker, 'quarterly'))
        if dfs:
            results['cashflow'] = insert_to_bigquery(client, pd.concat(dfs, ignore_index=True), "cash_flows")
            
    except Exception as e:
        logger.error(f"  Financial statements error for {ticker}: {str(e)}")
    
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def upload_training_data(batch_size: int = BATCH_SIZE):
    """Upload all training data to BigQuery."""
    
    logger.info("=" * 70)
    logger.info("UPLOADING TRAINING DATA TO BIGQUERY")
    logger.info("=" * 70)
    logger.info(f"Project:  {PROJECT_ID}")
    logger.info(f"Dataset:  {DATASET_ID}")
    logger.info("=" * 70)
    
    # Setup BigQuery
    client = get_bigquery_client()
    ensure_dataset_exists(client)
    create_daily_prices_table(client)
    
    # Load tickers
    tickers = load_tickers_from_file(TICKER_FILE)
    logger.info(f"Loaded {len(tickers)} tickers from {TICKER_FILE}")
    
    # Date range (last 5 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Process tickers
    all_results = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        logger.info(f"\n--- Batch {batch_num + 1}/{total_batches} ({start_idx + 1}-{end_idx}) ---")
        
        for ticker in batch_tickers:
            logger.info(f"Processing {ticker}...")
            
            results = {
                'daily_prices': download_daily_prices(client, ticker, start_date, end_date),
            }
            
            fin_results = download_financial_statements(client, ticker)
            results.update({
                'income_statements': fin_results['income'],
                'balance_sheets': fin_results['balance'],
                'cash_flows': fin_results['cashflow'],
            })
            
            all_results[ticker] = results
            success_count = sum(results.values())
            logger.info(f"  ✓ {ticker}: {success_count}/4 tables")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    total = len(all_results)
    daily_ok = sum(1 for t in all_results if all_results[t]['daily_prices'])
    income_ok = sum(1 for t in all_results if all_results[t]['income_statements'])
    balance_ok = sum(1 for t in all_results if all_results[t]['balance_sheets'])
    cashflow_ok = sum(1 for t in all_results if all_results[t]['cash_flows'])
    
    logger.info(f"Daily Prices:      {daily_ok}/{total}")
    logger.info(f"Income Statements: {income_ok}/{total}")
    logger.info(f"Balance Sheets:    {balance_ok}/{total}")
    logger.info(f"Cash Flows:        {cashflow_ok}/{total}")
    logger.info("=" * 70)
    logger.info(f"✓ Data uploaded to: {PROJECT_ID}.{DATASET_ID}")
    logger.info(f"View at: https://console.cloud.google.com/bigquery?project={PROJECT_ID}")
    logger.info("=" * 70)
    
    # Save failed tickers
    failed = [t for t in all_results if sum(all_results[t].values()) < 4]
    if failed:
        with open('data/training_failed_tickers.txt', 'w') as f:
            for t in sorted(failed):
                f.write(f"{t}\n")
        logger.info(f"Failed/partial tickers saved to: data/training_failed_tickers.txt")
    
    return all_results


if __name__ == '__main__':
    upload_training_data()

