"""
Get Ticker Sectors

Fetches sector information for tickers and uploads to BigQuery.
Also saves a local CSV backup.

Output:
    - BigQuery table: DecisionTreeTraining.ticker_sectors
    - Local backup: data/ticker_sector_training.csv

Usage:
    python get_ticker_sectors.py
    
    # Or with custom input:
    python get_ticker_sectors.py --input data/tickers_training_200.txt
"""

import logging
import argparse
from typing import List, Tuple

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
DATASET_ID = "DecisionTreeTraining"
TABLE_ID = "ticker_sectors"
DEFAULT_TICKER_FILE = "data/tickers_training_200.txt"
DEFAULT_OUTPUT_FILE = "data/ticker_sector_training.csv"


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_tickers_from_file(filename: str) -> List[str]:
    """Load ticker symbols from a text file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_sector_for_ticker(ticker: str) -> str:
    """Get sector for a single ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        return sector if sector else 'Unknown'
    except Exception as e:
        logger.warning(f"Could not get sector for {ticker}: {e}")
        return 'Unknown'


def get_sectors_for_tickers(tickers: List[str]) -> List[Tuple[str, str]]:
    """
    Get sectors for a list of tickers.
    
    Returns:
        List of (sector, ticker) tuples
    """
    results = []
    total = len(tickers)
    
    logger.info(f"Fetching sectors for {total} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        sector = get_sector_for_ticker(ticker)
        results.append((sector, ticker))
        
        if i % 20 == 0 or i == total:
            logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
    
    return results


def save_to_local_csv(results: List[Tuple[str, str]], output_file: str):
    """Save ticker-sector mapping to local CSV file."""
    with open(output_file, 'w') as f:
        f.write("sector,ticker\n")
        for sector, ticker in results:
            f.write(f"{sector},{ticker}\n")
    
    logger.info(f"✓ Saved local backup to {output_file}")


def upload_to_bigquery(results: List[Tuple[str, str]]):
    """Upload ticker-sector mapping to BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    # Create DataFrame
    df = pd.DataFrame(results, columns=['sector', 'ticker'])
    
    # Check if dataset exists
    try:
        client.get_dataset(f"{PROJECT_ID}.{DATASET_ID}")
    except NotFound:
        dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
        dataset.location = "US"
        client.create_dataset(dataset, timeout=30)
        logger.info(f"Created dataset {DATASET_ID}")
    
    # Define schema
    schema = [
        bigquery.SchemaField("sector", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    ]
    
    # Delete existing table if exists (to replace with new data)
    try:
        client.delete_table(table_ref)
        logger.info(f"Deleted existing table {TABLE_ID}")
    except NotFound:
        pass
    
    # Create table and upload
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
    )
    
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    
    logger.info(f"✓ Uploaded {len(df)} rows to {table_ref}")


def generate_ticker_sector_file(
    ticker_file: str = DEFAULT_TICKER_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    upload_to_gcp: bool = True
) -> List[Tuple[str, str]]:
    """
    Main function to generate ticker-sector mapping.
    
    Can be called from other scripts:
        from get_ticker_sectors import generate_ticker_sector_file
        generate_ticker_sector_file("data/my_tickers.txt")
    """
    logger.info("=" * 70)
    logger.info("GENERATING TICKER-SECTOR MAPPING")
    logger.info("=" * 70)
    
    # Load tickers
    tickers = load_tickers_from_file(ticker_file)
    logger.info(f"Loaded {len(tickers)} tickers from {ticker_file}")
    
    # Get sectors
    results = get_sectors_for_tickers(tickers)
    
    # Save local backup
    save_to_local_csv(results, output_file)
    
    # Upload to BigQuery
    if upload_to_gcp:
        logger.info("\nUploading to BigQuery...")
        upload_to_bigquery(results)
    
    # Summary
    sectors = {}
    for sector, ticker in results:
        sectors[sector] = sectors.get(sector, 0) + 1
    
    logger.info("\n" + "=" * 70)
    logger.info("SECTOR DISTRIBUTION")
    logger.info("=" * 70)
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        logger.info(f"  {sector}: {count}")
    logger.info("=" * 70)
    logger.info(f"✓ Data available at: {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")
    logger.info("=" * 70)
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ticker-sector mapping and upload to BigQuery')
    parser.add_argument('--input', '-i', default=DEFAULT_TICKER_FILE,
                        help=f'Input ticker file (default: {DEFAULT_TICKER_FILE})')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_FILE,
                        help=f'Output CSV file for local backup (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip BigQuery upload, only save locally')
    
    args = parser.parse_args()
    
    generate_ticker_sector_file(
        ticker_file=args.input,
        output_file=args.output,
        upload_to_gcp=not args.no_upload
    )
