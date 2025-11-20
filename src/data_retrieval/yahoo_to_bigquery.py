"""
Yahoo Finance to BigQuery Data Loader

This module downloads historical stock market data from Yahoo Finance
and inserts it directly into Google BigQuery with a clean, well-defined schema.

The data is stored with properly separated columns, clean date formats,
and appropriate decimal precision for financial data.
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

import yfinance as yf
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BigQueryStockLoader:
    """
    Handles downloading stock data from Yahoo Finance and loading it into BigQuery
    with a clean, properly structured schema.
    """
    
    def __init__(self, project_id: str, dataset_id: str = 'StockData', table_id: str = 'daily_prices'):
        """
        Initialize the BigQuery Stock Loader.
        
        Parameters
        ----------
        project_id : str
            Your Google Cloud project ID (e.g., 'pro-visitor-429015-f5')
        dataset_id : str
            BigQuery dataset name (default: 'StockData')
        table_id : str
            BigQuery table name (default: 'daily_prices')
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        logger.info(f"Initialized BigQuery loader for {self.table_ref}")
    
    def create_table_if_not_exists(self):
        """
        Create the BigQuery table with proper schema if it doesn't exist.
        
        Schema ensures clean column separation:
        - ticker: Stock symbol (STRING)
        - date: Trading date (DATE - clean format, no timezone)
        - open: Opening price (FLOAT64, rounded to 2 decimals)
        - high: Highest price (FLOAT64, rounded to 2 decimals)
        - low: Lowest price (FLOAT64, rounded to 2 decimals)
        - close: Closing price (FLOAT64, rounded to 2 decimals)
        - volume: Trading volume (INT64)
        - dividends: Dividend amount (FLOAT64)
        - stock_splits: Stock split ratio (FLOAT64)
        """
        # Define clean schema with proper data types
        schema = [
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED", description="Stock ticker symbol"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED", description="Trading date"),
            bigquery.SchemaField("open", "FLOAT64", mode="NULLABLE", description="Opening price"),
            bigquery.SchemaField("high", "FLOAT64", mode="NULLABLE", description="Highest price"),
            bigquery.SchemaField("low", "FLOAT64", mode="NULLABLE", description="Lowest price"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED", description="Closing price"),
            bigquery.SchemaField("volume", "INT64", mode="NULLABLE", description="Trading volume"),
            bigquery.SchemaField("dividends", "FLOAT64", mode="NULLABLE", description="Dividend amount"),
            bigquery.SchemaField("stock_splits", "FLOAT64", mode="NULLABLE", description="Stock split ratio"),
        ]
        
        # Check if dataset exists
        try:
            self.client.get_dataset(f"{self.project_id}.{self.dataset_id}")
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            # Create dataset
            dataset = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
            dataset.location = "US"  # Change to your preferred location
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {self.dataset_id}")
        
        # Check if table exists
        try:
            self.client.get_table(self.table_ref)
            logger.info(f"Table {self.table_id} already exists")
        except NotFound:
            # Create table with clean schema
            table = bigquery.Table(self.table_ref, schema=schema)
            
            # Add clustering for better query performance
            table.clustering_fields = ["ticker", "date"]
            
            table = self.client.create_table(table)
            logger.info(f"Created table {self.table_id} with clean column structure")
            logger.info("Columns: ticker, date, open, high, low, close, volume, dividends, stock_splits")
    
    def download_and_insert(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Download stock data from Yahoo Finance and insert into BigQuery.
        
        Data is automatically cleaned and formatted:
        - Dates converted to clean DATE format (no timestamps)
        - Prices rounded to 2 decimal places
        - Volume converted to integers
        - Proper column separation in BigQuery
        
        Parameters
        ----------
        tickers : List[str]
            List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : Optional[str]
            End date in format 'YYYY-MM-DD'. If None, uses today's date.
        
        Returns
        -------
        Dict[str, bool]
            Dictionary mapping ticker symbols to success status
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure table exists
        self.create_table_if_not_exists()
        
        results = {}
        
        logger.info("=" * 70)
        logger.info(f"Starting download and load for {len(tickers)} ticker(s)")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Target table: {self.table_ref}")
        logger.info("=" * 70)
        
        for ticker in tickers:
            try:
                logger.info(f"Processing {ticker}...")
                
                # Download data from Yahoo Finance
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                # Check if data was retrieved
                if df.empty:
                    logger.warning(f"No data retrieved for {ticker}")
                    results[ticker] = False
                    continue
                
                # Clean and format data for BigQuery
                df = df.reset_index()
                
                # Prepare clean data structure
                clean_data = pd.DataFrame({
                    'ticker': ticker,
                    'date': pd.to_datetime(df['Date']).dt.date,  # Clean date (no timestamp)
                    'open': df['Open'].round(2),  # 2 decimal places
                    'high': df['High'].round(2),
                    'low': df['Low'].round(2),
                    'close': df['Close'].round(2),
                    'volume': df['Volume'].astype('Int64'),  # Integer volume
                    'dividends': df['Dividends'].round(4),
                    'stock_splits': df['Stock Splits'].round(4)
                })
                
                # Insert into BigQuery
                job_config = bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",  # Append to existing data
                    schema_update_options=[
                        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
                    ],
                )
                
                job = self.client.load_table_from_dataframe(
                    clean_data,
                    self.table_ref,
                    job_config=job_config
                )
                
                # Wait for the job to complete
                job.result()
                
                logger.info(f"✓ Successfully loaded {len(clean_data)} rows for {ticker}")
                results[ticker] = True
                
            except Exception as e:
                logger.error(f"✗ Failed to process {ticker}: {str(e)}")
                results[ticker] = False
        
        # Summary
        successful = sum(results.values())
        logger.info("=" * 70)
        logger.info(f"Load complete: {successful}/{len(tickers)} successful")
        logger.info("=" * 70)
        
        return results
    
    def query_sample(self, limit: int = 10):
        """
        Run a sample query to verify data looks clean and properly separated.
        
        Parameters
        ----------
        limit : int
            Number of rows to return (default: 10)
        """
        query = f"""
        SELECT 
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume
        FROM `{self.table_ref}`
        ORDER BY date DESC, ticker
        LIMIT {limit}
        """
        
        logger.info("\nRunning sample query to verify clean column separation...")
        logger.info("-" * 70)
        
        try:
            df = self.client.query(query).to_dataframe()
            print(df.to_string())
            logger.info("-" * 70)
            logger.info("✓ Data looks clean with properly separated columns!")
        except Exception as e:
            logger.error(f"Query failed: {e}")


def main():
    """
    Example usage: Download data for a few tickers and load into BigQuery.
    """
    # CONFIGURATION - CHANGE THESE VALUES
    PROJECT_ID = "pro-visitor-429015-f5"  # Your Google Cloud project ID
    DATASET_ID = "StockData"  # Your dataset name
    TABLE_ID = "daily_prices"  # Your table name
    
    # Example tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Calculate date range (last 5 years)
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Initialize loader
    loader = BigQueryStockLoader(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID
    )
    
    # Download and insert data
    results = loader.download_and_insert(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date_str
    )
    
    # Display results
    logger.info("\nLoad Results:")
    for ticker, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {ticker}: {status}")
    
    # Query sample data to verify clean columns
    if any(results.values()):
        logger.info("\n" + "=" * 70)
        loader.query_sample(limit=10)


if __name__ == '__main__':
    main()

