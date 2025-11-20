"""
Yahoo Finance Data Downloader

This module provides functionality to download historical stock market data
from Yahoo Finance using the yfinance library and save it to TSV files.
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

import yfinance as yf
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    output_dir: str = 'data/raw'
) -> Dict[str, bool]:
    """
    Download historical stock data from Yahoo Finance and save to TSV files.
    
    Parameters
    ----------
    tickers : List[str]
        List of stock ticker symbols (e.g., ['AAPL', 'GOOGL'])
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : Optional[str]
        End date in format 'YYYY-MM-DD'. If None, uses today's date.
    output_dir : str
        Directory path where TSV files will be saved (default: 'data/raw')
    
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping ticker symbols to success status (True/False)
    
    Examples
    --------
    >>> results = download_stock_data(['AAPL', 'GOOGL'], '2020-01-01')
    >>> print(results)
    {'AAPL': True, 'GOOGL': True}
    """
    # Set end_date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    logger.info(f"Starting download for {len(tickers)} ticker(s)")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    for ticker in tickers:
        try:
            logger.info(f"Downloading data for {ticker}...")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Check if data was retrieved
            if df.empty:
                logger.warning(f"No data retrieved for {ticker}")
                results[ticker] = False
                continue
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Save to TSV file
            output_path = os.path.join(output_dir, f"{ticker}.tsv")
            df.to_csv(output_path, sep='\t', index=False)
            
            logger.info(f"Successfully saved {len(df)} rows to {output_path}")
            results[ticker] = True
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {str(e)}")
            results[ticker] = False
    
    # Summary
    successful = sum(results.values())
    logger.info(f"Download complete: {successful}/{len(tickers)} successful")
    
    return results


def main():
    """
    Main function demonstrating usage of the downloader.
    Downloads AAPL and GOOGL data for the last 5 years.
    """
    # Define parameters
    tickers = ['AAPL', 'GOOGL']
    
    # Calculate start date (5 years ago from today)
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info("=" * 60)
    logger.info("Yahoo Finance Data Downloader - Example Run")
    logger.info("=" * 60)
    
    # Download data
    results = download_stock_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date_str,
        output_dir='data/raw'
    )
    
    # Display results
    logger.info("\nDownload Results:")
    for ticker, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {ticker}: {status}")


if __name__ == '__main__':
    main()

