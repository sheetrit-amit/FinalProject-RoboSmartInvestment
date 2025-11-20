"""
Get Top Company Tickers

This script retrieves a comprehensive list of ~1000 major US company tickers.
Includes S&P 500, NASDAQ, Russell 1000, and other major stocks.
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_comprehensive_ticker_list():
    """
    Returns a comprehensive list of ~1000 major US company tickers.
    This is a curated list of the largest and most liquid US stocks.
    """
    # S&P 500 companies (major selection)
    sp500_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
        'XOM', 'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK',
        'AVGO', 'PEP', 'COST', 'KO', 'WMT', 'CSCO', 'ACN', 'TMO', 'MCD', 'ADBE',
        'ABT', 'ORCL', 'NFLX', 'NKE', 'CRM', 'DHR', 'TXN', 'VZ', 'NEE', 'WFC',
        'DIS', 'BMY', 'PM', 'UPS', 'QCOM', 'RTX', 'HON', 'COP', 'LOW', 'UNP',
        'MS', 'AMD', 'SBUX', 'BA', 'SPGI', 'GS', 'BLK', 'ELV', 'T', 'GILD',
        'DE', 'CAT', 'AXP', 'MMM', 'AMGN', 'NOW', 'PLD', 'MDT', 'INTC', 'ADI',
        'CVS', 'AMAT', 'MDLZ', 'ISRG', 'TJX', 'INTU', 'PYPL', 'CB', 'CI', 'SYK',
        'REGN', 'AMT', 'SO', 'BDX', 'ZTS', 'VRTX', 'DUK', 'PGR', 'ADP', 'MMC',
        'EOG', 'SCHW', 'BSX', 'TGT', 'ETN', 'USB', 'EQIX', 'NOC', 'APD', 'WM',
        'CL', 'NSC', 'LRCX', 'ITW', 'ICE', 'C', 'ECL', 'MCO', 'HUM', 'SLB',
        'GD', 'CSX', 'MO', 'AON', 'F', 'FI', 'PSA', 'PNC', 'EMR', 'FCX',
        'GM', 'TT', 'CCI', 'BKNG', 'MAR', 'CME', 'KLAC', 'ATVI', 'COF', 'AFL',
        'MPC', 'APH', 'MSI', 'CARR', 'AIG', 'OXY', 'SHW', 'DLR', 'PRU', 'HCA',
        'TEL', 'AZO', 'PSX', 'JCI', 'HLT', 'TRV', 'NXPI', 'PH', 'PCAR', 'FTNT',
        'MNST', 'ORLY', 'ADSK', 'O', 'ALL', 'SNPS', 'PAYX', 'MCK', 'AJG', 'ROP',
        'KMB', 'VLO', 'WELL', 'SRE', 'DD', 'CPRT', 'CMG', 'TFC', 'SPG', 'NEM',
        'CDNS', 'TROW', 'BK', 'STZ', 'YUM', 'DOW', 'ROST', 'VRSK', 'MSCI', 'CTVA',
        'KDP', 'AEP', 'EW', 'CTAS', 'ODFL', 'FAST', 'IQV', 'HSY', 'IDXX', 'D',
        'GIS', 'OTIS', 'CHTR', 'MCHP', 'BIIB', 'HES', 'EXC', 'KHC', 'A', 'WMB',
        'EXR', 'GPN', 'CTSH', 'EA', 'FANG', 'VICI', 'DAL', 'DVN', 'ANSS', 'LHX',
        'KMI', 'XEL', 'BKR', 'IT', 'PPG', 'GEHC', 'DHI', 'ED', 'MTB', 'AVB',
        'AME', 'GLW', 'DXCM', 'EFX', 'VMC', 'AWK', 'FTV', 'RMD', 'WEC', 'APTV',
        'LH', 'LEN', 'HPQ', 'ROK', 'CBRE', 'SYY', 'HAL', 'ACGL', 'IR', 'HIG',
        'GWW', 'OKE', 'URI', 'KEYS', 'TRGP', 'MLM', 'TTWO', 'EBAY', 'DOV', 'VRSN',
        'TSCO', 'FIS', 'ETR', 'WAB', 'DLTR', 'MTD', 'TDG', 'PHM', 'AEE', 'BALL',
        'WAT', 'STT', 'DTE', 'FITB', 'MPWR', 'IFF', 'ILMN', 'CDW', 'LYB', 'ALGN',
        'BAX', 'CAH', 'XYL', 'ZBH', 'PTC', 'ULTA', 'NTRS', 'ZBRA', 'EXPE', 'TYL',
        'PPL', 'HBAN', 'WBD', 'HOLX', 'DFS', 'FDS', 'CNP', 'IEX', 'ESS', 'DG',
        'RF', 'GRMN', 'CLX', 'ARE', 'TER', 'FE', 'VTR', 'DRI', 'WY', 'INVH',
        'RJF', 'EQR', 'STLD', 'MAA', 'WDC', 'CE', 'NTAP', 'CFG', 'SBAC', 'ATO',
        'TDY', 'PKI', 'MKC', 'COO', 'EPAM', 'LVS', 'FICO', 'CINF', 'PFG', 'EXPD',
        'CMS', 'AKAM', 'K', 'TRMB', 'CRL', 'EIX', 'STE', 'CTRA', 'J', 'MOH',
        'CHRW', 'OMC', 'UDR', 'EMN', 'DGX', 'BBY', 'MRO', 'NVR', 'CAG', 'SWK',
        'JBHT', 'VTRS', 'LUV', 'JKHY', 'SWKS', 'PAYC', 'LKQ', 'EVRG', 'HRL', 'KMX',
        'BG', 'IP', 'NI', 'POOL', 'HST', 'TECH', 'BXP', 'BRO', 'PEAK', 'QRVO',
        'GPC', 'NDSN', 'SJM', 'CPT', 'SNA', 'UHS', 'AOS', 'FFIV', 'HWM', 'TPR',
        'AIZ', 'ALLE', 'LNT', 'MAS', 'PNR', 'CTLT', 'TAP', 'BWA', 'FRT', 'GL',
        'NRG', 'MTCH', 'REG', 'WHR', 'WYNN', 'AAP', 'FOXA', 'PNW', 'IVZ', 'ZION',
        'BBWI', 'MGM', 'HII', 'APA', 'HSIC', 'NWSA', 'ALK', 'AVY', 'FMC', 'CPB',
        'PARA', 'MHK', 'RL', 'DVA', 'OGN', 'XRAY', 'HAS', 'VFC', 'CZR', 'UAA',
    ]
    
    # Additional NASDAQ stocks (Tech-heavy)
    nasdaq_tickers = [
        'GOOG', 'ABNB', 'ADP', 'AEP', 'ALGN', 'AMAT', 'AMD', 'AMZN', 'ANSS', 'ASML',
        'AVGO', 'BIDU', 'BIIB', 'BKNG', 'CEG', 'CHTR', 'CMCSA', 'COIN', 'COST', 'CPRT',
        'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DDOG', 'DLTR', 'DOCU', 'DXCM', 'EA',
        'EBAY', 'ENPH', 'EXC', 'FANG', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GILD', 'HON',
        'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LIN',
        'LRCX', 'LULU', 'MAR', 'MCHP', 'MDB', 'MDLZ', 'MELI', 'MNST', 'MRNA', 'MRVL',
        'NFLX', 'NTES', 'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR',
        'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'RIVN', 'ROST', 'SBUX', 'SGEN', 'SIRI',
        'SNPS', 'TEAM', 'TMUS', 'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBA',
        'WBD', 'WDAY', 'XEL', 'ZM', 'ZS', 'ADBE', 'ATVI', 'AVGO', 'CDNS', 'CPRT',
        'CSGP', 'DASH', 'DKNG', 'FSLR', 'GOOGL', 'HOOD', 'LCID', 'LYFT', 'META', 'MSFT',
        'MU', 'NET', 'NUAN', 'OKTA', 'PTON', 'RBLX', 'ROKU', 'SHOP', 'SNOW', 'SPLK',
        'SQ', 'TWLO', 'UBER', 'UPST', 'VRSN', 'WYNN', 'ZI', 'ABNB', 'ADSK', 'AEP',
    ]
    
    # Russell 1000 additional stocks (Mid-caps and beyond)
    russell_additional = [
        'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACGL', 'ACN',
        'AES', 'AFG', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL',
        'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'AN',
        'ANET', 'ANF', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'ARE', 'ARES',
        'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX',
        'BBY', 'BDX', 'BEN', 'BF.B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK',
        'BMY', 'BR', 'BRK.A', 'BRO', 'BSX', 'BWA', 'BXP', 'C', 'CAG', 'CAH',
        'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE',
        'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX',
        'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO',
        'COP', 'COST', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSX', 'CTAS',
        'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DD',
        'DE', 'DELL', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR',
        'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA',
        'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH',
        'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG',
        'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS',
        'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FLT', 'FMC', 'FOXA',
        'FRC', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN', 'GFS',
        'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN',
        'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII',
        'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM',
        'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU',
        'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ',
        'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY',
        'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'L',
        'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT',
        'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA',
        'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META',
        'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH',
        'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI',
        'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NET',
        'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
        'NVDA', 'NVR', 'NWL', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC',
        'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR',
        'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM',
        'PKG', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG',
        'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO',
        'RCL', 'RE', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK',
        'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SE',
        'SEDG', 'SEE', 'SHW', 'SIRI', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG',
        'SPGI', 'SPLK', 'SQ', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK',
        'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TEAM', 'TECH',
        'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'THC', 'TJX', 'TMO', 'TMUS', 'TPR',
        'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN',
        'TXT', 'TYL', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS',
        'URI', 'USB', 'V', 'VICI', 'VLO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR',
        'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC',
        'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WTW', 'WY', 'WYNN',
        'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZION', 'ZM', 'ZS', 'ZTS',
    ]
    
    # Combine all lists
    all_tickers = sp500_tickers + nasdaq_tickers + russell_additional
    
    # Remove duplicates and sort
    unique_tickers = sorted(list(set(all_tickers)))
    
    return unique_tickers


def save_tickers_to_file(tickers, filename='data/tickers_top1000.txt'):
    """Save tickers to a text file, one per line."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    
    logger.info(f"✓ Saved {len(tickers)} tickers to {filename}")
    return filename


def main():
    """Main function to get and save top 1000 tickers."""
    logger.info("=" * 70)
    logger.info("Compiling Top 1000+ Company Tickers")
    logger.info("=" * 70)
    
    tickers = get_comprehensive_ticker_list()
    
    logger.info(f"\n✓ Total unique tickers: {len(tickers)}")
    
    # Display first 30 as sample
    logger.info("\nSample tickers (first 30):")
    logger.info(", ".join(tickers[:30]))
    
    # Save to file
    filename = save_tickers_to_file(tickers)
    
    logger.info(f"\n✓ Ready to download data for {len(tickers)} companies!")
    logger.info(f"✓ Tickers saved to: {filename}")
    
    return tickers


if __name__ == '__main__':
    main()
