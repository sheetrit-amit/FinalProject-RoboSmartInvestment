"""
Stock Risk Classifier using Decision Tree

This module trains a Decision Tree model on financial data to classify stocks
into risk levels (Low, Med-Low, Medium, Med-High, High).

The model uses financial ratios as features:
- Debt to Equity
- Current Ratio
- Profit Margin
- Return on Assets (ROA)
- Asset Turnover
- Price to Book

Usage:
    python risk_classifier.py

Output:
    - stock_risk_model.pkl (trained model)
    - all_companies_risk_ratings.csv (classifications)
"""

import logging
import pandas as pd
import numpy as np
import joblib
from google.cloud import bigquery
from sklearn.tree import DecisionTreeClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ID = "pro-visitor-429015-f5"

# Training Tables (200 companies for training)
TRAIN_TABLES = {
    'prices': f"{PROJECT_ID}.DecisionTreeTraining.daily_prices",
    'balance': f"{PROJECT_ID}.DecisionTreeTraining.balance_sheets",
    'income': f"{PROJECT_ID}.DecisionTreeTraining.income_statements"
}

# Prediction Tables (1000+ companies to classify)
PREDICT_TABLES = {
    'prices': f"{PROJECT_ID}.StockData.daily_prices",
    'balance': f"{PROJECT_ID}.StockData.balance_sheets",
    'income': f"{PROJECT_ID}.StockData.income_statements"
}

MODEL_FILE = 'stock_risk_model.pkl'
OUTPUT_FILE = 'all_companies_risk_ratings.csv'

FEATURES = [
    'debt_to_equity',
    'current_ratio',
    'profit_margin',
    'roa',
    'asset_turnover',
    'price_to_book'
]


# ==============================================================================
# BigQuery Connection
# ==============================================================================

def get_bigquery_client():
    """Get BigQuery client using Application Default Credentials."""
    client = bigquery.Client(project=PROJECT_ID)
    logger.info(f"✅ Connected to project: {PROJECT_ID}")
    return client


# ==============================================================================
# Part A: Model Training
# ==============================================================================

def train_the_model(client):
    """
    Train Decision Tree model on training data.
    
    Returns:
        dict: Volatility thresholds for fallback classification
    """
    logger.info("\n📚 --- Part A: Training Model on DecisionTreeTraining Data ---")

    # 1. Fetching training data
    logger.info("   Fetching training data...")

    q_prices = f"""
        SELECT ticker, date, close 
        FROM `{TRAIN_TABLES['prices']}` 
        WHERE date >= '2020-01-01'
    """
    df_prices = client.query(q_prices).to_dataframe()

    q_balance = f"""
        SELECT ticker, report_date, total_debt, stockholders_equity, 
               working_capital, current_liabilities, total_assets, ordinary_shares_number
        FROM `{TRAIN_TABLES['balance']}`
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY report_date DESC) = 1
    """
    df_balance = client.query(q_balance).to_dataframe()

    q_income = f"""
        SELECT ticker, report_date, net_income, total_revenue
        FROM `{TRAIN_TABLES['income']}`
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY report_date DESC) = 1
    """
    df_income = client.query(q_income).to_dataframe()

    # 2. Calculate Label (True risk based on volatility)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices = df_prices.sort_values(['ticker', 'date'])
    df_prices['daily_return'] = df_prices.groupby('ticker')['close'].pct_change()
    risk_metric = df_prices.groupby('ticker')['daily_return'].std() * np.sqrt(252)
    risk_df = risk_metric.reset_index(name='volatility')

    # Clean infinity values from volatility
    risk_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    risk_df.dropna(inplace=True)

    risk_labels = ['Low', 'Med-Low', 'Medium', 'Med-High', 'High']
    risk_df['risk_label'] = pd.qcut(risk_df['volatility'], 5, labels=risk_labels)

    # 3. Build features for training
    latest_price = df_prices.sort_values('date').groupby('ticker').tail(1)[['ticker', 'close']]

    data = pd.merge(df_balance, df_income, on='ticker', how='inner')
    data = pd.merge(data, latest_price, on='ticker', how='inner')

    # Calculate ratios (replacing 0 in denominator with NaN to avoid crash)
    data['debt_to_equity'] = data['total_debt'] / data['stockholders_equity'].replace(0, np.nan)
    data['current_assets'] = data['working_capital'] + data['current_liabilities']
    data['current_ratio'] = data['current_assets'] / data['current_liabilities'].replace(0, np.nan)
    data['profit_margin'] = data['net_income'] / data['total_revenue'].replace(0, np.nan)
    data['roa'] = data['net_income'] / data['total_assets']
    data['asset_turnover'] = data['total_revenue'] / data['total_assets']
    data['market_cap'] = data['close'] * data['ordinary_shares_number']
    data['price_to_book'] = data['market_cap'] / data['stockholders_equity'].replace(0, np.nan)

    # Handle missing and infinite values for the model
    train_set = pd.merge(data, risk_df[['ticker', 'risk_label']], on='ticker', how='inner')
    train_set.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_set.dropna(subset=FEATURES, inplace=True)

    logger.info(f"   ✅ Model will train on {len(train_set)} quality companies.")

    # 4. Training and Saving
    X = train_set[FEATURES]
    y = train_set['risk_label']

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_FILE)
    logger.info(f"   💾 Model saved to file: {MODEL_FILE}")

    # Save volatility thresholds for fallback classification
    volatility_thresholds = {
        'Low': risk_df[risk_df['risk_label'] == 'Low']['volatility'].max(),
        'Med-Low': risk_df[risk_df['risk_label'] == 'Med-Low']['volatility'].max(),
        'Medium': risk_df[risk_df['risk_label'] == 'Medium']['volatility'].max(),
        'Med-High': risk_df[risk_df['risk_label'] == 'Med-High']['volatility'].max()
    }
    
    return volatility_thresholds


# ==============================================================================
# Part B: Prediction on all companies - Hybrid Approach
# ==============================================================================

def classify_fallback(vol_val, thresholds):
    """
    Fallback function: Ranks by volatility if reports are missing.
    
    Args:
        vol_val: Annualized volatility value
        thresholds: Dictionary of volatility thresholds per risk level
    
    Returns:
        str: Risk level classification
    """
    if pd.isna(vol_val):
        return "Unknown"
    if vol_val <= thresholds['Low']:
        return "Low"
    if vol_val <= thresholds['Med-Low']:
        return "Med-Low"
    if vol_val <= thresholds['Medium']:
        return "Medium"
    if vol_val <= thresholds['Med-High']:
        return "Med-High"
    return "High"


def run_full_prediction(client, vol_thresholds):
    """
    Classify all companies using hybrid approach:
    1. Smart Model: Uses Decision Tree for companies with full financial data
    2. Fallback: Uses volatility-based classification for others
    
    Args:
        client: BigQuery client
        vol_thresholds: Volatility thresholds from training
    """
    logger.info("\n🚀 --- Part B: Classifying all companies from StockData ---")

    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        logger.error("❌ Model not found. Please run training first.")
        return

    # 1. Fetching data
    logger.info("   Fetching data (Volatility + Reports)...")

    # Calculate volatility using SAFE_DIVIDE
    q_vol = f"""
        WITH Returns AS (
            SELECT ticker, date,
            SAFE_DIVIDE(
                close - LAG(close) OVER(PARTITION BY ticker ORDER BY date),
                LAG(close) OVER(PARTITION BY ticker ORDER BY date)
            ) as daily_return
            FROM `{PREDICT_TABLES['prices']}`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)
        )
        SELECT ticker, STDDEV(daily_return) * SQRT(252) as annualized_volatility
        FROM Returns 
        GROUP BY ticker
    """
    df_vol = client.query(q_vol).to_dataframe()

    # Fetching latest reports
    q_p = f"""
        SELECT ticker, close 
        FROM `{PREDICT_TABLES['prices']}` 
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) = 1
    """
    q_b = f"""
        SELECT ticker, total_debt, stockholders_equity, working_capital, 
               current_liabilities, total_assets, ordinary_shares_number 
        FROM `{PREDICT_TABLES['balance']}` 
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY report_date DESC) = 1
    """
    q_i = f"""
        SELECT ticker, net_income, total_revenue 
        FROM `{PREDICT_TABLES['income']}` 
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY report_date DESC) = 1
    """

    df_prices = client.query(q_p).to_dataframe()
    df_bal = client.query(q_b).to_dataframe()
    df_inc = client.query(q_i).to_dataframe()

    # 2. Data processing for Smart Model
    data = pd.merge(df_bal, df_inc, on='ticker', how='inner')
    data = pd.merge(data, df_prices, on='ticker', how='inner')

    # Calculate ratios
    data['debt_to_equity'] = data['total_debt'] / data['stockholders_equity'].replace(0, np.nan)
    data['current_assets'] = data['working_capital'] + data['current_liabilities']
    data['current_ratio'] = data['current_assets'] / data['current_liabilities'].replace(0, np.nan)
    data['profit_margin'] = data['net_income'] / data['total_revenue'].replace(0, np.nan)
    data['roa'] = data['net_income'] / data['total_assets']
    data['asset_turnover'] = data['total_revenue'] / data['total_assets']
    data['market_cap'] = data['close'] * data['ordinary_shares_number']
    data['price_to_book'] = data['market_cap'] / data['stockholders_equity'].replace(0, np.nan)

    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], 0, inplace=True)
    for col in FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    df_smart = data[FEATURES + ['ticker']].copy()

    # 3. Hybrid Classification
    # A. Smart Classification (Companies with valid reports)
    logger.info(f"   🤖 Running Smart Model on {len(df_smart)} companies...")
    if not df_smart.empty:
        df_smart['risk_level'] = model.predict(df_smart[FEATURES])
        df_smart['method'] = 'Smart Model'
    else:
        logger.warning("   ⚠️ No companies found suitable for Smart Model, moving all to fallback.")

    # B. Fallback Classification (Everyone else)
    smart_tickers = set(df_smart['ticker']) if not df_smart.empty else set()
    df_fallback = df_vol[~df_vol['ticker'].isin(smart_tickers)].copy()

    logger.info(f"   📉 Running Fallback Model (Volatility) on {len(df_fallback)} remaining companies...")
    df_fallback['risk_level'] = df_fallback['annualized_volatility'].apply(
        lambda x: classify_fallback(x, vol_thresholds)
    )
    df_fallback['method'] = 'Volatility Fallback'

    # 4. Merge and Export
    dfs_to_concat = []
    if not df_smart.empty:
        dfs_to_concat.append(df_smart[['ticker', 'risk_level', 'method']])
    if not df_fallback.empty:
        dfs_to_concat.append(df_fallback[['ticker', 'risk_level', 'method']])

    final_df = pd.concat(dfs_to_concat, ignore_index=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    logger.info("\n" + "=" * 50)
    logger.info(f"✅ Process complete! Classified {len(final_df)} companies.")
    logger.info(f"📄 File saved: {OUTPUT_FILE}")
    logger.info("=" * 50)
    logger.info("Results Distribution:")
    for level, count in final_df['risk_level'].value_counts().items():
        logger.info(f"   {level}: {count}")

    return final_df


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Main entry point for training and prediction."""
    # Initialize BigQuery client
    client = get_bigquery_client()
    
    # Step 1: Training
    thresholds = train_the_model(client)
    logger.info(f"\n💡 Learned fallback thresholds: {thresholds}")

    # Step 2: Prediction on the entire market
    run_full_prediction(client, thresholds)


if __name__ == "__main__":
    main()

