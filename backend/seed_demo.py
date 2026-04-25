"""
seed_demo.py — Quick-start data seeder
=======================================
Downloads 3 years of real price data for 60 well-known stocks via yfinance,
computes volatility-based risk labels, generates placeholder fundamental scores,
and uploads everything to BigQuery (StockData dataset, EU region).

Run once from the backend/ directory:
    python seed_demo.py

Tables created / replaced:
    StockData.daily_prices
    StockData.companies_risk_ratings
    StockData.ticker_grades
"""

import warnings
warnings.filterwarnings("ignore")

import datetime
import logging
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("seed")

PROJECT  = "pro-visitor-429015-f5"
DATASET  = "StockData"
LOCATION = "EU"

# 60 large-cap US stocks covering multiple sectors
TICKERS = [
    # Tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","ORCL","CRM","ADBE",
    # Finance
    "JPM","BAC","GS","MS","V","MA","AXP","BRK-B","WFC","C",
    # Healthcare
    "LLY","UNH","JNJ","PFE","ABBV","MRK","TMO","ABT","DHR","BMY",
    # Energy
    "XOM","CVX","COP","EOG","SLB",
    # Consumer
    "WMT","COST","HD","MCD","KO","PEP","NKE","SBUX","TGT","LOW",
    # Industrials / Other
    "GE","CAT","HON","UPS","DE","MMM","BA","LMT","RTX","NEE",
]

END   = datetime.date.today()
START = END - datetime.timedelta(days=3*365)

# ── helpers ────────────────────────────────────────────────────────────────

def get_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT, location=LOCATION)


def ensure_dataset(client: bigquery.Client):
    ref = f"{PROJECT}.{DATASET}"
    try:
        client.get_dataset(ref)
        log.info("Dataset %s already exists.", ref)
    except Exception:
        ds = bigquery.Dataset(ref)
        ds.location = LOCATION
        client.create_dataset(ds)
        log.info("Created dataset %s in %s.", ref, LOCATION)


def upload(client: bigquery.Client, df: pd.DataFrame, table_id: str, schema=None):
    full = f"{PROJECT}.{DATASET}.{table_id}"
    job_cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=schema,
        autodetect=(schema is None),
    )
    job = client.load_table_from_dataframe(df, full, job_config=job_cfg)
    job.result()
    log.info("Uploaded %d rows → %s", len(df), full)


# ── Step 1: download prices ────────────────────────────────────────────────

def download_prices() -> pd.DataFrame:
    log.info("Downloading prices for %d tickers (%s → %s) …", len(TICKERS), START, END)
    raw = yf.download(
        TICKERS, start=str(START), end=str(END),
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        sys.exit("yfinance returned no data — check your internet connection.")

    close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    close = close.ffill().dropna(how="all", axis=1)

    rows = []
    for ticker in close.columns:
        for date, price in close[ticker].dropna().items():
            rows.append({
                "date":   str(date.date()),
                "ticker": str(ticker),
                "close":  float(price),
            })
    df = pd.DataFrame(rows)
    log.info("Price rows: %d  tickers: %d", len(df), df["ticker"].nunique())
    return df


# ── Step 2: compute volatility-based risk labels ──────────────────────────

RISK_LABELS = ["Low", "Med-Low", "Medium", "Med-High", "High"]

def compute_risk(prices_df: pd.DataFrame) -> pd.DataFrame:
    pivot = prices_df.pivot(index="date", columns="ticker", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    returns = pivot.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    vol = returns.std() * np.sqrt(252)
    vol = vol.dropna()

    vol_df = vol.reset_index()
    vol_df.columns = ["ticker", "volatility"]
    vol_df["risk_level"] = pd.qcut(
        vol_df["volatility"], q=5, labels=RISK_LABELS
    ).astype(str)

    log.info("Risk distribution:\n%s", vol_df["risk_level"].value_counts().to_string())
    return vol_df[["ticker", "risk_level"]]


# ── Step 3: generate placeholder fundamental scores ───────────────────────

FUNDAMENTALS = {
    "AAPL":  (85, "Strong iPhone ecosystem, services revenue growing rapidly, solid cash generation."),
    "MSFT":  (90, "Azure cloud dominant, AI integration across products, robust recurring revenue."),
    "NVDA":  (88, "AI chip demand unprecedented, data-centre revenue surging, wide moat."),
    "GOOGL": (82, "Search monopoly intact, YouTube and Cloud growing, strong free cash flow."),
    "META":  (79, "Ad revenue recovery strong, AI investments paying off, cost discipline improved."),
    "AMZN":  (84, "AWS re-accelerating, retail margins expanding, advertising high-margin growth."),
    "TSLA":  (62, "Vehicle delivery growth slowing, energy business strong, competition rising."),
    "ORCL":  (75, "Cloud transition underway, AI-driven database demand, stable enterprise base."),
    "CRM":   (72, "Salesforce AI copilot early traction, margin expansion on track."),
    "ADBE":  (78, "Creative cloud sticky, Firefly AI differentiator, strong FCF."),
    "JPM":   (80, "Best-in-class US bank, net interest income elevated, loan quality solid."),
    "BAC":   (71, "Rate sensitivity a headwind but improving, consumer deposits stable."),
    "GS":    (69, "Investment banking recovering, asset management growing, trading volatile."),
    "MS":    (74, "Wealth management moat, fee-based revenue resilient."),
    "V":     (87, "Network effects unmatched, cross-border volumes recovering fully."),
    "MA":    (86, "Similar to Visa, faster international expansion, strong margin profile."),
    "LLY":  (92, "GLP-1 weight loss drugs transformative, pipeline deep, pricing power."),
    "UNH":  (81, "Health insurance scale advantages, Optum synergies strong."),
    "JNJ":  (70, "Pharma and medtech mix solid, litigation mostly resolved."),
    "PFE":  (55, "Post-COVID revenue normalising, pipeline rebuilding underway."),
    "XOM":  (73, "Strong FCF at current oil prices, Permian production growing."),
    "CVX":  (71, "Similar to XOM, Hess acquisition adds deepwater exposure."),
    "WMT":  (78, "Grocery share gains, e-commerce profitable, advertising revenue growing."),
    "COST": (83, "Membership model highly resilient, international expansion on track."),
    "HD":   (76, "Home improvement structurally supported, Pro segment strong."),
    "MCD":  (80, "Franchise model resilient, digital ordering and loyalty scaling."),
    "KO":   (72, "Pricing power proven, emerging market growth, stable dividends."),
    "NVDA": (88, "See above."),
    "NEE":  (74, "Renewables leadership, rate-base growth visible, regulated utility stability."),
    "GE":   (77, "Aerospace turnaround complete, engines demand at record backlog."),
}

def build_grades(tickers) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for t in tickers:
        if t in FUNDAMENTALS:
            mark, expl = FUNDAMENTALS[t]
            # add small jitter
            mark = min(100, max(0, int(mark + rng.integers(-3, 4))))
        else:
            mark = int(rng.integers(45, 85))
            expl = f"{t}: No analyst commentary available. Score based on quantitative proxies."
        rows.append({"ticker": t, "mark": mark, "explanation": expl})
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=== RoboSmartInvest — BigQuery Seeder ===")
    client = get_client()
    ensure_dataset(client)

    # 1. Prices
    prices_df = download_prices()
    upload(client, prices_df, "daily_prices", schema=[
        bigquery.SchemaField("date",   "STRING"),
        bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("close",  "FLOAT64"),
    ])

    # 2. Risk ratings
    seeded_tickers = prices_df["ticker"].unique().tolist()
    risk_df = compute_risk(prices_df)
    upload(client, risk_df, "companies_risk_ratings", schema=[
        bigquery.SchemaField("ticker",     "STRING"),
        bigquery.SchemaField("risk_level", "STRING"),
    ])

    # 3. Fundamental grades
    grades_df = build_grades(seeded_tickers)
    upload(client, grades_df, "ticker_grades", schema=[
        bigquery.SchemaField("ticker",      "STRING"),
        bigquery.SchemaField("mark",        "FLOAT64"),
        bigquery.SchemaField("explanation", "STRING"),
    ])

    log.info("")
    log.info("✅  Seeding complete!")
    log.info("    daily_prices         → %d rows", len(prices_df))
    log.info("    companies_risk_ratings → %d tickers", len(risk_df))
    log.info("    ticker_grades         → %d tickers", len(grades_df))
    log.info("")
    log.info("Restart the backend and try a chat message.")


if __name__ == "__main__":
    main()
