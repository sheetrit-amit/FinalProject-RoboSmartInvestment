#!/usr/bin/env python3
"""
tag_stocks.py — Populate ticker_grades for every untagged ticker.

Pipeline (same data as the original n8n DataTaggingWF, but using free sources):
  1. BQ:          find tickers in companies_risk_ratings but NOT in ticker_grades
  2. SEC EDGAR:   ticker → CIK → most-recent 10-Q filing → HTML text
  3. yfinance:    latest quarterly income statement figures
  4. OpenRouter:  score 0-100 + explanation  (same LLM prompt as n8n workflow)
  5. BQ:          INSERT row into ticker_grades

Usage:
    cd scripts
    python tag_stocks.py                      # process all untagged tickers
    python tag_stocks.py NVDA TSLA AMZN       # process specific tickers (bypass BQ check)
"""

import json
import logging
import os
import re
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")

load_dotenv(dotenv_path=os.path.join(_BACKEND, ".env"), override=True)

OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_API_KEY", "")
GCP_PROJECT     = os.getenv("GCP_PROJECT_ID", "pro-visitor-429015-f5")
BQ_DATASET      = os.getenv("BQ_DATASET", "StockData")
BQ_LOCATION     = os.getenv("BQ_LOCATION", "EU")

TABLE_GRADES = f"{GCP_PROJECT}.{BQ_DATASET}.ticker_grades"
TABLE_RISK   = f"{GCP_PROJECT}.{BQ_DATASET}.companies_risk_ratings"

# SEC EDGAR rate limit: 10 req/s. We stay well under it.
SEC_USER_AGENT   = "RoboSmartInvest shvachko9768@gmail.com"
SEC_SLEEP        = 0.5     # seconds between SEC requests
CLEAN_TEXT_LIMIT = 12_000  # chars sent to LLM (prevent token overflow)

# OpenRouter free models — same list as backend/model_router.py
LLM_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "google/gemma-3-27b-it:free",
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tag_stocks")

# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

def _bq_client() -> bigquery.Client:
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(credentials=creds, project=GCP_PROJECT, location=BQ_LOCATION)
    return bigquery.Client(project=GCP_PROJECT, location=BQ_LOCATION)


def get_untagged_tickers(bq: bigquery.Client) -> list[str]:
    """Tickers in companies_risk_ratings that have no row in ticker_grades."""
    query = f"""
        SELECT rr.ticker
        FROM   `{TABLE_RISK}` AS rr
        LEFT JOIN `{TABLE_GRADES}` AS tg ON rr.ticker = tg.ticker
        WHERE  tg.ticker IS NULL
        ORDER  BY rr.ticker
    """
    rows = list(bq.query(query, location=BQ_LOCATION).result())
    return [r.ticker for r in rows]


def insert_grade(bq: bigquery.Client, ticker: str, mark: float, explanation: str) -> None:
    errors = bq.insert_rows_json(TABLE_GRADES, [{"ticker": ticker, "mark": mark, "explanation": explanation}])
    if errors:
        raise RuntimeError(f"BQ insert errors: {errors}")

# ---------------------------------------------------------------------------
# SEC EDGAR — ticker → CIK mapping (cached once)
# ---------------------------------------------------------------------------

_CIK_MAP: dict[str, str] = {}   # ticker (upper) → zero-padded 10-digit CIK

def _load_cik_map() -> None:
    global _CIK_MAP
    if _CIK_MAP:
        return
    log.info("Loading SEC EDGAR CIK map…")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": SEC_USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    _CIK_MAP = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in data.values()
    }
    log.info("  CIK map loaded: %d companies", len(_CIK_MAP))
    time.sleep(SEC_SLEEP)


def _cik_for(ticker: str) -> Optional[str]:
    _load_cik_map()
    return _CIK_MAP.get(ticker.upper())

# ---------------------------------------------------------------------------
# SEC EDGAR — fetch most-recent 10-Q filing HTML
# ---------------------------------------------------------------------------

def _sec_get(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
    r.raise_for_status()
    time.sleep(SEC_SLEEP)
    return r


def fetch_10q_text(ticker: str) -> Optional[str]:
    """
    1. Look up CIK for *ticker*
    2. Fetch submission history from data.sec.gov
    3. Find most-recent 10-Q filing
    4. Download its primary HTML document
    5. Return cleaned natural-language text

    Mirrors n8n nodes: HTTP Request1 (FMP SEC search) → Filter (10-Q)
                       → HTTP Request2 (finalLink) → Code in JavaScript
    """
    cik = _cik_for(ticker)
    if not cik:
        log.warning("  [%s] CIK not found in SEC EDGAR map", ticker)
        return None

    log.info("  CIK: %s", cik)

    # Fetch all submission metadata
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        sub_r = _sec_get(sub_url)
    except Exception as exc:
        log.error("  [%s] SEC submissions fetch failed: %s", ticker, exc)
        return None

    sub = sub_r.json()
    recent = sub.get("filings", {}).get("recent", {})

    forms       = recent.get("form", [])
    acc_nos     = recent.get("accessionNumber", [])
    doc_names   = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    # Find most-recent 10-Q
    filing_idx = None
    for i, form in enumerate(forms):
        if form == "10-Q":
            filing_idx = i
            break

    if filing_idx is None:
        log.warning("  [%s] No 10-Q found in submission history", ticker)
        return None

    acc_no   = acc_nos[filing_idx].replace("-", "")
    doc_name = doc_names[filing_idx]
    f_date   = filing_dates[filing_idx]
    final_link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{doc_name}"

    log.info("  10-Q filing: %s  →  %s", f_date, final_link)

    # Download the filing HTML
    try:
        html_r = _sec_get(final_link)
    except Exception as exc:
        log.error("  [%s] SEC filing HTML download failed: %s", ticker, exc)
        return None

    text = _extract_clean_text(html_r.text)
    log.info("  Filing HTML: %d chars raw  →  %d chars clean", len(html_r.text), len(text))
    return text


def _extract_clean_text(html: str) -> str:
    """
    Convert SEC filing HTML to clean natural-language text.
    Mirrors the JavaScript 'Code in JavaScript' node in the n8n workflow exactly.
    """
    if not html:
        return ""

    text = html

    # Remove script / style / head
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>",   " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<head[\s\S]*?</head>",      " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<meta[\s\S]*?>",            " ", text, flags=re.IGNORECASE)

    # Inline XBRL (SEC-specific noise)
    text = re.sub(r"<ix:header[\s\S]*?</ix:header>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<ix:hidden[\s\S]*?</ix:hidden>",  " ", text, flags=re.IGNORECASE)

    # Add line breaks before block-level elements
    text = re.sub(r"</?(?:p|div|tr|br|li|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Table column separation
    text = re.sub(r"</(?:td|th)>", " | ", text, flags=re.IGNORECASE)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Clean whitespace
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Decode HTML entities
    import html as _html
    text = _html.unescape(text)

    # Remove junk lines
    clean_lines = []
    for line in text.split("\n"):
        t = line.strip()
        if not t:
            continue
        if len(t) < 5 and not re.search(r"[a-zA-Z]", t):
            continue
        if re.fullmatch(r"[\d\s|.()\-]+", t):
            continue
        clean_lines.append(t)

    return "\n".join(clean_lines).strip()

# ---------------------------------------------------------------------------
# yfinance — income statement (mirrors FMP /income-statement, limit=1)
# ---------------------------------------------------------------------------

def fetch_income_statement(ticker: str) -> Optional[dict]:
    """
    Fetch the most-recent quarterly income statement via yfinance.
    Returns a flat dict mirroring the FMP income-statement schema.
    """
    try:
        import yfinance as yf
        yf_ticker = yf.Ticker(ticker)
        qf = yf_ticker.quarterly_financials

        if qf is None or qf.empty:
            log.warning("  [%s] yfinance: no quarterly financials", ticker)
            return None

        # Most-recent column
        col = qf.columns[0]

        def _val(row: str) -> Optional[float]:
            try:
                v = qf.loc[row, col]
                return float(v) if v is not None and str(v) != "nan" else None
            except Exception:
                return None

        info   = yf_ticker.info or {}
        period = str(col.date()) if hasattr(col, "date") else str(col)

        stmt = {
            "date":             period,
            "symbol":           ticker,
            "reportedCurrency": info.get("financialCurrency", "USD"),
            "fiscalYear":       period[:4],
            "period":           "Q",
            "revenue":          _val("Total Revenue"),
            "grossProfit":      _val("Gross Profit"),
            "operatingIncome":  _val("Operating Income"),
            "netIncome":        _val("Net Income"),
            "eps":              info.get("trailingEps"),
            "epsDiluted":       info.get("trailingEps"),
        }

        log.info(
            "  Income stmt: period=%s  revenue=%s  netIncome=%s",
            stmt["period"], stmt["revenue"], stmt["netIncome"],
        )
        return stmt

    except Exception as exc:
        log.warning("  [%s] yfinance income statement failed: %s", ticker, exc)
        return None

# ---------------------------------------------------------------------------
# LLM scoring (OpenRouter) — same prompt as n8n DataTaggingWF
# ---------------------------------------------------------------------------

SCORE_SYSTEM = (
    "You are an expert financial analyst.\n"
    "Your job is to analyze the following company financial data and the native-language "
    "10-Q report text, then return a JSON with:\n\n"
    '- "mark": integer 0–100 assessing the company\'s situation '
    "(how well it is doing, stability, risks)\n"
    '- "explanation": a concise explanation (max 2 short paragraphs) supporting the score\n\n'
    "Scoring guide:\n"
    "0–20 = severe distress\n"
    "21–40 = weak / deteriorating\n"
    "41–60 = average / mixed\n"
    "61–80 = good / stable\n"
    "81–100 = excellent\n\n"
    "Return ONLY valid JSON with exactly:\n"
    '{"mark": <0-100>, "explanation": "<max 2 short paragraphs>"}'
)


def _build_user_prompt(ticker: str, income: Optional[dict], filing_text: str) -> str:
    fin_block = json.dumps(income, indent=2) if income else (
        f'{{"symbol": "{ticker}", "note": "income statement unavailable"}}'
    )
    truncated = filing_text[:CLEAN_TEXT_LIMIT]
    if len(filing_text) > CLEAN_TEXT_LIMIT:
        truncated += "\n\n[... text truncated ...]"

    return (
        f"Financial data:\n{fin_block}\n\n"
        f"Native-language 10-Q text:\n{truncated}"
    )


def llm_score(ticker: str, income: Optional[dict], filing_text: str) -> dict:
    """Call OpenRouter; try models in order; return {"mark": int, "explanation": str}."""
    user_prompt = _build_user_prompt(ticker, income, filing_text)
    headers = {
        "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://robosmart.dev",
        "X-Title":       "RoboSmartInvest Tagger",
    }
    base_payload = {
        "messages": [
            {"role": "system", "content": SCORE_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":  600,
        "temperature": 0.2,
    }

    last_err = None
    for model in LLM_MODELS:
        try:
            log.info("  LLM → %s", model)
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={**base_payload, "model": model},
                timeout=90,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)
            mark   = int(parsed["mark"])
            expl   = str(parsed["explanation"])
            assert 0 <= mark <= 100, f"mark {mark} out of 0-100 range"

            log.info("  LLM success: mark=%d", mark)
            return {"mark": mark, "explanation": expl}

        except Exception as exc:
            log.warning("  LLM model %s failed: %s", model, exc)
            last_err = exc
            time.sleep(2)

    raise RuntimeError(f"All LLM models failed. Last: {last_err}")

# ---------------------------------------------------------------------------
# Per-ticker pipeline
# ---------------------------------------------------------------------------

def process_ticker(bq: bigquery.Client, ticker: str) -> bool:
    t0 = time.perf_counter()
    log.info("━" * 60)
    log.info("▶  %s  — starting", ticker)

    # Step 1: SEC EDGAR 10-Q text
    filing_text = fetch_10q_text(ticker)
    if not filing_text:
        log.warning("  [%s] Could not retrieve 10-Q filing text — skipping", ticker)
        return False

    if len(filing_text) < 200:
        log.warning("  [%s] Filing text too short (%d chars) — skipping", ticker, len(filing_text))
        return False

    # Step 2: income statement (yfinance)
    income = fetch_income_statement(ticker)

    # Step 3: LLM scoring
    try:
        result = llm_score(ticker, income, filing_text)
    except Exception as exc:
        log.error("  [%s] LLM scoring failed: %s", ticker, exc)
        return False

    mark  = result["mark"]
    expl  = result["explanation"]
    log.info("  Score: %d/100", mark)
    log.info("  Explanation snippet: %s", expl[:180].replace("\n", " "))

    # Step 4: write to BigQuery
    try:
        insert_grade(bq, ticker, float(mark), expl)
    except Exception as exc:
        log.error("  [%s] BQ insert failed: %s", ticker, exc)
        return False

    elapsed = time.perf_counter() - t0
    log.info("  ✓ %s  →  mark=%d  (%.1fs total)", ticker, mark, elapsed)
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not OPEN_ROUTER_KEY:
        log.error("OPEN_ROUTER_API_KEY not set — check backend/.env")
        sys.exit(1)

    log.info("=" * 60)
    log.info("RoboSmartInvest — Stock Tagger")
    log.info("Project: %s  |  Table: %s", GCP_PROJECT, TABLE_GRADES)

    # Connect to BigQuery
    try:
        bq = _bq_client()
        bq.query("SELECT 1", location=BQ_LOCATION).result()
        log.info("BigQuery connected ✓")
    except Exception as exc:
        log.error("BigQuery connection failed: %s", exc)
        sys.exit(1)

    # Tickers to process
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
        log.info("Mode: explicit  →  %s", tickers)
    else:
        tickers = get_untagged_tickers(bq)
        log.info("Mode: auto-detect untagged  →  %d tickers", len(tickers))

    if not tickers:
        log.info("Nothing to process — all tickers are already tagged.")
        return

    log.info("Tickers: %s", tickers)

    # Pre-load CIK map once (one HTTP call)
    _load_cik_map()

    t_global = time.perf_counter()
    success = failed = 0

    for ticker in tickers:
        ok = process_ticker(bq, ticker)
        if ok:
            success += 1
        else:
            failed += 1
        # Brief pause between tickers (respects SEC rate limit)
        time.sleep(1.0)

    total = time.perf_counter() - t_global
    log.info("=" * 60)
    log.info("DONE  ✓ %d tagged  ✗ %d failed/skipped  —  %.1fs total", success, failed, total)


if __name__ == "__main__":
    main()
