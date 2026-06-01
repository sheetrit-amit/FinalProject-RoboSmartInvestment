# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the backend
```bash
cd backend && python main.py
# Starts FastAPI on http://localhost:8000
```

### Validate Python syntax (CI check)
```bash
python -m compileall backend src
pip check
```

### Test the technical scanner standalone
```bash
cd scripts
python test_scanner.py                   # default: AAPL MSFT NVDA TSLA JPM
python test_scanner.py GOOGL META AMZN   # custom tickers
```

### Full pipeline test (requires backend running)
```bash
cd scripts && python test_scanner.py --full
```

### Tag unscored stocks in BigQuery
```bash
cd scripts
python tag_stocks.py                     # auto: all untagged tickers
python tag_stocks.py NFLX SPOT UBER      # specific tickers
```

### Quick REPL check for scanner
```python
import sys; sys.path.insert(0, 'backend')
from technical_scanner import scan_tickers
results = scan_tickers(['AAPL', 'MSFT'], min_score=0)
```

## Architecture

### Request pipeline (Build Mode)
1. `POST /chat` with `explicit_risk`, `explicit_top_k`, `explicit_budget`, `explicit_currency`
2. `bigquery_client.py` → `get_tickers_by_risk(risk)` queries `companies_risk_ratings`
3. `technical_scanner.py` → `scan_tickers()` downloads 6mo daily data via yfinance, scores 0–100 on EMA trend, MACD, RSI divergence, relative strength vs S&P 500, volume; drops tickers below `MIN_SCORE=50`
4. `markowitz.py` → `run_markowitz()` fetches close prices from `daily_prices` in BigQuery, runs SciPy SLSQP to maximise Sharpe ratio, returns weights sorted descending
5. `bigquery_client.py` → `get_fundamental_scores()` fetches pre-computed scores from `ticker_grades` (populated by `tag_stocks.py`)
6. `model_router.py` → `ModelRouter.chat()` calls OpenRouter for narrative synthesis, auto-rotates across 12 free-tier models on HTTP 429

### Two-mode `/chat` endpoint
- **Conversational mode** (no explicit params): LLM extracts budget/risk/top_k from natural language and guides the user to click "Build My Portfolio"
- **Build mode** (explicit params set by UI button): runs the full pipeline above

### BigQuery tables (`pro-visitor-429015-f5.StockData`, EU region)
| Table | Purpose |
|---|---|
| `companies_risk_ratings` | Ticker → risk label (Low/Med-Low/Medium/Med-High/High) |
| `daily_prices` | OHLCV history used by Markowitz |
| `ticker_grades` | Fundamental scores (0–100) + LLM explanation per ticker |

### GCP credential resolution order (in `bigquery_client.py`)
1. `GCP_SERVICE_ACCOUNT_JSON` env var (inline JSON string — used on Render)
2. `GOOGLE_APPLICATION_CREDENTIALS` env var (path to SA file)
3. Application Default Credentials (`gcloud auth application-default login`)

## CORS allowed origins
Defined in `backend/main.py:_ALLOWED_ORIGINS`. To test from a new local origin (e.g. a different port), add it to that list. The deployed GitHub Pages origin (`https://sheetrit-amit.github.io`) is always included. `allow_credentials` is `False` — the API uses no cookies or auth tokens.

## Backend `.env` (required for local self-hosting)
```
OPEN_ROUTER_API_KEY=sk-or-v1-...
GCP_PROJECT_ID=pro-visitor-429015-f5
BQ_DATASET=StockData
BQ_LOCATION=EU
```

## Frontend
`frontend/index.html` is a self-contained SPA (no build step). It talks directly to the backend URL hardcoded in the file. For local development against a local backend, update the `API_BASE` constant in `index.html`. The frontend is deployed via GitHub Pages (`frontend-pages.yml` workflow).

## Deployment
- **Backend**: Render (`https://finalproject-robosmartinvestment.onrender.com`) — secrets set as Render env vars
- **Frontend**: GitHub Pages (`https://sheetrit-amit.github.io/FinalProject-RoboSmartInvestment/`)

## Portfolio labelling logic
Implemented in `backend/main.py:_label()`:
- `weight > 15%` AND `score ≥ 70` → **Top Pick**
- `weight > 15%` AND `score < 50` → **Statistical Hold**
- `weight ≤ 15%` AND `score ≥ 70` → **Growth Opportunity**
- otherwise → **Balanced Position**
