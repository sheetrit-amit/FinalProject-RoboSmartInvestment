# RoboSmartInvest

**AI-Powered Stock Portfolio Builder**

> Tell the system your investment amount and risk tolerance.  
> Get back an optimised portfolio with AI-generated investment reasoning.

---

## What It Does

1. **Understands your intent** — An LLM extracts your balance, currency, and risk tolerance from natural language.
2. **Classifies risk** — Stocks in BigQuery are pre-labelled Low / Med-Low / Medium / Med-High / High using volatility-based quintiles.
3. **Technical pre-filter** — Each candidate ticker is scored 0–100 using momentum and trend signals (EMA trend, MACD, RSI divergence, relative strength vs S&P 500, volume). Only tickers scoring ≥ 50 pass to the optimiser.
4. **Optimises the portfolio** — Markowitz mean-variance optimisation finds weights that maximise the Sharpe Ratio on the Efficient Frontier.
5. **Adds fundamental context** — Pre-computed analyst scores (from SEC 10-Q filings) are attached to each position.
6. **Synthesises a recommendation** — A final LLM call writes a human-readable investment thesis for the basket.

---

## Architecture

```
Browser (frontend/index.html)
        │  POST /chat  { message, explicit_* params }
        ▼
FastAPI Backend (backend/main.py)
   ├─ ModelRouter           (OpenRouter free-tier, auto-rotates on 429)
   ├─ BigQuery client        (ticker_grades, companies_risk_ratings, daily_prices)
   ├─ TechnicalScanner       (yfinance + custom indicators → 0-100 score per ticker)
   └─ Markowitz optimizer    (SciPy SLSQP, max-Sharpe)

Pipeline (build mode):
  1. BQ → candidate tickers by risk label
  2. TechnicalScanner → drop tickers scoring < 50, re-rank remainder
  3. Markowitz → optimal weights on filtered set
  4. BQ → fundamental scores (ticker_grades)
  5. OpenRouter LLM → narrative synthesis
        │
        ▼
Response  { text, portfolio[], risk_level, balance, currency }
  portfolio item: { ticker, weight, score, technical_score, label, explanation }
        │
        ▼
Live donut chart + stock-allocation bars (Chart.js)
```

---

## Usage

### Hosted App (No Local Credentials Needed)

Use the live deployment:

- Frontend: `https://sheetrit-amit.github.io/FinalProject-RoboSmartInvestment/`
- Backend API: `https://finalproject-robosmartinvestment.onrender.com`

To verify backend health:

```bash
curl https://finalproject-robosmartinvestment.onrender.com/health
```

Example API call:

```bash
curl -X POST https://finalproject-robosmartinvestment.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I want to invest $10,000 with medium risk"}'
```

### Local Frontend Only

Open `frontend/index.html` in your browser (double-click on Windows/macOS/Linux).  
The current frontend config points to the hosted Render backend.

### Optional: Self-Host Backend

Only needed if you want to run your own backend instance.

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Required `backend/.env`:
```
OPEN_ROUTER_API_KEY=sk-or-v1-...
GCP_PROJECT_ID=pro-visitor-429015-f5
BQ_DATASET=StockData
BQ_LOCATION=EU
```

Google credentials: either set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json` or run `gcloud auth application-default login`.

---

## Testing the Technical Scanner

The scanner runs automatically inside the build pipeline, but you can test it independently.

### 1. Standalone scanner test (no backend needed)

```bash
cd scripts
python test_scanner.py                          # scores AAPL MSFT NVDA TSLA JPM
python test_scanner.py GOOGL META AMZN          # custom tickers
```

Expected output:
```
TEST 1 — Technical Scanner (standalone)
Ticker    Score  Pass (≥50)
--------------------------------
AAPL         75  ✓ PASS
MSFT         80  ✓ PASS
NVDA         60  ✓ PASS
TSLA         45  ✗ filtered
JPM          55  ✓ PASS

Tickers that would pass to Markowitz: 4/5
```

> Note: downloads 300 days of daily data via yfinance — allow ~3–5s per ticker.

### 2. Full pipeline test (backend must be running)

```bash
# Terminal 1 — start backend
cd backend && python main.py

# Terminal 2 — run full test
cd scripts
python test_scanner.py --full
python test_scanner.py --full --url http://localhost:8000
```

This sends a real build-mode `/chat` request and verifies `technical_score` appears in every portfolio position.

### 3. Watch scanner logs during a live build

When the backend is running, every build request prints scanner output:

```
INFO  Technical filter: 18 → 11 tickers    ← scanner dropped 7 weak tickers
INFO  Running Markowitz on 11 assets …
INFO  Optimal basket: 5 positions
```

Look for `Technical filter:` and `scan_tickers:` lines in the backend console.

### 4. Quick REPL check

```python
# from project root
import sys; sys.path.insert(0, 'backend')
from technical_scanner import scan_tickers
results = scan_tickers(['AAPL', 'MSFT', 'NVDA'], min_score=0)
print(results)
# [{'ticker': 'MSFT', 'technical_score': 80.0}, ...]
```

---

## Data Tagging Script

`scripts/tag_stocks.py` populates `ticker_grades` in BigQuery for any ticker not yet scored.  
It mirrors the original n8n DataTaggingWF pipeline using free, no-key data sources.

**Pipeline per ticker:**
1. SEC EDGAR free API → CIK lookup → most-recent 10-Q filing HTML
2. Strip HTML to clean natural-language text (same logic as the original n8n JS node)
3. yfinance → latest quarterly income statement (revenue, gross profit, net income, EPS)
4. OpenRouter LLM → `{"mark": 0-100, "explanation": "..."}` (same prompt as n8n AI Agent)
5. BigQuery INSERT → `ticker_grades`

```bash
cd scripts
python tag_stocks.py                  # auto: tags all untagged tickers in companies_risk_ratings
python tag_stocks.py NFLX SPOT UBER   # tag specific tickers (ignores BQ check)
```

Requires `OPEN_ROUTER_API_KEY` in `backend/.env` and BigQuery credentials.

---

## Working On This Project

### Recommended Local Flow

1. Create a feature branch from `main`.
2. Make changes (frontend in `frontend/`, backend in `backend/`).
3. Run local checks:
   ```bash
   python -m compileall backend src
   ```
4. Commit and push your branch.
5. Open a Pull Request to `main`.

### Frontend/Backend Responsibilities

- `frontend/index.html` is the static UI served by GitHub Pages.
- `backend/main.py` is the FastAPI API served on Render.
- Frontend sends requests to the deployed backend URL.

### Important

- Do not commit secrets (`.env`, API keys, service account JSON).
- Render environment variables are used for backend runtime secrets.

---

## Project Structure

```
FinalProject-RoboSmartInvestment/
│
├── backend/
│   ├── .env                    ← not committed; holds API keys
│   ├── main.py                 # FastAPI orchestration pipeline
│   ├── model_router.py         # OpenRouter model rotation on 429
│   ├── markowitz.py            # Markowitz max-Sharpe optimiser
│   ├── bigquery_client.py      # BigQuery helpers
│   ├── technical_scanner.py    # yfinance momentum/trend scorer (0-100)
│   ├── seed_demo.py            # One-time data seeder (run once)
│   └── requirements.txt
│
├── frontend/
│   └── index.html              # Self-contained SPA — open directly in browser
│
├── scripts/
│   ├── tag_stocks.py           # Populate ticker_grades (SEC EDGAR + yfinance + LLM)
│   └── test_scanner.py         # Smoke-test the technical scanner integration
│
├── notebooks/
│   ├── ScanerStock.ipynb       # Original stock scanner prototype (reference)
│   └── Markov_LM_toy.ipynb     # N-gram language model experiment (reference)
│
├── src/
│   ├── data_retrieval/         # Extended ETL scripts (1,001-stock universe)
│   └── decision_tree/
│       └── risk_classifier.py
│
├── data/
│   ├── tickers_top1000.txt
│   ├── tickers_training_200.txt
│   └── ticker_sector_training.csv
│
└── .gitignore
```

---

## Model Rotation

`backend/model_router.py` maintains a prioritised list of free OpenRouter models.  
When any model returns HTTP 429 (rate-limited), the router automatically advances to the next model.  
After exhausting all models it waits 8 seconds, resets, and retries from the top.

Current free-tier rotation order:
1. `openai/gpt-oss-120b:free`
2. `meta-llama/llama-3.3-70b-instruct:free`
3. `google/gemma-4-31b-it:free`
4. `google/gemma-4-26b-a4b-it:free`
5. `nousresearch/hermes-3-llama-3.1-405b:free`
6. `nvidia/nemotron-3-super-120b-a12b:free`
7. `qwen/qwen3-next-80b-a3b-instruct:free`
8. `google/gemma-3-27b-it:free`
9. `openai/gpt-oss-20b:free`
10. `google/gemma-3-12b-it:free`
11. `meta-llama/llama-3.2-3b-instruct:free`
12. `google/gemma-3-4b-it:free`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Returns server status |
| `POST` | `/chat`   | Main pipeline: message → portfolio response |

### POST /chat

**Request (conversational):**
```json
{ "message": "I want to invest $10,000 with medium risk", "session_id": "..." }
```

**Request (build mode — sent by UI button):**
```json
{
  "message": "Build my portfolio",
  "session_id": "...",
  "explicit_budget":   10000,
  "explicit_currency": "USD",
  "explicit_risk":     "Medium",
  "explicit_top_k":    5
}
```

**Response:**
```json
{
  "text": "Based on your medium-risk profile…",
  "portfolio": [
    {
      "ticker": "AAPL",
      "weight": 0.18,
      "score": 84,
      "technical_score": 75.0,
      "label": "Top Pick",
      "explanation": "Strong iPhone ecosystem…"
    }
  ],
  "risk_level": "Medium",
  "balance": 10000,
  "currency": "USD",
  "build_mode": true
}
```

**Portfolio labels:**

| Label | Condition |
|---|---|
| Top Pick | weight > 15% AND fundamental score ≥ 70 |
| Statistical Hold | weight > 15% AND fundamental score < 50 |
| Growth Opportunity | weight ≤ 15% AND fundamental score ≥ 70 |
| Balanced Position | everything else |

---

## GitHub Actions (CI/CD)

This repository uses two workflows:

1. `/.github/workflows/ci.yml`
   - Runs on push and pull requests.
   - Installs dependencies.
   - Validates Python files with `python -m compileall backend src`.
   - Runs `pip check`.

2. `/.github/workflows/frontend-pages.yml`
   - Runs automatically on every push to `main` (and manually with `workflow_dispatch`).
   - Builds a static artifact from `frontend/`.
   - Deploys to GitHub Pages.
   - This is the workflow responsible for frontend publishing.

If the Pages URL still shows stale content, verify this workflow's `deploy` job is green in the Actions tab.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML/CSS/JS · Chart.js · Inter + JetBrains Mono |
| Backend | Python 3.11+ · FastAPI · Uvicorn |
| LLM | OpenRouter (free-tier, auto-rotation) |
| Portfolio Optimisation | NumPy · SciPy SLSQP |
| Technical Analysis | Custom indicators (EMA, RSI, MACD, BB, CMF, RS) via yfinance |
| Data Storage | Google BigQuery (EU region) |
| Data Source | SEC EDGAR (free API) · Yahoo Finance (yfinance) |

*For educational purposes only. Not financial advice.*
