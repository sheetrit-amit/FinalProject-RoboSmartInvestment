# RoboSmartInvest

**AI-Powered Stock Portfolio Builder**

> Tell the system your investment amount and risk tolerance.  
> Get back an optimised portfolio with AI-generated investment reasoning.

---

## What It Does

1. **Understands your intent** — An LLM extracts your balance, currency, and risk tolerance from natural language.
2. **Classifies risk** — Stocks in BigQuery are pre-labelled Low / Med-Low / Medium / Med-High / High using volatility-based quintiles.
3. **Optimises the portfolio** — Markowitz mean-variance optimisation finds weights that maximise the Sharpe Ratio on the Efficient Frontier.
4. **Adds fundamental context** — Pre-computed analyst scores are attached to each position.
5. **Synthesises a recommendation** — A final LLM call writes a human-readable investment thesis for the basket.

---

## Architecture

```
Browser (frontend/index.html)
        │  POST /chat  { message }
        ▼
FastAPI Backend (backend/main.py)
   ├─ ModelRouter          (OpenRouter free-tier, auto-rotates on 429)
   ├─ BigQuery client       (ticker_grades, companies_risk_ratings, daily_prices)
   └─ Markowitz optimizer   (SciPy SLSQP, max-Sharpe)
        │
        ▼
Response  { text, portfolio[], risk_level, balance, currency, model_used }
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
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

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
│   ├── .env                  ← optional for self-hosting only (not committed)
│   ├── main.py               # FastAPI orchestration pipeline
│   ├── model_router.py       # OpenRouter model rotation on 429
│   ├── markowitz.py          # Markowitz max-Sharpe optimiser
│   ├── bigquery_client.py    # BigQuery helpers
│   ├── seed_demo.py          # One-time data seeder (run once)
│   └── requirements.txt
│
├── frontend/
│   └── index.html            # Self-contained SPA — open directly in browser
│
├── src/
│   ├── data_retrieval/       # Extended ETL scripts (1,001-stock universe)
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
| `GET`  | `/health` | Returns server status + active model |
| `GET`  | `/models` | Lists all models + currently active one |
| `POST` | `/chat`   | Main pipeline: message → portfolio response |

### POST /chat

**Request:**
```json
{ "message": "I want to invest $10,000 with medium risk" }
```

**Response:**
```json
{
  "text": "Based on your medium-risk profile…",
  "portfolio": [
    { "ticker": "AAPL", "weight": 0.18, "score": 84, "label": "Top Pick", "explanation": "…" }
  ],
  "risk_level": "Medium",
  "balance": 10000,
  "currency": "USD",
  "model_used": "openai/gpt-oss-120b:free"
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
| Data Storage | Google BigQuery (EU region) |
| Data Source | Yahoo Finance (yfinance) |

*For educational purposes only. Not financial advice.*
