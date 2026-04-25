# RoboSmartInvest

**AI-Powered Stock Portfolio Builder**  
Ben-Gurion University of the Negev — Final Year Project

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

## Full Installation Guide

Follow these steps **in order** on a fresh machine. Steps 1–3 are one-time setup.

---

### Step 1 — Install Python 3.11+

Download and install from https://www.python.org/downloads/  
During installation, tick **"Add Python to PATH"**.

Verify:
```bash
python --version
# Python 3.11.x or higher
```

---

### Step 2 — Install Google Cloud CLI

The Google Cloud CLI (`gcloud`) is required for authenticating with BigQuery.

1. Download the installer from:  
   **https://cloud.google.com/sdk/docs/install**

2. Run the installer, keep all defaults, and let it add `gcloud` to PATH.

3. Open a **new** terminal and verify:
   ```bash
   gcloud --version
   ```

4. Log in to your Google account:
   ```bash
   gcloud auth login
   ```
   This opens a browser — sign in with the Google account that has access to the BigQuery project.

5. Set **Application Default Credentials** (this is a separate step from the login above — both are required):
   ```bash
   gcloud auth application-default login
   ```
   Sign in again in the browser. This creates the credentials file that Python libraries use automatically.

> **Why two logins?**  
> `gcloud auth login` lets the CLI tool talk to Google.  
> `gcloud auth application-default login` lets Python libraries (like `google-cloud-bigquery`) talk to Google on your behalf.  
> Both are needed.

---

### Step 3 — Get an OpenRouter API Key

OpenRouter provides free access to powerful LLMs (GPT, Llama, Gemma, etc.) with automatic fallback.

1. Go to **https://openrouter.ai** and create a free account.
2. Navigate to **Keys** → **Create Key**.
3. Copy the key — it looks like `sk-or-v1-xxxxxxxxxxxxxxxx`.

---

### Step 4 — Clone the repository

```bash
git clone <repo-url>
cd FinalProject-RoboSmartInvestment
```

---

### Step 5 — Create the `.env` file

Create a file named **`.env`** inside the **`backend/`** folder:

```
backend/.env
```

Paste the following content, replacing the placeholder with your real OpenRouter key:

```env
OPEN_ROUTER_API_KEY=sk-or-v1-your-key-here
```

> The file must be named exactly `.env` (dot prefix, no extension) and placed inside `backend/`.  
> It is already listed in `.gitignore` and will never be committed to git.

---

### Step 6 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs FastAPI, BigQuery client, NumPy, SciPy, pandas, and all other required packages.

---

### Step 7 — Populate BigQuery (one-time data seed)

This step downloads 3 years of real stock price data via Yahoo Finance and uploads it to BigQuery.  
**Run once — skip on subsequent runs.**

```bash
# From the backend/ directory
python seed_demo.py
```

This creates three tables in BigQuery (`StockData` dataset):
- `daily_prices` — ~40,000 rows of daily close prices for 60 large-cap stocks
- `companies_risk_ratings` — each stock labelled Low / Med-Low / Medium / Med-High / High
- `ticker_grades` — analyst-style fundamental score (0–100) + explanation per stock

Expected output:
```
✅  Seeding complete!
    daily_prices           → 40500 rows
    companies_risk_ratings → 54 tickers
    ticker_grades          → 54 tickers
```

---

### Step 8 — Start the backend server

```bash
# From the backend/ directory
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Or simply:
```bash
python main.py
```

Verify the server is running — open this URL in your browser:  
**http://localhost:8000/health**

You should see:
```json
{ "status": "ok", "model": "openai/gpt-oss-120b:free" }
```

---

### Step 9 — Open the frontend

Open `frontend/index.html` directly in your browser — no build step, no local server needed.  
On Windows: double-click the file, or drag it into Chrome/Edge/Firefox.

Scroll down past the hero section to reach the chat interface.  
Type something like:

> *"I want to invest $10,000 with medium risk"*

---

## Credentials Summary

| Credential | Where to Get | Where to Put |
|---|---|---|
| OpenRouter API key | https://openrouter.ai → Keys → Create Key | `backend/.env` as `OPEN_ROUTER_API_KEY=sk-or-v1-...` |
| Google Cloud credentials | `gcloud auth application-default login` | Stored automatically by gcloud (no file to manage) |
| BigQuery project access | Project owner grants `roles/bigquery.dataViewer` | Handled via gcloud login |

---

## Project Structure

```
FinalProject-RoboSmartInvestment/
│
├── backend/
│   ├── .env                  ← CREATE THIS — your OpenRouter API key goes here
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

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML/CSS/JS · Chart.js · Inter + JetBrains Mono |
| Backend | Python 3.11+ · FastAPI · Uvicorn |
| LLM | OpenRouter (free-tier, auto-rotation) |
| Portfolio Optimisation | NumPy · SciPy SLSQP |
| Data Storage | Google BigQuery (EU region) |
| Data Source | Yahoo Finance (yfinance) |

---

## Academic Context

Developed as a Final Year Project at **Ben-Gurion University of the Negev**, combining:
- Financial Theory (Markowitz Modern Portfolio Theory)
- Natural Language Processing (LLM intent extraction + synthesis)
- Data Engineering (BigQuery, ETL pipelines)

---

*For educational purposes only. Not financial advice.*
