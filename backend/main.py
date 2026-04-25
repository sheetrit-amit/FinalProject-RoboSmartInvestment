"""
RoboSmartInvestment — FastAPI Backend
======================================
Replaces the entire n8n workflow with a single, self-contained Python service.

Pipeline (one HTTP POST /chat):
  1. LLM: extract { balance, currency, risk } from the user message
  2. BigQuery: fetch tickers matching the risk level
  3. Markowitz: compute optimal portfolio weights
  4. BigQuery: fetch fundamental scores for the selected tickers
  5. LLM: synthesise a human-readable recommendation
  6. Return structured JSON (text + portfolio data for charting)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env — check backend/ first, then project root
_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_here, ".env")) or \
load_dotenv(dotenv_path=os.path.join(_here, "..", ".env"))

from bigquery_client import get_bigquery_client, get_fundamental_scores, get_tickers_by_risk
from markowitz import run_markowitz
from model_router import ModelRouter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("robosmart")

# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RoboSmartInvestment API",
    version="2.0.0",
    description="AI-powered portfolio optimisation service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model router (singleton)
# ---------------------------------------------------------------------------

_api_key = os.getenv("OPEN_ROUTER_API_KEY", "")
if not _api_key:
    logger.warning("OPEN_ROUTER_API_KEY is not set — LLM calls will fail.")

router = ModelRouter(api_key=_api_key)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You are a financial-intent parser. "
    "Output ONLY a valid JSON object — no markdown, no extra keys, no explanation.\n\n"
    "Extract from the user message:\n"
    '  "balance"  : number or null   (investment amount; midpoint if a range)\n'
    '  "currency" : string or null   (e.g. "USD", "EUR", "ILS")\n'
    '  "risk"     : one of "Low" | "Med-Low" | "Medium" | "Med-High" | "High"\n\n'
    'If risk is not mentioned, default to "Medium".\n'
    'If balance is not mentioned, set null.\n'
    'JSON only. Example: {"balance": 5000, "currency": "USD", "risk": "Medium"}'
)

SYNTHESIS_SYSTEM = """You are a Senior Investment Strategist for a Robo-Advisor.
Synthesise a concise, user-friendly portfolio recommendation from two data sources.

DATA SOURCE 1 — Markowitz (Quantitative):
Weights maximising Sharpe ratio. High weight ≠ low risk — it means this stock
provides the best risk-adjusted return historically (efficient frontier logic).

DATA SOURCE 2 — Fundamental Scores (0-100):
LLM-generated scores from recent 10-K/10-Q filings assessing growth, sentiment, risk.

LABELLING:
- weight >15 % AND score ≥70  → "Top Pick"    (double win: math + fundamentals)
- weight >15 % AND score <50  → "Statistical Hold" (caution: watch closely)
- weight ≤15 % AND score ≥70  → "Growth Opportunity" (volatile but strong fundamentals)
- otherwise                   → "Balanced Position"

WRITING RULES:
- 6-10 sentences, paragraph style (not bullet points)
- Bold only ticker symbols with **TICKER**
- Focus on WHY each key position is included
- Close with one sentence risk disclaimer
- Do NOT invent data not present in the inputs
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RISKS = {"Low", "Med-Low", "Medium", "Med-High", "High"}


def _label(weight: float, score: Optional[float]) -> str:
    if score is None:
        return "Quantitative Pick"
    if weight > 0.15 and score >= 70:
        return "Top Pick"
    if weight > 0.15 and score < 50:
        return "Statistical Hold"
    if weight <= 0.15 and score >= 70:
        return "Growth Opportunity"
    return "Balanced Position"


def _build_fund_text(fundamentals: List[Dict[str, Any]]) -> str:
    if not fundamentals:
        return "(no fundamental data available)"
    lines = []
    for f in fundamentals:
        expl = str(f.get("explanation") or "").replace("\n", " ")[:300]
        lines.append(f"{f['ticker']} | score: {f['mark']} | {expl}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class PortfolioItem(BaseModel):
    ticker: str
    weight: float
    score: Optional[float] = None
    explanation: Optional[str] = None
    label: str = "Balanced Position"


class ChatResponse(BaseModel):
    text: str
    portfolio: List[Dict[str, Any]]
    risk_level: str
    balance: Optional[float] = None
    currency: Optional[str] = None
    model_used: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model": router.current_model}


@app.get("/models")
def list_models():
    return {
        "models": ModelRouter.FREE_MODELS,
        "current": router.current_model,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    logger.info("→ /chat  message=%r", body.message[:80])

    # ── 1. Extract intent ────────────────────────────────────────────────────
    try:
        intent = router.chat_json(
            [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user",   "content": body.message},
            ],
            temperature=0.05,
            max_tokens=120,
        )
        logger.info("Intent extracted: %s", intent)
    except Exception as exc:
        logger.error("Intent extraction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Intent extraction failed: {exc}")

    balance  = intent.get("balance")
    currency = intent.get("currency") or "USD"
    risk     = intent.get("risk", "Medium")

    if risk not in _VALID_RISKS:
        risk = "Medium"

    # ── 2. Fetch tickers from BigQuery ───────────────────────────────────────
    try:
        bq = get_bigquery_client()
        tickers = get_tickers_by_risk(risk, bq)
    except Exception as exc:
        logger.error("BigQuery ticker fetch failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable — check Google Cloud credentials: {exc}",
        )

    if not tickers:
        raise HTTPException(
            status_code=404,
            detail=f"No tickers found for risk level '{risk}'. "
                   "Ensure the companies_risk_ratings table is populated.",
        )

    # ── 3. Markowitz optimisation ────────────────────────────────────────────
    try:
        weights = run_markowitz(tickers, bq)
        logger.info("Markowitz → %d positions", len(weights))
    except Exception as exc:
        logger.error("Markowitz failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Portfolio optimisation failed: {exc}")

    # ── 4. Fundamental scores ────────────────────────────────────────────────
    opt_tickers = [w["ticker"] for w in weights]
    try:
        fundamentals = get_fundamental_scores(opt_tickers, bq)
    except Exception as exc:
        logger.warning("Could not fetch fundamental scores: %s", exc)
        fundamentals = []

    fund_map = {f["ticker"]: f for f in fundamentals}

    # ── 5. Build portfolio payload ───────────────────────────────────────────
    portfolio = []
    for w in weights:
        f     = fund_map.get(w["ticker"], {})
        score = f.get("mark")
        portfolio.append(
            {
                "ticker":      w["ticker"],
                "weight":      w["weight"],
                "score":       score,
                "explanation": f.get("explanation"),
                "label":       _label(w["weight"], score),
            }
        )

    # ── 6. LLM synthesis ─────────────────────────────────────────────────────
    markowitz_text = "\n".join(
        f"{w['ticker']}: {w['weight'] * 100:.2f}%" for w in weights
    )
    fund_text = _build_fund_text(fundamentals)

    user_ctx = (
        f"User profile — Risk: {risk}"
        + (f", Amount: {balance:,.0f} {currency}" if balance else "")
        + f"\n\nMarkowitz weights:\n{markowitz_text}"
        + f"\n\nFundamental scores:\n{fund_text}"
    )

    try:
        text = router.chat(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user",   "content": user_ctx},
            ],
            temperature=0.4,
            max_tokens=900,
        )
    except Exception as exc:
        logger.error("Synthesis LLM call failed: %s", exc)
        top5 = ", ".join(f"**{w['ticker']}**" for w in weights[:5])
        text = (
            f"Portfolio optimised for **{risk}** risk. "
            f"Top positions: {top5}. "
            "Past performance does not guarantee future results. "
            "Please consult a licensed financial adviser before investing."
        )

    logger.info("← /chat  model=%s  positions=%d", router.current_model, len(portfolio))
    return ChatResponse(
        text=text,
        portfolio=portfolio,
        risk_level=risk,
        balance=balance,
        currency=currency,
        model_used=router.current_model,
    )


# ---------------------------------------------------------------------------
# Entry point (python backend/main.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
