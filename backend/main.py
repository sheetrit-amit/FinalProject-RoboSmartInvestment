"""
RoboSmartInvestment — FastAPI Backend v2.2
==========================================
Two-mode /chat endpoint:
  CONVERSATIONAL mode (default): collect params through dialogue, no portfolio built.
  BUILD mode (explicit params from UI button): run full Markowitz pipeline.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_here, ".env")) or \
load_dotenv(dotenv_path=os.path.join(_here, "..", ".env"))

from bigquery_client import (
    get_bigquery_client,
    get_bq_context,
    get_fundamental_scores,
    get_tickers_by_risk,
)
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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="RoboSmartInvestment API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model router
# ---------------------------------------------------------------------------
_api_key = os.getenv("OPEN_ROUTER_API_KEY", "")
if not _api_key:
    logger.warning("OPEN_ROUTER_API_KEY is not set — LLM calls will fail.")
else:
    logger.info("API key loaded: %s… (len=%d)", _api_key[:12], len(_api_key))

router = ModelRouter(api_key=_api_key)

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_sessions: Dict[str, dict] = {}
_SESSION_TTL  = timedelta(hours=4)
_MAX_HISTORY  = 16


def _get_session(session_id: str) -> dict:
    now = datetime.utcnow()
    for k in list(_sessions):
        if now - _sessions[k]["ts"] > _SESSION_TTL:
            del _sessions[k]
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "ts": now}
    else:
        _sessions[session_id]["ts"] = now
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM = """You are a friendly AI investment advisor for RoboSmartInvest.

Your job is to have a warm, natural conversation to understand the user's investment goals
and collect three pieces of information before a portfolio can be built:
  1. Budget — how much they want to invest (amount + currency)
  2. Risk level — Low | Med-Low | Medium | Med-High | High
  3. Number of stocks — how many distinct positions (2–15; default suggestion: 5)

Rules:
- Be concise (2–4 sentences). Do NOT write walls of text.
- If any of the three params are still missing, ask about them naturally.
- Once all three are confirmed, tell the user the parameters are set and invite them
  to click the "Build My Portfolio" button that appeared below the chat.
- Do NOT invent or suggest stock names, weights, or portfolio contents — that is handled
  by the system when the button is pressed.
- If the user asks something unrelated to investing, gently redirect.
"""

EXTRACT_SYSTEM = (
    "You are a financial-intent parser. "
    "Output ONLY a valid JSON object — no markdown, no extra keys, no explanation.\n\n"
    "Extract from the user message:\n"
    '  "balance"  : number or null   (investment amount; midpoint if range)\n'
    '  "currency" : string or null   (e.g. "USD", "EUR", "ILS")\n'
    '  "risk"     : one of "Low"|"Med-Low"|"Medium"|"Med-High"|"High" or null\n'
    '  "top_k"    : integer or null  (number of stocks, e.g. "5 stocks", "7 positions")\n\n'
    'Set null for anything not explicitly mentioned.\n'
    'JSON only. Example: {"balance": 5000, "currency": "USD", "risk": "Medium", "top_k": null}'
)

SYNTHESIS_SYSTEM = """You are a Senior Investment Strategist for a Robo-Advisor.
Synthesise a concise, user-friendly portfolio recommendation from two data sources.

DATA SOURCE 1 — Markowitz (Quantitative):
Weights maximising Sharpe ratio. High weight ≠ low risk — it means best risk-adjusted return historically.

DATA SOURCE 2 — Fundamental Scores (0-100):
LLM-generated scores from recent 10-K/10-Q filings.

LABELLING:
- weight >15% AND score ≥70  → "Top Pick"
- weight >15% AND score <50  → "Statistical Hold"
- weight ≤15% AND score ≥70  → "Growth Opportunity"
- otherwise                  → "Balanced Position"

WRITING RULES:
- 6-10 sentences, paragraph style (no bullet points)
- Bold only ticker symbols with **TICKER**
- Focus on WHY each key position is included
- Close with one risk disclaimer sentence
- Do NOT invent data not present in the inputs
- If conversation history is present, personalise the response
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_VALID_RISKS  = {"Low", "Med-Low", "Medium", "Med-High", "High"}
_MAX_TOP_K    = 15


def _label(weight: float, score: Optional[float]) -> str:
    if score is None:        return "Quantitative Pick"
    if weight > 0.15 and score >= 70: return "Top Pick"
    if weight > 0.15 and score < 50:  return "Statistical Hold"
    if weight <= 0.15 and score >= 70: return "Growth Opportunity"
    return "Balanced Position"


def _build_fund_text(fundamentals: List[Dict[str, Any]]) -> str:
    if not fundamentals:
        return "(no fundamental data available)"
    return "\n\n".join(
        f"{f['ticker']} | score: {f['mark']} | {str(f.get('explanation') or '')[:300].replace(chr(10),' ')}"
        for f in fundamentals
    )


def _renormalize(weights: List[Dict]) -> List[Dict]:
    total = sum(w["weight"] for w in weights)
    if total <= 0:
        return weights
    return [{"ticker": w["ticker"], "weight": round(w["weight"] / total, 4)} for w in weights]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # Explicit params — set by UI when user clicks "Build My Portfolio"
    explicit_budget:   Optional[float] = None
    explicit_currency: Optional[str]   = None
    explicit_risk:     Optional[str]   = None
    explicit_top_k:    Optional[int]   = None


class ChatResponse(BaseModel):
    text: str
    portfolio:       List[Dict[str, Any]] = []
    risk_level:      str = ""
    balance:         Optional[float] = None
    currency:        Optional[str]   = None
    params_detected: Dict[str, Any]  = {}
    build_mode:      bool = False   # tells frontend this was a real portfolio build


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/session/new")
def new_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = {"messages": [], "ts": datetime.utcnow()}
    logger.info("New session: %s", sid)
    return {"session_id": sid}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, http_response: Response):
    logger.info("→ /chat  session=%s  message=%r", body.session_id, body.message[:80])
    router.was_exhausted = False

    session       = _get_session(body.session_id) if body.session_id else {"messages": []}
    recent_history = session["messages"][-_MAX_HISTORY:]

    build_mode = (body.explicit_risk is not None and body.explicit_top_k is not None)

    # ── CONVERSATIONAL MODE ──────────────────────────────────────────────────
    if not build_mode:
        # Step A: extract any params mentioned in the message
        try:
            intent = router.chat_json(
                [
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user",   "content": body.message},
                ],
                temperature=0.05,
                max_tokens=120,
            )
        except Exception:
            intent = {}

        params_detected = {
            "balance":  intent.get("balance"),
            "currency": intent.get("currency"),
            "risk":     intent.get("risk") if intent.get("risk") in _VALID_RISKS else None,
            "top_k":    int(intent["top_k"]) if intent.get("top_k") else None,
        }

        # Step B: conversational reply
        try:
            text = router.chat(
                [
                    {"role": "system", "content": ADVISOR_SYSTEM},
                    *recent_history,
                    {"role": "user",   "content": body.message},
                ],
                temperature=0.55,
                max_tokens=280,
            )
        except Exception as exc:
            logger.error("Advisor LLM failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

        if body.session_id:
            session["messages"].append({"role": "user",      "content": body.message})
            session["messages"].append({"role": "assistant", "content": text})
            session["messages"] = session["messages"][-_MAX_HISTORY:]

        if router.was_exhausted:
            http_response.headers["X-Overloaded"] = "true"

        return ChatResponse(text=text, params_detected=params_detected, build_mode=False)

    # ── BUILD MODE (explicit params from button) ─────────────────────────────
    balance  = body.explicit_budget
    currency = body.explicit_currency or "USD"
    risk     = body.explicit_risk if body.explicit_risk in _VALID_RISKS else "Medium"
    top_k    = max(1, min(body.explicit_top_k or 5, _MAX_TOP_K))

    # 0. BQ context probe
    try:
        bq     = get_bigquery_client()
        bq_ctx = get_bq_context(bq)
    except Exception as exc:
        logger.warning("BQ context probe failed: %s", exc)
        bq     = None
        bq_ctx = {}

    # 1. Fetch tickers
    try:
        if bq is None:
            bq = get_bigquery_client()
        tickers = get_tickers_by_risk(risk, bq)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not tickers:
        raise HTTPException(status_code=404,
            detail=f"No tickers for risk level '{risk}'. Ensure the DB is seeded.")

    # 2. Markowitz + top-k
    try:
        weights = run_markowitz(tickers, bq)
        weights = weights[:top_k]
        # If optimizer concentrated on fewer stocks than requested, pad with
        # remaining tickers at equal small weight so the user gets top_k positions.
        used = {w["ticker"] for w in weights}
        remaining = [t for t in tickers if t not in used]
        fill_count = top_k - len(weights)
        if fill_count > 0 and remaining:
            small_w = (1.0 / (top_k * 4))  # small equal nudge weight
            for t in remaining[:fill_count]:
                weights.append({"ticker": t, "weight": small_w})
        weights = _renormalize(weights)
        logger.info("Markowitz → %d positions", len(weights))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Portfolio optimisation failed: {exc}")

    # 3. Fundamental scores
    opt_tickers = [w["ticker"] for w in weights]
    try:
        fundamentals = get_fundamental_scores(opt_tickers, bq)
    except Exception:
        fundamentals = []

    fund_map  = {f["ticker"]: f for f in fundamentals}
    portfolio = []
    for w in weights:
        f     = fund_map.get(w["ticker"], {})
        score = f.get("mark")
        portfolio.append({
            "ticker":      w["ticker"],
            "weight":      w["weight"],
            "score":       score,
            "explanation": f.get("explanation"),
            "label":       _label(w["weight"], score),
        })

    # 4. LLM synthesis
    markowitz_text = "\n".join(f"{w['ticker']}: {w['weight']*100:.2f}%" for w in weights)
    fund_text      = _build_fund_text(fundamentals)
    user_ctx = (
        f"User profile — Risk: {risk}, Stocks requested: {top_k}"
        + (f", Amount: {balance:,.0f} {currency}" if balance else "")
        + f"\n\nMarkowitz weights:\n{markowitz_text}"
        + f"\n\nFundamental scores:\n{fund_text}"
    )

    try:
        text = router.chat(
            [
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                *recent_history,
                {"role": "user",   "content": user_ctx},
            ],
            temperature=0.4,
            max_tokens=900,
        )
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        top5 = ", ".join(f"**{w['ticker']}**" for w in weights[:5])
        text = (
            f"Portfolio optimised for **{risk}** risk with {top_k} positions. "
            f"Top holdings: {top5}. Past performance does not guarantee future results."
        )

    if body.session_id:
        session["messages"].append({"role": "user",      "content": body.message})
        session["messages"].append({"role": "assistant", "content": text})
        session["messages"] = session["messages"][-_MAX_HISTORY:]

    if router.was_exhausted:
        http_response.headers["X-Overloaded"] = "true"

    logger.info("← /chat  build=True  positions=%d", len(portfolio))
    return ChatResponse(
        text=text,
        portfolio=portfolio,
        risk_level=risk,
        balance=balance,
        currency=currency,
        params_detected={"balance": balance, "currency": currency, "risk": risk, "top_k": top_k},
        build_mode=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
