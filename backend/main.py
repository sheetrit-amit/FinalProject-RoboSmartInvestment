"""
RoboSmartInvestment — FastAPI Backend v2.2
==========================================
Two-mode /chat endpoint:
  CONVERSATIONAL mode (default): collect params through dialogue, no portfolio built.
  BUILD mode (explicit params from UI button): run full Markowitz pipeline.
"""

import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

_here = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_here, ".env")) or \
load_dotenv(dotenv_path=os.path.join(_here, "..", ".env"))

from bigquery_client import (
    get_bigquery_client,
    get_fundamental_scores,
    get_tickers_by_risk,
)
from markowitz import run_markowitz
from model_router import ModelRouter
from technical_scanner import scan_tickers
from usage_logger import log_output_async, log_usage_async

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
_ALLOWED_ORIGINS = [
    "https://sheetrit-amit.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",   # Live Server / VSCode
    "null",                    # file:// double-click on Windows/macOS
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
# BigQuery client singleton (one connection pool for the process lifetime)
# ---------------------------------------------------------------------------
_bq_client = None
_bq_lock = threading.Lock()


def _get_bq() -> Any:
    global _bq_client
    if _bq_client is None:
        with _bq_lock:
            if _bq_client is None:
                _bq_client = get_bigquery_client()
    return _bq_client


# ---------------------------------------------------------------------------
# Technical scan cache — TTL-keyed by risk level
# Prevents O(N_users) redundant yfinance downloads for the same risk bucket.
# ---------------------------------------------------------------------------
_scan_cache: Dict[str, Any] = {}
_scan_cache_lock = threading.Lock()
_SCAN_TTL = timedelta(minutes=30)


def _cached_scan(risk: str, tickers: List[str]) -> List[Dict[str, Any]]:
    with _scan_cache_lock:
        if risk in _scan_cache:
            ts, cached = _scan_cache[risk]
            if datetime.utcnow() - ts < _SCAN_TTL:
                logger.info("Technical scan cache HIT for risk=%s (%d tickers)", risk, len(cached))
                return cached

    results = scan_tickers(tickers)

    with _scan_cache_lock:
        _scan_cache[risk] = (datetime.utcnow(), results)
    return results


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
_sessions: Dict[str, dict] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL  = timedelta(hours=4)
_MAX_HISTORY  = 16


def _purge_expired_sessions(now: datetime) -> None:
    """Remove sessions older than SESSION_TTL. Caller must hold _sessions_lock."""
    for k in [k for k, v in _sessions.items() if now - v["ts"] > _SESSION_TTL]:
        del _sessions[k]


def _get_session(session_id: str) -> dict:
    now = datetime.utcnow()
    with _sessions_lock:
        _purge_expired_sessions(now)
        if session_id not in _sessions:
            _sessions[session_id] = {
                "messages": [],
                "ts": now,
                "portfolio_built": False,
                "post_build_count": 0,
            }
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
  3. Number of stocks — how many distinct positions (2–50; default suggestion: 5)

Rules:
- Be concise (2–4 sentences). Do NOT write walls of text.
- If any of the three params are still missing, ask about them naturally.
- If the user requests more than 50 stocks, confirm you've capped it to 50 (the maximum) — do NOT ask them to re-specify.
- Once all three are confirmed, tell the user the parameters are set and invite them
  to click the "Build My Portfolio" button that appeared below the chat.
- Do NOT invent or suggest stock names, weights, or portfolio contents — that is handled
  by the system when the button is pressed.
- If the user asks something unrelated to investing, gently redirect.
"""

DISCUSSION_SYSTEM = """You are a friendly AI investment advisor for RoboSmartInvest.

A portfolio has already been built for this user in the current conversation.
Your role is to discuss and explain the portfolio based on what appears in the conversation history.

Rules:
- Be concise (3–5 sentences). Reference the actual tickers, weights, and scores from the history.
- Explain decisions using Markowitz (Sharpe ratio optimisation), fundamental scores, and technical scores.
- If asked to build a new portfolio, tell the user to update the parameters in the drawer and click "Build My Portfolio" again.
- Do NOT invent data not present in the conversation history.
- Do NOT ask for budget/risk/stock count again — those were already collected.
- Do not promise future returns or give specific buy/sell advice.
"""

_POST_BUILD_LIMIT = 10
_POST_BUILD_LIMIT_MSG = (
    "You've reached the 10-message post-portfolio limit for this session. "
    "To continue exploring, please refresh the page to start a new session, "
    "or click \"Build My Portfolio\" again to generate a new portfolio."
)

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

DATA SOURCE 3 — Technical Scores (0-100):
Momentum/trend scoring: EMA trend, MACD, RSI divergence, relative strength vs S&P 500,
volume signals, VCP squeeze. Stocks below 50 were pre-filtered out.

LABELLING:
- weight >15% AND fund_score ≥70  → "Top Pick"
- weight >15% AND fund_score <50  → "Statistical Hold"
- weight ≤15% AND fund_score ≥70  → "Growth Opportunity"
- otherwise                       → "Balanced Position"

WRITING RULES:
- 6-10 sentences, paragraph style (no bullet points)
- Bold only ticker symbols with **TICKER**
- Focus on WHY each key position is included
- STOCK COUNT: Always address the number of positions delivered vs what the user requested.
  If delivered == requested: briefly confirm (e.g. "Here are your X positions as requested.").
  If delivered < requested: explain honestly that the investable universe for the chosen
  risk level currently holds only that many names, so the portfolio uses all of them —
  and invite the user to try a different risk level for a wider selection.
- Close with one risk disclaimer sentence
- Do NOT invent data not present in the inputs
- If conversation history is present, personalise the response
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_VALID_RISKS  = {"Low", "Med-Low", "Medium", "Med-High", "High"}
_MAX_TOP_K    = 50
_MIN_OPTIMIZER_TICKERS = 2  # Markowitz needs at least 2 names to optimise


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


def _equal_weight_fallback(tickers: List[str], top_k: int) -> List[Dict[str, Any]]:
    """Equal-weight basket over up to *top_k* tickers.

    Last-resort allocation so a build always returns a portfolio even when
    Markowitz optimisation cannot run (e.g. too few usable price series).
    """
    picks = tickers[:max(1, top_k)]
    if not picks:
        return []
    w = 1.0 / len(picks)
    return [{"ticker": t, "weight": w} for t in picks]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(max_length=2000)
    session_id: Optional[str] = None
    # Explicit params — set by UI when user clicks "Build My Portfolio"
    explicit_budget:   Optional[float] = Field(None, gt=0, le=1_000_000_000)
    explicit_currency: Optional[str]   = Field(None, max_length=10)
    explicit_risk:     Optional[str]   = Field(None, max_length=20)
    explicit_top_k:    Optional[int]   = Field(None, ge=1, le=50)

    @field_validator("explicit_budget")
    @classmethod
    def budget_must_be_finite(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not math.isfinite(v):
            raise ValueError("budget must be a finite number")
        return v


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
    now = datetime.utcnow()
    with _sessions_lock:
        _purge_expired_sessions(now)
        _sessions[sid] = {
            "messages": [],
            "ts": now,
            "portfolio_built": False,
            "post_build_count": 0,
        }
    logger.info("New session: %s", sid)
    return {"session_id": sid}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, http_response: Response):
    """Public endpoint: delegates to the handler and logs exactly one usage row
    per request (success or error) without adding latency."""
    start = time.perf_counter()
    token_usage: Dict[str, Any] = {
        "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "calls": 0, "model": None,
    }
    build_mode = body.explicit_risk is not None
    usage: Dict[str, Any] = {
        "event_id":   str(uuid.uuid4()),
        "event_time": datetime.now(timezone.utc).isoformat(),
        "session_id": body.session_id,
        "user_message": body.message,
        "mode":       "build" if build_mode else "conversational",
        "status":     "ok",
        "error":      None,
        "response_text": None,
        "risk":       None, "budget": None, "currency": None, "top_k": None,
        "stocks_delivered": None, "holdings": [],
    }
    try:
        return _run_chat(body, http_response, build_mode, token_usage, usage)
    except HTTPException as exc:
        usage["status"] = "error"
        usage["error"]  = str(exc.detail)
        raise
    except Exception as exc:
        usage["status"] = "error"
        usage["error"]  = str(exc)
        raise
    finally:
        usage["latency_ms"]        = int((time.perf_counter() - start) * 1000)
        usage["overloaded"]        = router.was_exhausted
        usage["model_used"]        = token_usage["model"]
        usage["prompt_tokens"]     = token_usage["prompt_tokens"]
        usage["completion_tokens"] = token_usage["completion_tokens"]
        usage["total_tokens"]      = token_usage["total_tokens"]
        usage["llm_calls"]         = token_usage["calls"]
        log_usage_async(_get_bq, usage)
        log_output_async(_get_bq, {
            "event_id":           usage["event_id"],
            "event_time":         usage["event_time"],
            "response_text":      usage.get("response_text"),
            "requested_risk":     usage["risk"],
            "requested_budget":   usage["budget"],
            "requested_currency": usage["currency"],
            "requested_top_k":    usage["top_k"],
        })


def _run_chat(
    body: ChatRequest,
    http_response: Response,
    build_mode: bool,
    token_usage: Dict[str, Any],
    usage: Dict[str, Any],
) -> ChatResponse:
    logger.info("→ /chat  session=%s  message=%r", body.session_id, body.message[:80])
    router.was_exhausted = False  # reset per-thread flag for this request

    session       = _get_session(body.session_id) if body.session_id else {
        "messages": [], "portfolio_built": False, "post_build_count": 0
    }
    recent_history = session["messages"][-_MAX_HISTORY:]

    # ── CONVERSATIONAL MODE ──────────────────────────────────────────────────
    if not build_mode:
        portfolio_built = session.get("portfolio_built", False)

        # Enforce post-build message limit
        if portfolio_built:
            count = session.get("post_build_count", 0)
            if count >= _POST_BUILD_LIMIT:
                usage["response_text"] = _POST_BUILD_LIMIT_MSG
                return ChatResponse(text=_POST_BUILD_LIMIT_MSG, build_mode=False)
            session["post_build_count"] = count + 1

        # Step A: extract params only when no portfolio has been built yet
        params_detected: Dict[str, Any] = {}
        if not portfolio_built:
            try:
                intent = router.chat_json(
                    [
                        {"role": "system", "content": EXTRACT_SYSTEM},
                        {"role": "user",   "content": body.message},
                    ],
                    temperature=0.05,
                    max_tokens=120,
                    usage_accumulator=token_usage,
                )
            except Exception:
                intent = {}

            raw_top_k = intent.get("top_k")
            params_detected = {
                "balance":  intent.get("balance"),
                "currency": intent.get("currency"),
                "risk":     intent.get("risk") if intent.get("risk") in _VALID_RISKS else None,
                "top_k":    min(int(raw_top_k), _MAX_TOP_K) if raw_top_k else None,
            }
            usage["risk"]     = params_detected["risk"]
            usage["budget"]   = params_detected["balance"]
            usage["currency"] = params_detected["currency"]
            usage["top_k"]    = params_detected["top_k"]

        # Step B: conversational reply — use discussion mode when portfolio already exists
        system_prompt = DISCUSSION_SYSTEM if portfolio_built else ADVISOR_SYSTEM
        try:
            text = router.chat(
                [
                    {"role": "system", "content": system_prompt},
                    *recent_history,
                    {"role": "user",   "content": body.message},
                ],
                temperature=0.55,
                max_tokens=400,
                usage_accumulator=token_usage,
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

        usage["response_text"] = text
        return ChatResponse(text=text, params_detected=params_detected, build_mode=False)

    # ── BUILD MODE (explicit params from button) ─────────────────────────────
    balance  = body.explicit_budget
    currency = body.explicit_currency or "USD"
    risk     = body.explicit_risk if body.explicit_risk in _VALID_RISKS else "Medium"
    top_k    = max(1, min(body.explicit_top_k, _MAX_TOP_K)) if body.explicit_top_k is not None else None

    usage["risk"]     = risk
    usage["budget"]   = balance
    usage["currency"] = currency
    usage["top_k"]    = top_k

    # 1. Fetch tickers
    try:
        bq = _get_bq()
        tickers = get_tickers_by_risk(risk, bq)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")

    if not tickers:
        raise HTTPException(status_code=404,
            detail=f"No tickers for risk level '{risk}'. Ensure the DB is seeded.")

    # 1b. Technical pre-filter (yfinance-based scoring, cached 30 min per risk level).
    #     The scan is a *preference* re-ranking, not a hard universe gate: a choppy
    #     market or a flaky yfinance run can drop almost everything, and Markowitz
    #     needs >= 2 names. So we keep the scan ordering but never let it starve the
    #     optimiser — when too few names survive, top up from the full risk pool.
    full_pool = list(tickers)
    tech_map: Dict[str, float] = {}
    tech_results = _cached_scan(risk, tickers)
    if tech_results:
        survivors = [r["ticker"] for r in tech_results]
        tech_map = {r["ticker"]: r["technical_score"] for r in tech_results}
        # Top up when survivors are fewer than what the user requested (or the
        # bare minimum for the optimiser). This prevents Markowitz from being
        # handed only 2 tickers when top_k=15 — the scan preference order is
        # kept; unscored tickers are appended at the back.
        min_needed = max(top_k or 0, _MIN_OPTIMIZER_TICKERS)
        if len(survivors) < min_needed:
            seen = set(survivors)
            survivors += [t for t in full_pool if t not in seen]
            logger.info(
                "Technical filter too thin (%d/%d needed) — topped up to %d tickers",
                len(tech_results), min_needed, len(survivors),
            )
        else:
            logger.info("Technical filter: %d → %d tickers", len(full_pool), len(survivors))
        tickers = survivors if survivors else full_pool

    # 2. Markowitz + top-k. Never hard-fail the build: if optimisation cannot run
    #    (too few usable price series, etc.) fall back to an equal-weight basket so
    #    the user always receives a portfolio.
    fallback_k = top_k or _MAX_TOP_K
    try:
        weights = run_markowitz(tickers, _get_bq(), top_k=top_k)
    except Exception as exc:
        logger.warning("Markowitz failed (%s) — falling back to equal weight", exc)
        weights = _equal_weight_fallback(tickers or full_pool, fallback_k)
    if not weights:
        weights = _equal_weight_fallback(full_pool, fallback_k)

    # Hard count guarantee. markowitz returns min(top_k, usable price series); a thin
    # or partial daily_prices snapshot can price fewer candidates than requested and
    # silently deliver too few. Top up from the rest of the ranked risk pool so the
    # delivered count always equals min(top_k, universe). A genuinely small universe
    # (e.g. Med-High) caps below top_k — the synthesis reports that honestly.
    if top_k is not None and len(weights) < top_k:
        have  = {w["ticker"] for w in weights}
        floor = min((w["weight"] for w in weights), default=1.0)
        for t in dict.fromkeys(tickers + full_pool):
            if len(weights) >= top_k:
                break
            if t not in have:
                weights.append({"ticker": t, "weight": floor})
                have.add(t)
        logger.info("Count top-up → %d positions (requested %d)", len(weights), top_k)

    # The equal-weight fallback respects the same count. Renormalise to sum 1.
    weights = _renormalize(weights)
    logger.info("Portfolio → %d positions (top_k=%s)", len(weights), top_k)

    # 3. Fundamental scores
    opt_tickers = [w["ticker"] for w in weights]
    try:
        fundamentals = get_fundamental_scores(opt_tickers, _get_bq())
    except Exception:
        fundamentals = []

    fund_map  = {f["ticker"]: f for f in fundamentals}
    portfolio = []
    for w in weights:
        f     = fund_map.get(w["ticker"], {})
        score = f.get("mark")
        portfolio.append({
            "ticker":          w["ticker"],
            "weight":          w["weight"],
            "score":           score,
            "technical_score": tech_map.get(w["ticker"]),
            "explanation":     f.get("explanation"),
            "label":           _label(w["weight"], score),
        })

    usage["stocks_delivered"] = len(weights)
    usage["holdings"] = [
        {
            "ticker":            p["ticker"],
            "weight":            p["weight"],
            "fundamental_score": p["score"],
            "technical_score":   p["technical_score"],
            "label":             p["label"],
            "explanation":       p["explanation"],
        }
        for p in portfolio
    ]

    # 4. LLM synthesis
    markowitz_text = "\n".join(f"{w['ticker']}: {w['weight']*100:.2f}%" for w in weights)
    fund_text      = _build_fund_text(fundamentals)
    tech_text = "\n".join(
        f"{p['ticker']}: {p['technical_score']}/100" if p['technical_score'] is not None else f"{p['ticker']}: N/A"
        for p in portfolio
    )
    delivered = len(weights)
    requested_str = str(top_k) if top_k is not None else "auto (optimiser decides)"
    user_ctx = (
        f"User profile — Risk: {risk}, Stocks requested: {requested_str}, Stocks delivered: {delivered}"
        + (f", Amount: {balance:,.0f} {currency}" if balance else "")
        + f"\n\nMarkowitz weights:\n{markowitz_text}"
        + f"\n\nFundamental scores:\n{fund_text}"
        + f"\n\nTechnical scores (momentum/trend):\n{tech_text}"
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
            usage_accumulator=token_usage,
        )
    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        top5 = ", ".join(f"**{w['ticker']}**" for w in weights[:5])
        text = (
            f"Portfolio optimised for **{risk}** risk with {delivered} positions. "
            f"Top holdings: {top5}. Past performance does not guarantee future results."
        )

    if body.session_id:
        session["messages"].append({"role": "user",      "content": body.message})
        session["messages"].append({"role": "assistant", "content": text})
        session["messages"] = session["messages"][-_MAX_HISTORY:]
        session["portfolio_built"] = True
        session["post_build_count"] = 0   # reset limit on each new build

    if router.was_exhausted:
        http_response.headers["X-Overloaded"] = "true"

    usage["response_text"] = text
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
