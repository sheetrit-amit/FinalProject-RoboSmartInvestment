"""
Seed mock telemetry into BigQuery for dashboard development.

Creates (if needed) and fills both telemetry tables with linked mock data:
  * usage_logs    — structured per-request rows
  * usage_outputs — raw model output text + requested params (linked by event_id)

Every mock event writes one row to each table sharing the same event_id, so the
two tables join cleanly. Safe to run repeatedly (appends new mock events).

Usage:
    cd scripts
    python seed_usage_mock.py            # default: 12 mock events
    python seed_usage_mock.py 50         # custom count
"""

import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

# Make the backend package importable so we reuse the table schemas/names.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backend"))

from google.cloud import bigquery  # noqa: E402
import usage_logger as ul  # noqa: E402

RISKS = ["Low", "Med-Low", "Medium", "Med-High", "High"]
CURRENCIES = ["USD", "EUR", "ILS"]
MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "GOOGL", "META",
           "AMZN", "NFLX", "AMD", "V", "JNJ", "KO", "PG", "XOM"]
LABELS = ["Top Pick", "Growth Opportunity", "Balanced Position", "Statistical Hold"]


def _mock_holdings(top_k):
    chosen = random.sample(TICKERS, k=top_k)
    raw = [random.random() for _ in chosen]
    total = sum(raw)
    holdings = []
    for tkr, r in zip(chosen, raw):
        holdings.append({
            "ticker": tkr,
            "weight": round(r / total, 4),
            "fundamental_score": round(random.uniform(40, 95), 1),
            "technical_score": round(random.uniform(50, 95), 1),
            "label": random.choice(LABELS),
        })
    holdings.sort(key=lambda h: h["weight"], reverse=True)
    return holdings


def _synthesis_text(risk, currency, budget, holdings):
    names = ", ".join(f"**{h['ticker']}**" for h in holdings[:3])
    top = holdings[0]["ticker"]
    return (
        f"Based on your {risk} risk profile and a budget of {budget:,.0f} {currency}, "
        f"this portfolio concentrates risk-adjusted return into {names}. "
        f"**{top}** carries the largest weight at {holdings[0]['weight']*100:.1f}% on "
        f"the strength of its momentum and fundamental score. The remaining positions "
        f"diversify sector exposure while preserving the Sharpe-optimal allocation. "
        f"Past performance does not guarantee future results."
    )


def _advisor_text(risk, top_k):
    bits = []
    if risk:
        bits.append(f"a {risk} risk level")
    if top_k:
        bits.append(f"{top_k} positions")
    detail = " with " + " and ".join(bits) if bits else ""
    return (
        f"Great — I've noted your preferences{detail}. "
        f"Could you confirm your budget so I can finalise the parameters? "
        f"Once everything is set, click \"Build My Portfolio\" below."
    )


def build_event(now):
    eid = str(uuid.uuid4())
    event_time = (now - timedelta(
        hours=random.randint(0, 120), minutes=random.randint(0, 59)
    )).isoformat()
    is_build = random.random() < 0.7  # 70% build-mode events
    risk = random.choice(RISKS)
    currency = random.choice(CURRENCIES)
    budget = float(random.choice([1000, 2500, 5000, 10000, 25000, 50000, 100000]))
    top_k = random.randint(3, 10)
    model = random.choice(MODELS)
    prompt_tokens = random.randint(300, 2500)
    completion_tokens = random.randint(150, 900)

    if is_build:
        holdings = _mock_holdings(top_k)
        delivered = len(holdings)
        text = _synthesis_text(risk, currency, budget, holdings)
        llm_calls = 1
    else:
        holdings = []
        delivered = None
        text = _advisor_text(risk, top_k)
        llm_calls = 2  # extract + advisor reply
        prompt_tokens += random.randint(100, 400)
        completion_tokens += random.randint(40, 120)

    usage_row = {
        "event_id": eid,
        "event_time": event_time,
        "session_id": str(uuid.uuid4()),
        "mode": "build" if is_build else "conversational",
        "status": "ok",
        "error": None,
        "latency_ms": random.randint(800, 9000),
        "model_used": model,
        "overloaded": random.random() < 0.1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_calls": llm_calls,
        "risk": risk,
        "budget": budget,
        "currency": currency,
        "top_k": top_k,
        "stocks_delivered": delivered,
        "holdings": holdings,
    }

    output_row = {
        "event_id": eid,
        "event_time": event_time,
        "response_text": text,
        "requested_risk": risk,
        "requested_budget": budget,
        "requested_currency": currency,
        "requested_top_k": top_k,
    }
    return usage_row, output_row


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    client = bigquery.Client(project=ul.PROJECT_ID, location="EU")

    if not ensure_both(client):
        print("Could not ensure tables — aborting.")
        return 1

    now = datetime.now(timezone.utc)
    usage_rows, output_rows = [], []
    for _ in range(count):
        u, o = build_event(now)
        usage_rows.append(u)
        output_rows.append(o)

    err1 = client.insert_rows_json(ul.USAGE_TABLE, usage_rows)
    err2 = client.insert_rows_json(ul.OUTPUTS_TABLE, output_rows)
    if err1 or err2:
        print("Insert errors:", err1, err2)
        return 1

    print(f"Inserted {count} mock events into both tables (linked by event_id).")
    print(f"  {ul.USAGE_TABLE}")
    print(f"  {ul.OUTPUTS_TABLE}")
    return 0


def ensure_both(client):
    ok1 = ul.ensure_table(client, ul.USAGE_TABLE)
    ok2 = ul.ensure_table(client, ul.OUTPUTS_TABLE)
    return ok1 and ok2


if __name__ == "__main__":
    raise SystemExit(main())
