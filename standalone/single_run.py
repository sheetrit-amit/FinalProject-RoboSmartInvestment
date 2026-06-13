import json
import logging
import os
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ID  = "pro-visitor-429015-f5"
_BQ_LOCATION = "EU"
_TICKER_GRADES = f"{_PROJECT_ID}.StockData.ticker_grades"
_RISK_RATINGS  = f"{_PROJECT_ID}.StockData.companies_risk_ratings"
_DAILY_PRICES  = f"{_PROJECT_ID}.StockData.daily_prices"

VALID_RISKS      = {"Low", "Med-Low", "Medium", "Med-High", "High"}
MAX_TOP_K        = 15
MIN_OPTIMIZER    = 2
RISK_FREE_RATE   = 0.045
MAX_MARKOWITZ    = 150
MIN_TECH_SCORE   = 50
BENCHMARK        = "^GSPC"
PERIOD           = "6mo"
LOOKBACK_RS      = 60
LOOKBACK_DIV     = 10

EXTRACT_SYSTEM = (
    "You are a financial-intent parser. "
    "Output ONLY a valid JSON object - no markdown, no extra keys, no explanation.\n\n"
    "Extract from the user message:\n"
    '  "balance"  : number or null   (investment amount; midpoint if range)\n'
    '  "currency" : string or null   (e.g. "USD", "EUR", "ILS")\n'
    '  "risk"     : one of "Low"|"Med-Low"|"Medium"|"Med-High"|"High" or null\n'
    '  "top_k"    : integer or null  (number of stocks)\n\n'
    "Set null for anything not explicitly mentioned.\n"
    'JSON only. Example: {"balance": 5000, "currency": "USD", "risk": "Medium", "top_k": null}'
)

SYNTHESIS_SYSTEM = """You are a Senior Investment Strategist for a Robo-Advisor.
Synthesise a concise, user-friendly portfolio recommendation from the data provided.

WRITING RULES:
- 6-10 sentences, paragraph style (no bullet points)
- Bold only ticker symbols with **TICKER**
- Focus on WHY each key position is included
- If delivered fewer positions than requested, explain that the optimiser
  concentrated the allocation into fewer names
- Close with one risk disclaimer sentence
- Do NOT invent data not present in the inputs
"""


def _bq_client() -> bigquery.Client:
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if sa_json:
        try:
            creds = service_account.Credentials.from_service_account_info(json.loads(sa_json))
            return bigquery.Client(credentials=creds, project=_PROJECT_ID, location=_BQ_LOCATION)
        except Exception:
            pass
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        creds = service_account.Credentials.from_service_account_file(sa_path)
        return bigquery.Client(credentials=creds, project=_PROJECT_ID, location=_BQ_LOCATION)
    return bigquery.Client(project=_PROJECT_ID, location=_BQ_LOCATION)


def _get_tickers_by_risk(risk: str, bq: bigquery.Client) -> List[str]:
    query = """
        SELECT tg.ticker
        FROM `pro-visitor-429015-f5.StockData.ticker_grades` AS tg
        JOIN  `pro-visitor-429015-f5.StockData.companies_risk_ratings` AS rr
              ON tg.ticker = rr.ticker
        WHERE tg.mark IS NOT NULL AND rr.risk_level = @risk
        ORDER BY tg.ticker
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("risk", "STRING", risk)]
    )
    return [r.ticker for r in bq.query(query, job_config=cfg).result()]


def _get_fundamental_scores(tickers: List[str], bq: bigquery.Client) -> List[Dict]:
    if not tickers:
        return []
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", tickers)]
    )
    rows = bq.query(
        f"SELECT ticker, mark, explanation FROM `{_TICKER_GRADES}` WHERE ticker IN UNNEST(@tickers)",
        job_config=cfg,
    ).result()
    return [{"ticker": r.ticker, "mark": float(r.mark) if r.mark is not None else None,
             "explanation": str(r.explanation or "")} for r in rows]


def _fetch_prices(tickers: List[str], bq: bigquery.Client) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("tickers", "STRING", tickers)]
    )
    df = bq.query(
        f"SELECT date, ticker, close FROM `{_DAILY_PRICES}` WHERE ticker IN UNNEST(@tickers) ORDER BY date ASC",
        job_config=cfg,
        location=_BQ_LOCATION,
    ).to_dataframe()
    if df.empty:
        raise ValueError("No price data for requested tickers.")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _run_markowitz(tickers: List[str], bq: bigquery.Client, top_k: Optional[int] = None) -> List[Dict]:
    tickers = tickers[:MAX_MARKOWITZ]
    raw = _fetch_prices(tickers, bq)
    prices = raw.pivot_table(index="date", columns="ticker", values="close", aggfunc="mean").ffill().dropna()
    if prices.shape[0] < 60:
        raise ValueError(f"Only {prices.shape[0]} days of price history — need at least 60.")
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").dropna()
    if returns.shape[1] < 2:
        raise ValueError("Fewer than 2 tickers have usable return data.")
    if returns.shape[1] > 80:
        sharpe = returns.mean() / returns.std().replace(0, np.nan)
        returns = returns[sharpe.nlargest(80).index]
    exp_ret = returns.mean() * 252
    cov = returns.cov() * 252
    exp_ret.fillna(0, inplace=True)
    cov.fillna(0, inplace=True)
    n = len(exp_ret)
    mu, sigma = exp_ret.values, cov.values
    max_w = 1.0 / top_k if top_k is not None else 1.0
    result = minimize(
        lambda w: (0.0 if np.sqrt(max(w @ sigma @ w, 1e-10)) < 1e-6
                   else -(w @ mu - RISK_FREE_RATE) / np.sqrt(max(w @ sigma @ w, 1e-10))),
        np.full(n, 1 / n),
        method="SLSQP",
        bounds=[(0.0, max_w)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        tol=1e-8,
        options={"maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")
    weights = pd.Series(result.x, index=exp_ret.index)
    weights = weights[weights > 1e-6].round(6).sort_values(ascending=False)
    return [{"ticker": t, "weight": float(w)} for t, w in weights.items()]


def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rsi(c, n=14):
    d = c.diff(); g = d.clip(lower=0).rolling(n).mean(); l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def _macd_hist(c): return (_ema(c, 12) - _ema(c, 26)) - _ema(_ema(c, 12) - _ema(c, 26), 9)
def _cmf(h, l, c, v, n=20):
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    return (clv * v).rolling(n).sum() / v.rolling(n).sum()
def _detect_div(price, ind, lb=LOOKBACK_DIV, tail=7):
    if len(price) < lb: return False
    lows = price == price.rolling(lb).min()
    return bool(((price < price.shift(lb)) & (ind > ind.shift(lb)) & lows).tail(tail).any())


def _score_ticker(df: pd.DataFrame, bench_returns: pd.Series) -> float:
    if len(df) < 60:
        return float("nan")
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    ema50, ema20 = _ema(c, 50), _ema(c, 20)
    rsi_s, macd_h = _rsi(c), _macd_hist(c)
    cmf_s = _cmf(h, l, c, v)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr20 = _sma(tr, 20)
    mid = _sma(c, 20); sig = c.rolling(20).std()
    bb_lo, bb_hi = mid - 2 * sig, mid + 2 * sig
    rvol = v / _sma(v, 10).replace(0, np.nan)
    rs_value = c.pct_change(LOOKBACK_RS) - bench_returns.reindex(c.index).ffill()
    i = -1
    score = 0.0
    if c.iloc[i] > ema50.iloc[i]:                                                score += 15
    if ema20.iloc[i] > ema50.iloc[i]:                                            score += 15
    if _detect_div(l, rsi_s):                                                    score += 10
    if _detect_div(l, macd_h):                                                   score += 10
    if macd_h.iloc[i] > 0 and macd_h.iloc[i] > macd_h.iloc[-2]:                 score += 10
    if rs_value.iloc[i] > 0:                                                     score += 10
    if cmf_s.iloc[i] > 0:                                                        score +=  5
    if (bb_lo.iloc[i] > ema20.iloc[i] - atr20.iloc[i] * 1.5 and
            bb_hi.iloc[i] < ema20.iloc[i] + atr20.iloc[i] * 1.5):               score +=  5
    piv = l.rolling(5).min()
    if piv.iloc[i] * 0.98 <= c.iloc[i] <= piv.iloc[i] * 1.025:                  score +=  5
    body = (c - o).abs(); lw = o.combine(c, min) - l
    if (lw > body * 2).iloc[i] and (l < o.combine(c, min)).iloc[i]:             score +=  5
    if rvol.iloc[i] > 1.5 and c.iloc[i] > l.iloc[i] + (h.iloc[i] - l.iloc[i]) * 0.66: score += 5
    if rvol.iloc[i] > 1.2:                                                       score +=  5
    return float(score)


def _scan_tickers(tickers: List[str]) -> List[Dict]:
    try:
        import yfinance as yf
    except ImportError:
        return [{"ticker": t, "technical_score": None} for t in tickers]
    try:
        bench_raw = yf.download(BENCHMARK, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
        if isinstance(bench_raw.columns, pd.MultiIndex):
            bench_raw.columns = bench_raw.columns.get_level_values(0)
        bench_returns = bench_raw["Close"].pct_change(LOOKBACK_RS)
    except Exception:
        bench_returns = pd.Series(dtype=float)
    results = []
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna()
            score = _score_ticker(raw, bench_returns)
            if not np.isnan(score) and score >= MIN_TECH_SCORE:
                results.append({"ticker": ticker, "technical_score": round(score, 1)})
        except Exception:
            continue
    results.sort(key=lambda x: x["technical_score"], reverse=True)
    return results


class _ModelRouter:
    FREE_MODELS = [
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "google/gemma-3-27b-it:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-3-12b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3-4b-it:free",
    ]
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._failed: set = set()
        self._idx = 0
        self._lock = threading.Lock()

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/RoboSmartInvestment", "X-Title": "RoboSmartInvestment"}

    def _rotate(self) -> bool:
        self._failed.add(self._idx)
        for i in range(len(self.FREE_MODELS)):
            if i not in self._failed:
                self._idx = i
                return True
        return False

    def chat(self, messages, *, temperature=0.35, max_tokens=1200) -> str:
        last_err = None
        for _ in range(len(self.FREE_MODELS) + 2):
            with self._lock:
                model = self.FREE_MODELS[self._idx]
            try:
                resp = requests.post(self.BASE_URL, headers=self._headers(), json={
                    "model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                }, timeout=90)
                if resp.status_code == 429:
                    with self._lock:
                        if not self._rotate():
                            self._failed.clear()
                            time.sleep(8)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except requests.RequestException as exc:
                last_err = exc
                with self._lock:
                    if not self._rotate():
                        self._failed.clear()
                        time.sleep(3)
        raise RuntimeError(f"All models failed. Last: {last_err}")

    def chat_json(self, messages, *, temperature=0.1, max_tokens=400) -> Dict:
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens).strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
        return json.loads(raw)


def _label(weight, score):
    if score is None:
        return "Quantitative Pick"
    if weight > 0.15 and score >= 70:
        return "Top Pick"
    if weight > 0.15 and score < 50:
        return "Statistical Hold"
    if weight <= 0.15 and score >= 70:
        return "Growth Opportunity"
    return "Balanced Position"


def _renormalize(weights):
    total = sum(w["weight"] for w in weights)
    if total <= 0:
        return weights
    return [{"ticker": w["ticker"], "weight": round(w["weight"] / total, 4)} for w in weights]


def _equal_weight(tickers, top_k):
    picks = tickers[:max(1, top_k)]
    w = 1.0 / len(picks)
    return [{"ticker": t, "weight": w} for t in picks]


class PortfolioEngine:
    def __init__(self):
        self.router = _ModelRouter(api_key=os.getenv("OPEN_ROUTER_API_KEY", ""))
        self.bq = _bq_client()

    def extract_params(self, question: str) -> Dict:
        try:
            intent = self.router.chat_json(
                [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": question}],
                temperature=0.05, max_tokens=120,
            )
        except Exception:
            intent = {}
        risk = intent.get("risk") if intent.get("risk") in VALID_RISKS else "Medium"
        raw_top_k = intent.get("top_k")
        top_k = max(1, min(int(raw_top_k), MAX_TOP_K)) if raw_top_k is not None else None
        return {"balance": intent.get("balance"), "currency": intent.get("currency") or "USD",
                "risk": risk, "top_k": top_k}

    def build(self, params: Dict) -> List[Dict]:
        risk, top_k = params["risk"], params["top_k"]
        tickers = _get_tickers_by_risk(risk, self.bq)
        if not tickers:
            raise RuntimeError(f"No tickers for risk '{risk}'")
        full_pool = list(tickers)
        tech_map = {}
        tech_results = _scan_tickers(tickers)
        if tech_results:
            survivors = [r["ticker"] for r in tech_results]
            tech_map = {r["ticker"]: r["technical_score"] for r in tech_results}
            if len(survivors) < MIN_OPTIMIZER:
                seen = set(survivors)
                survivors += [t for t in full_pool if t not in seen]
            tickers = survivors or full_pool
        fallback_k = top_k or MAX_TOP_K
        try:
            weights = _run_markowitz(tickers, self.bq, top_k=top_k)
        except Exception:
            weights = _equal_weight(tickers or full_pool, fallback_k)
        if not weights:
            weights = _equal_weight(full_pool, fallback_k)
        weights = _renormalize(weights[:top_k] if top_k is not None else weights)
        opt_tickers = [w["ticker"] for w in weights]
        try:
            fundamentals = _get_fundamental_scores(opt_tickers, self.bq)
        except Exception:
            fundamentals = []
        fund_map = {f["ticker"]: f for f in fundamentals}
        portfolio = []
        for w in weights:
            f = fund_map.get(w["ticker"], {})
            score = f.get("mark")
            portfolio.append({"ticker": w["ticker"], "weight": w["weight"], "score": score,
                               "technical_score": tech_map.get(w["ticker"]),
                               "explanation": f.get("explanation"), "label": _label(w["weight"], score)})
        return portfolio

    def narrate(self, question: str, params: Dict, portfolio: List[Dict]) -> str:
        weights_text = "\n".join(f"{p['ticker']}: {p['weight'] * 100:.2f}%" for p in portfolio)
        fund_text = "\n".join(
            f"{p['ticker']} | score: {p['score']} | {str(p['explanation'] or '')[:300]}" for p in portfolio)
        tech_text = "\n".join(
            f"{p['ticker']}: {p['technical_score'] if p['technical_score'] is not None else 'N/A'}"
            for p in portfolio)
        ctx = (f"User question: {question}\nProfile - Risk: {params['risk']}, "
               f"Stocks requested: {params['top_k']}, delivered: {len(portfolio)}"
               + (f", Amount: {params['balance']:,.0f} {params['currency']}" if params["balance"] else "")
               + f"\n\nMarkowitz weights:\n{weights_text}\n\nFundamental scores:\n{fund_text}"
               + f"\n\nTechnical scores:\n{tech_text}")
        try:
            return self.router.chat(
                [{"role": "system", "content": SYNTHESIS_SYSTEM}, {"role": "user", "content": ctx}],
                temperature=0.4, max_tokens=900,
            )
        except Exception:
            top = ", ".join(f"**{p['ticker']}**" for p in portfolio[:5])
            return (f"Portfolio optimised for {params['risk']} risk. Top holdings: {top}. "
                    "Past performance does not guarantee future results.")

    def ask(self, question: str) -> Dict:
        params = self.extract_params(question)
        portfolio = self.build(params)
        text = self.narrate(question, params, portfolio)
        return {"params": params, "portfolio": portfolio, "text": text}


def main():
    if len(sys.argv) < 2:
        print('Usage: python single_run.py "i want to invest 10000 usd, medium risk, 5 stocks"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    result = PortfolioEngine().ask(question)
    p = result["params"]
    print(f"\nDetected: risk={p['risk']}  top_k={p['top_k']}  budget={p['balance']} {p['currency']}\n")
    print(f"{'TICKER':<8}{'WEIGHT':>9}{'FUND':>7}{'TECH':>7}  LABEL")
    print("-" * 50)
    for h in result["portfolio"]:
        fund = h["score"] if h["score"] is not None else "-"
        tech = h["technical_score"] if h["technical_score"] is not None else "-"
        print(f"{h['ticker']:<8}{h['weight'] * 100:>8.2f}%{fund:>7}{tech:>7}  {h['label']}")
    print(f"\n{result['text']}\n")


if __name__ == "__main__":
    main()
