"""technical scanner, scores tickers 0-100 on momentum/trend via yfinance"""

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

BENCHMARK       = "^GSPC"
PERIOD          = "6mo"
MIN_SCORE       = 50
LOOKBACK_RS     = 60
LOOKBACK_DIV    = 10


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(close: pd.Series) -> pd.Series:
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal    = _ema(macd_line, 9)
    return macd_line - signal


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)


def _bb_bands(close: pd.Series, n: int = 20, std: float = 2.0):
    mid   = _sma(close, n)
    sigma = close.rolling(n).std()
    return mid - std * sigma, mid + std * sigma


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = clv * volume
    return mfv.rolling(n).sum() / volume.rolling(n).sum()


def _detect_div(price: pd.Series, indicator: pd.Series, lookback: int = LOOKBACK_DIV, tail: int = 7) -> bool:
    if len(price) < lookback:
        return False
    lows  = price == price.rolling(lookback).min()
    div   = (price < price.shift(lookback)) & (indicator > indicator.shift(lookback)) & lows
    return bool(div.tail(tail).any())


def _bull_candle(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    body    = (close - open_).abs()
    lower_w = open_.combine(close, min) - low
    return (lower_w > body * 2) & (low < open_.combine(close, min))


def _score_ticker(df: pd.DataFrame, bench_returns: pd.Series) -> float:
    # 0-100 score for one ticker, nan when data insufficient
    if len(df) < 60:
        return float("nan")

    close, high, low, volume, open_ = (
        df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    )

    ema50     = _ema(close, 50)
    ema20     = _ema(close, 20)
    rsi       = _rsi(close)
    macd_h    = _macd_hist(close)
    cmf       = _cmf(high, low, close, volume)
    tr        = _true_range(high, low, close)
    bb_lo, bb_hi = _bb_bands(close)
    atr20     = _sma(tr, 20)
    avg_vol   = _sma(volume, 10)
    rvol      = volume / avg_vol.replace(0, np.nan)
    pivot_low = low.rolling(5).min()

    stock_ret = close.pct_change(LOOKBACK_RS)
    aligned   = bench_returns.reindex(stock_ret.index).ffill()
    rs_value  = stock_ret - aligned

    last = -1

    score = 0.0
    if close.iloc[last] > ema50.iloc[last]:                                   score += 15
    if ema20.iloc[last]  > ema50.iloc[last]:                                  score += 15
    if _detect_div(low, rsi):                                                 score += 10
    if _detect_div(low, macd_h):                                              score += 10
    if macd_h.iloc[last] > 0 and macd_h.iloc[last] > macd_h.iloc[-2]:        score += 10
    if rs_value.iloc[last] > 0:                                               score += 10
    if cmf.iloc[last] > 0:                                                    score +=  5
    vcp = (bb_lo.iloc[last] > ema20.iloc[last] - atr20.iloc[last] * 1.5) and \
          (bb_hi.iloc[last] < ema20.iloc[last] + atr20.iloc[last] * 1.5)
    if vcp:                                                                   score +=  5
    near_sup = (close.iloc[last] <= pivot_low.iloc[last] * 1.025) and \
               (close.iloc[last] >= pivot_low.iloc[last] * 0.98)
    if near_sup:                                                              score +=  5
    if _bull_candle(open_, high, low, close).iloc[last]:                      score +=  5
    inst_abs = (rvol.iloc[last] > 1.5) and \
               (close.iloc[last] > low.iloc[last] + (high.iloc[last] - low.iloc[last]) * 0.66)
    if inst_abs:                                                              score +=  5
    if rvol.iloc[last] > 1.2:                                                 score +=  5

    return float(score)


def scan_tickers(
    tickers: List[str],
    min_score: int = MIN_SCORE,
    benchmark: str = BENCHMARK,
    period: str = PERIOD,
) -> List[Dict[str, Any]]:
    # download ohlcv, score each, return those >= min_score sorted desc
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping technical scan, returning all tickers unscored")
        return [{"ticker": t, "technical_score": None} for t in tickers]

    try:
        bench_raw = yf.download(benchmark, period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(bench_raw.columns, pd.MultiIndex):
            bench_raw.columns = bench_raw.columns.get_level_values(0)
        bench_returns = bench_raw["Close"].pct_change(LOOKBACK_RS)
    except Exception as exc:
        logger.warning("Could not download benchmark %s: %s — skipping RS signal", benchmark, exc)
        bench_returns = pd.Series(dtype=float)

    results: List[Dict[str, Any]] = []

    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.dropna()

            score = _score_ticker(raw, bench_returns)
            if np.isnan(score):
                continue

            logger.debug("%s  score=%.0f", ticker, score)

            if score >= min_score:
                results.append({"ticker": ticker, "technical_score": round(score, 1)})

        except Exception as exc:
            logger.debug("scan_tickers: skipping %s — %s", ticker, exc)
            continue

    results.sort(key=lambda x: x["technical_score"], reverse=True)
    logger.info(
        "Technical scan: %d/%d tickers passed score >= %d",
        len(results), len(tickers), min_score,
    )
    return results
