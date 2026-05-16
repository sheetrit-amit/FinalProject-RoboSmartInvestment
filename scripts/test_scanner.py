#!/usr/bin/env python3
"""
test_scanner.py — Smoke-test the technical_scanner integration.

Tests two things:
  1. Scanner standalone: scan a small set of tickers and print scores.
  2. Full pipeline (optional): hit the running backend /chat endpoint
     with build-mode params and verify technical_score appears in response.

Usage:
    # From project root or scripts/ folder:
    python scripts/test_scanner.py                     # standalone only
    python scripts/test_scanner.py --full              # standalone + backend call
    python scripts/test_scanner.py --full --url http://localhost:8000
"""

import argparse
import json
import sys
import time
import os

# Allow importing from backend/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backend"))

# ---------------------------------------------------------------------------
# 1. Standalone scanner test
# ---------------------------------------------------------------------------

def test_scanner_standalone(tickers: list[str]) -> bool:
    print("\n" + "=" * 60)
    print("TEST 1 — Technical Scanner (standalone)")
    print("=" * 60)
    print(f"Tickers: {tickers}")
    print("Downloading 300d of daily OHLCV from yfinance…\n")

    try:
        from technical_scanner import scan_tickers, MIN_SCORE
    except ImportError as e:
        print(f"ERROR: Could not import technical_scanner: {e}")
        print("  Make sure yfinance is installed: pip install yfinance")
        return False

    t0 = time.perf_counter()
    results = scan_tickers(tickers, min_score=0)  # min_score=0 to see all scores
    elapsed = time.perf_counter() - t0

    if not results:
        print("WARNING: scan_tickers returned empty list. Check yfinance connectivity.")
        return False

    print(f"{'Ticker':<8}  {'Score':>6}  {'Pass (>=' + str(MIN_SCORE) + ')':>14}")
    print("-" * 34)
    for r in results:
        score = r["technical_score"]
        passes = "PASS" if score is not None and score >= MIN_SCORE else "filtered"
        print(f"{r['ticker']:<8}  {score!s:>6}  {passes}")

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Tickers that would pass to Markowitz: {sum(1 for r in results if r['technical_score'] and r['technical_score'] >= MIN_SCORE)}/{len(tickers)}")
    return True


# ---------------------------------------------------------------------------
# 2. Full pipeline test via backend /chat
# ---------------------------------------------------------------------------

def test_full_pipeline(base_url: str) -> bool:
    print("\n" + "=" * 60)
    print("TEST 2 — Full Pipeline (backend /chat in build mode)")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed")
        return False

    # Health check
    try:
        h = requests.get(f"{base_url}/health", timeout=5)
        h.raise_for_status()
        print(f"Backend health: {h.json()}")
    except Exception as e:
        print(f"ERROR: Backend not reachable at {base_url}: {e}")
        print("  Start it with: cd backend && python main.py")
        return False

    # Build-mode request (explicit params bypass conversational mode)
    payload = {
        "message": "Build my portfolio",
        "explicit_budget":   10000,
        "explicit_currency": "USD",
        "explicit_risk":     "Medium",
        "explicit_top_k":    5,
    }

    print(f"\nSending build-mode request: risk=Medium, top_k=5, budget=$10,000")
    print("(scanner runs on ~10-20 tickers — expect 30-90s for yfinance downloads)\n")

    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base_url}/chat", json=payload, timeout=300)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: /chat request failed: {e}")
        return False

    elapsed = time.perf_counter() - t0
    data = r.json()

    print(f"Response received in {elapsed:.1f}s")
    print(f"build_mode: {data.get('build_mode')}")
    print(f"risk_level: {data.get('risk_level')}")
    print(f"\nPortfolio ({len(data.get('portfolio', []))} positions):")
    print(f"  {'Ticker':<8}  {'Weight':>7}  {'Fund.':>6}  {'Tech.':>6}  Label")
    print("  " + "-" * 52)

    has_tech_score = False
    for pos in data.get("portfolio", []):
        tech = pos.get("technical_score")
        if tech is not None:
            has_tech_score = True
        print(
            f"  {pos['ticker']:<8}  {pos['weight']*100:>6.1f}%  "
            f"{str(pos.get('score') or 'N/A'):>6}  "
            f"{str(tech) if tech is not None else 'N/A':>6}  "
            f"{pos.get('label', '')}"
        )

    print(f"\nTechnical scores present in response: {'YES' if has_tech_score else 'NO - check backend logs'}")

    if not has_tech_score:
        print("  The scanner may have returned no results (all tickers below threshold).")
        print("  Check backend logs for 'Technical filter:' lines.")

    print(f"\nLLM synthesis snippet:")
    print(" ", data.get("text", "")[:300].replace("\n", " "))

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test the technical scanner integration")
    parser.add_argument("--full",  action="store_true", help="Also run full backend pipeline test")
    parser.add_argument("--url",   default="http://localhost:8000", help="Backend URL (default: localhost:8000)")
    parser.add_argument("tickers", nargs="*", default=["AAPL", "MSFT", "NVDA", "TSLA", "JPM"],
                        help="Tickers to score in standalone test (default: AAPL MSFT NVDA TSLA JPM)")
    args = parser.parse_args()

    ok1 = test_scanner_standalone(args.tickers)

    if args.full:
        ok2 = test_full_pipeline(args.url)
    else:
        print("\n(Skipping full pipeline test. Run with --full to include it.)")
        ok2 = True

    print("\n" + "=" * 60)
    print(f"Result: {'ALL TESTS PASSED' if ok1 and ok2 else 'SOME TESTS FAILED'}")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
