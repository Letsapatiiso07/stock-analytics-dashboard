"""Fetch, store and test forecasts for a list of symbols.

    python -m stockdash.pipeline --start 2016-01-01 --end 2026-10-01 --test-start 2023-10-01 --report docs
"""
import argparse
import logging
import time
from typing import Optional

import pandas as pd

from stockdash import indicators, model, store

DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]
DEFAULT_START = "2016-01-01"

log = logging.getLogger("stockdash")


def default_test_start(end: Optional[str]) -> str:
    last = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    return (last - pd.DateOffset(years=3)).strftime("%Y-%m-%d")


def update_symbol(conn, symbol: str, start: str, end: Optional[str], test_start: str) -> pd.DataFrame:
    """Download one symbol, store bars and indicators, run the walk-forward test and store forecasts."""
    bars = indicators.add_indicators(store.download(symbol, start, end))
    feats = model.make_features(bars)
    preds = model.walk_forward(feats, test_start)
    nxt = model.forecast_next(feats)
    # Save only once every step has worked, so the app never sees prices without forecasts
    store.save_prices(conn, symbol, bars, start, end, test_start)
    store.save_predictions(conn, symbol, preds, nxt, feats.index[-1], float(feats["close"].iloc[-1]))
    log.info("%s: %d bars %s to %s, %d hold-out days", symbol, len(bars),
             bars.index[0].date(), bars.index[-1].date(), len(preds))
    return preds


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default=DEFAULT_START, help="first day to download (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="day after the last bar to download; default: today")
    parser.add_argument("--test-start", default=None, help="first hold-out day; default: three years before --end")
    parser.add_argument("--report", default=None, help="folder to write results.md and evaluation.png into")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    test_start = args.test_start or default_test_start(args.end)
    conn = store.connect()
    results = {}
    for symbol in args.symbols:
        preds = update_symbol(conn, symbol, args.start, args.end, test_start)
        results[symbol] = preds
        print(f"\n{symbol}  hold-out {preds.index[0].date()} to {preds.index[-1].date()}, {len(preds)} days")
        print(model.score(preds).round(3).to_string())
        time.sleep(1)  # be polite to Yahoo

    if args.report:
        from stockdash import report
        report.write(results, conn, args.report)


if __name__ == "__main__":
    main()
