"""Fetch daily bars from Yahoo Finance and keep them, with forecasts, in SQLite."""
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

# yfinance still uses pandas' inplace= arguments, which pandas 3.1 warns about on every call
warnings.filterwarnings("ignore", message="The inplace keyword", module=r"yfinance\.")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "stock_analytics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stock_prices (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    sma_5 REAL, sma_20 REAL, rsi REAL, volatility REAL, volume_ratio REAL,
    PRIMARY KEY (symbol, date)
);
-- One row per hold-out day and model. The row for the latest day has no actual yet.
CREATE TABLE IF NOT EXISTS predictions (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    model TEXT NOT NULL,
    close REAL,
    predicted_close REAL,
    actual_close REAL,
    PRIMARY KEY (symbol, date, model)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    symbol TEXT PRIMARY KEY,
    fetched_at TEXT,
    start_date TEXT,
    end_date TEXT,
    test_start TEXT
);
"""

PRICE_COLUMNS = ["open", "high", "low", "close", "volume",
                 "sma_5", "sma_20", "rsi", "volatility", "volume_ratio"]


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def download(symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Daily bars, adjusted for splits and dividends. `end` is exclusive."""
    raw = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no rows for {symbol}")
    # During trading hours Yahoo returns a bar for today that is still moving. Drop it until
    # 18:00 exchange time so every stored row is a finished day.
    now = pd.Timestamp.now(tz=raw.index.tz)
    if raw.index[-1].normalize() == now.normalize() and now.hour < 18:
        raw = raw.iloc[:-1]
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize(), name="date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def save_prices(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame,
                start: str, end: Optional[str], test_start: str) -> None:
    """Replace the stored history for `symbol` with `df` (bars plus indicators)."""
    rows = df[PRICE_COLUMNS].copy()
    rows.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    rows.insert(0, "symbol", symbol)
    with conn:
        conn.execute("DELETE FROM stock_prices WHERE symbol = ?", (symbol,))
        rows.to_sql("stock_prices", conn, if_exists="append", index=False)
        conn.execute(
            "INSERT OR REPLACE INTO fetch_log VALUES (?, ?, ?, ?, ?)",
            (symbol, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), start, end, test_start),
        )


def save_predictions(conn: sqlite3.Connection, symbol: str, preds: pd.DataFrame,
                     next_forecast: Dict[str, float], last_date: pd.Timestamp, last_close: float) -> None:
    """Store hold-out forecasts and the forecast for the day after `last_date`."""
    models = [c for c in preds.columns if c not in ("close", "next_close")]
    records = []
    for name in models:
        for date, row in preds.iterrows():
            records.append((symbol, date.strftime("%Y-%m-%d"), name, row["close"], row[name], row["next_close"]))
        records.append((symbol, last_date.strftime("%Y-%m-%d"), name, last_close, next_forecast[name], None))
    with conn:
        conn.execute("DELETE FROM predictions WHERE symbol = ?", (symbol,))
        conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)", records)


def load_prices(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT * FROM stock_prices WHERE symbol = ? ORDER BY date", conn, params=(symbol,), parse_dates=["date"]
    )
    return df.drop(columns="symbol").set_index("date")


def load_predictions(conn: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    """Wide frame: one row per day with close, next_close and one column per model."""
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE symbol = ? ORDER BY date", conn, params=(symbol,), parse_dates=["date"]
    )
    if df.empty:
        return df
    wide = df.pivot(index="date", columns="model", values="predicted_close")
    base = df.groupby("date")[["close", "actual_close"]].first().rename(columns={"actual_close": "next_close"})
    return base.join(wide)


def symbols(conn: sqlite3.Connection) -> list:
    return [r[0] for r in conn.execute("SELECT symbol FROM fetch_log ORDER BY symbol")]


def fetch_info(conn: sqlite3.Connection, symbol: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM fetch_log WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        return None
    return dict(zip(["symbol", "fetched_at", "start_date", "end_date", "test_start"], row))
