"""Technical indicators on daily bars.

Every value at row t uses only rows up to and including t, so the indicators
are known at the close of day t and can feed a forecast for day t + 1.
"""
import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average of the last `window` closes."""
    return close.rolling(window).mean()


def wilder_mean(values: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing: seed with a simple mean, then avg = (prev * (n - 1) + x) / n."""
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    start = int(np.argmax(~np.isnan(x))) if (~np.isnan(x)).any() else len(x)
    seed_end = start + window
    if seed_end <= len(x):
        out[seed_end - 1] = x[start:seed_end].mean()
        for i in range(seed_end, len(x)):
            out[i] = (out[i - 1] * (window - 1) + x[i]) / window
    return pd.Series(out, index=values.index)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder, 1978), 0 to 100."""
    delta = close.diff()
    avg_gain = wilder_mean(delta.clip(lower=0), window)
    avg_loss = wilder_mean(-delta.clip(upper=0), window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + avg_gain / avg_loss)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add sma_5, sma_20, rsi, volatility and volume_ratio to a frame of daily bars.

    `df` needs lower-case `close` and `volume` columns, sorted by date (oldest first).
    """
    out = df.copy()
    close = out["close"]
    out["sma_5"] = sma(close, 5)
    out["sma_20"] = sma(close, 20)
    out["rsi"] = rsi(close, 14)
    # Standard deviation of daily returns over 20 days, in percent
    out["volatility"] = close.pct_change(fill_method=None).rolling(20).std() * 100
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    return out
