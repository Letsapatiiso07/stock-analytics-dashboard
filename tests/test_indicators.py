import numpy as np
import pandas as pd
import pytest

from stockdash.indicators import add_indicators, rsi, sma, wilder_mean


def bars(n=80, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    volume = rng.integers(1_000, 5_000, n).astype(float)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)


def test_sma_by_hand():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0, 5.0]
    assert sma(s, 3).iloc[:2].isna().all()


def test_wilder_mean_by_hand():
    x = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0])
    out = wilder_mean(x, 3)
    # seed is the mean of the first three values, then (prev * 2 + x) / 3
    assert out.iloc[3] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx((2.0 * 2 + 4.0) / 3)
    assert out.iloc[:3].isna().all()


def test_rsi_by_hand():
    # Changes: +1, -1, +2, +1 with a 2-day window
    close = pd.Series([10.0, 11, 10, 12, 13])
    out = rsi(close, 2)
    # Day 2: avg gain (1 + 0) / 2 = 0.5, avg loss (0 + 1) / 2 = 0.5 -> 50
    assert out.iloc[2] == pytest.approx(50.0)
    # Day 3: gain (0.5 + 2) / 2 = 1.25, loss (0.5 + 0) / 2 = 0.25 -> 100 - 100 / 6
    assert out.iloc[3] == pytest.approx(100 - 100 / 6)
    # Day 4: gain (1.25 + 1) / 2 = 1.125, loss 0.125 -> 90
    assert out.iloc[4] == pytest.approx(90.0)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 40.0))
    assert (rsi(up).dropna() == 100).all()
    assert (rsi(up[::-1].reset_index(drop=True)).dropna() == 0).all()


def test_indicators_use_only_past_rows():
    df = bars()
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], cut)

    # Changing the future must not change any earlier indicator value
    shocked = df.copy()
    shocked.iloc[50:, :] *= 3
    pd.testing.assert_frame_equal(add_indicators(shocked).iloc[:50], full.iloc[:50])


def test_volatility_and_volume_ratio():
    df = bars()
    out = add_indicators(df)
    expected_vol = df["close"].pct_change().iloc[-20:].std() * 100
    assert out["volatility"].iloc[-1] == pytest.approx(expected_vol)
    assert out["volume_ratio"].iloc[-1] == pytest.approx(df["volume"].iloc[-1] / df["volume"].iloc[-20:].mean())
