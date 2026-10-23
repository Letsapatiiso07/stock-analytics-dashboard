import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stockdash.indicators import add_indicators
from stockdash.model import make_features, score, walk_forward, walk_forward_splits


def bars(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
    volume = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame({"close": close, "volume": volume}, index=pd.bdate_range("2020-01-01", periods=n))


def small_models():
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0))}


def test_splits_train_before_test_and_cover_each_test_row_once():
    seen = []
    for train, test in walk_forward_splits(100, 60, 7):
        assert train.max() < test.min()
        assert len(train) == test.min()
        seen.extend(test)
    assert seen == list(range(60, 100))


def test_target_is_next_day_return():
    feats = make_features(add_indicators(bars()))
    close = feats["close"]
    assert feats["target"].iloc[10] == pytest.approx(feats["next_close"].iloc[10] / close.iloc[10] - 1)
    assert np.isnan(feats["target"].iloc[-1])


def test_walk_forward_has_no_look_ahead():
    df = bars()
    test_start = df.index[400].strftime("%Y-%m-%d")
    base = walk_forward(make_features(add_indicators(df)), test_start, step=20, models=small_models())

    # Change prices from day 440 on. The first two blocks (forecasts made on days 400 to 439)
    # were fitted on rows whose targets end at day 420 at the latest, and their inputs end
    # on the forecast day, so they must not move. Later blocks must.
    shocked = df.copy()
    shocked.iloc[440:, 0] *= np.linspace(0.5, 2.0, len(df) - 440)
    moved = walk_forward(make_features(add_indicators(shocked)), test_start, step=20, models=small_models())
    pd.testing.assert_series_equal(base["ridge"].iloc[:40], moved["ridge"].iloc[:40])
    assert not np.allclose(base["ridge"].iloc[40:], moved["ridge"].iloc[40:])


def test_score_baselines():
    preds = pd.DataFrame({"close": [10.0, 10, 10, 10], "next_close": [11.0, 9, 11, 12],
                          "ridge": [10.5, 10.5, 9.5, 10.5]})
    s = score(preds)
    assert s.loc["Naive: tomorrow = today", "mae"] == pytest.approx(1.25)
    assert s.loc["Always up", "direction_acc"] == pytest.approx(75.0)
    assert s.loc["Ridge regression", "mae"] == pytest.approx((0.5 + 1.5 + 1.5 + 1.5) / 4)
    assert s.loc["Ridge regression", "direction_acc"] == pytest.approx(50.0)
