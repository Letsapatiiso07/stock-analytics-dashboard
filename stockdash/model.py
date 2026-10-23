"""Next-day forecasts, walk-forward testing and baselines.

The models predict tomorrow's return from what is known at today's close.
The forecast price is today's close times (1 + predicted return).
"""
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret_1", "ret_5", "close_vs_sma5", "close_vs_sma20", "rsi", "volatility", "volume_ratio"]

MODEL_NAMES = {"ridge": "Ridge regression", "random_forest": "Random forest"}


def make_models() -> Dict[str, object]:
    # The scaler sits inside the pipeline, so each refit learns it from training rows only
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=50, max_features=0.5, n_jobs=2, random_state=0
        ),
    }


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs known at the close of day t, plus `target`, the return from t to t + 1.

    `df` is the output of indicators.add_indicators. The last row has no target.
    """
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change(fill_method=None)
    out["ret_5"] = close.pct_change(5, fill_method=None)
    out["close_vs_sma5"] = close / df["sma_5"] - 1
    out["close_vs_sma20"] = close / df["sma_20"] - 1
    out["rsi"] = df["rsi"]
    out["volatility"] = df["volatility"]
    out["volume_ratio"] = df["volume_ratio"]
    out["close"] = close
    out["next_close"] = close.shift(-1)
    out["target"] = out["next_close"] / close - 1
    return out.dropna(subset=FEATURES)


def walk_forward_splits(n_rows: int, first_test: int, step: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits. Every training row comes before every test row."""
    for start in range(first_test, n_rows, step):
        yield np.arange(start), np.arange(start, min(start + step, n_rows))


def walk_forward(feats: pd.DataFrame, test_start: str, step: int = 21, models=None) -> pd.DataFrame:
    """Refit each model every `step` trading days on all earlier rows and forecast the next block.

    Returns one row per hold-out day: close, actual next close and each model's forecast.
    """
    models = models or make_models()
    data = feats.dropna(subset=["target"])
    first_test = int(np.searchsorted(data.index, pd.Timestamp(test_start)))
    if first_test < 250 or first_test >= len(data):
        raise ValueError(f"need at least 250 training rows before {test_start} and some rows after it")

    X, y = data[FEATURES].to_numpy(), data["target"].to_numpy()
    out = data[["close", "next_close"]].iloc[first_test:].copy()
    for name, model in models.items():
        pred = np.empty(len(data) - first_test)
        for train_idx, test_idx in walk_forward_splits(len(data), first_test, step):
            fitted = clone(model).fit(X[train_idx], y[train_idx])
            pred[test_idx - first_test] = fitted.predict(X[test_idx])
        out[name] = out["close"] * (1 + pred)
    return out


def forecast_next(feats: pd.DataFrame, models=None) -> Dict[str, float]:
    """Fit on every row with a known target and forecast the close after the last row."""
    models = models or make_models()
    train = feats.dropna(subset=["target"])
    last = feats.iloc[[-1]]
    out = {}
    for name, model in models.items():
        fitted = clone(model).fit(train[FEATURES].to_numpy(), train["target"].to_numpy())
        out[name] = float(last["close"].iloc[0] * (1 + fitted.predict(last[FEATURES].to_numpy())[0]))
    return out


def score(preds: pd.DataFrame) -> pd.DataFrame:
    """Compare model forecasts with the naive and always-up baselines on the same days."""
    models = [c for c in preds.columns if c not in ("close", "next_close")]
    close, actual = preds["close"], preds["next_close"]
    went_up = actual > close
    naive_mae = (close - actual).abs().mean()
    rows = [
        {"forecaster": "Naive: tomorrow = today", "mae": naive_mae,
         "mae_pct": ((close - actual).abs() / close).mean() * 100, "mae_vs_naive": 1.0, "direction_acc": np.nan},
        {"forecaster": "Always up", "mae": np.nan, "mae_pct": np.nan, "mae_vs_naive": np.nan,
         "direction_acc": went_up.mean() * 100},
    ]
    for name in models:
        err = (preds[name] - actual).abs()
        rows.append({
            "forecaster": MODEL_NAMES.get(name, name),
            "mae": err.mean(),
            "mae_pct": (err / close).mean() * 100,
            "mae_vs_naive": err.mean() / naive_mae,
            "direction_acc": ((preds[name] > close) == went_up).mean() * 100,
        })
    return pd.DataFrame(rows).set_index("forecaster")


def gains(preds: pd.DataFrame, name: str) -> Dict[str, float]:
    """How much a model beats each baseline, with a rough 95% interval from day-by-day differences.

    mae_gain: percent cut in mean absolute error versus the naive forecast (> 0 is better).
    hit_gain: up/down hit rate minus the always-up hit rate, in points (> 0 is better).
    """
    close, actual = preds["close"], preds["next_close"]
    naive_err = (close - actual).abs()
    d_err = naive_err - (preds[name] - actual).abs()
    went_up = actual > close
    d_hit = ((preds[name] > close) == went_up).astype(float) - went_up.astype(float)
    n = len(preds)
    scale = 100 / naive_err.mean()
    return {
        "mae_gain": d_err.mean() * scale, "mae_ci": 1.96 * d_err.std() / np.sqrt(n) * scale,
        "hit_gain": d_hit.mean() * 100, "hit_ci": 1.96 * d_hit.std() / np.sqrt(n) * 100,
    }
