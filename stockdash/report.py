"""Write the hold-out results as a Markdown table and a PNG chart."""
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stockdash import model, store

SERIES = {"ridge": "#2a78d6", "random_forest": "#eb6834"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def summary_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, preds in results.items():
        s = model.score(preds)
        rows.append({
            "Symbol": symbol,
            "Hold-out days": len(preds),
            "Naive MAE (% of price)": s.loc["Naive: tomorrow = today", "mae_pct"],
            "Ridge MAE / naive": s.loc["Ridge regression", "mae_vs_naive"],
            "Forest MAE / naive": s.loc["Random forest", "mae_vs_naive"],
            "Always-up hit rate (%)": s.loc["Always up", "direction_acc"],
            "Ridge hit rate (%)": s.loc["Ridge regression", "direction_acc"],
            "Forest hit rate (%)": s.loc["Random forest", "direction_acc"],
        })
    table = pd.DataFrame(rows).set_index("Symbol")
    table.loc[f"Mean of {len(table)}"] = table.mean()
    return table


def to_markdown(table: pd.DataFrame) -> str:
    fmt = {c: "{:.0f}" if c == "Hold-out days" else "{:.3f}" if "/ naive" in c else "{:.2f}" for c in table}
    head = "| Symbol | " + " | ".join(table.columns) + " |"
    sep = "|---|" + "---:|" * len(table.columns)
    body = ["| " + str(i) + " | " + " | ".join(fmt[c].format(r[c]) for c in table.columns) + " |"
            for i, r in table.iterrows()]
    return "\n".join([head, sep] + body)


def plot(results: Dict[str, pd.DataFrame], path: Path, period: str) -> None:
    symbols = list(results)
    g = {s: {m: model.gains(results[s], m) for m in SERIES} for s in symbols}
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    y = np.arange(len(symbols))[::-1]
    h = 0.32
    panels = [("mae_gain", "mae_ci", "Price error: % less than naive (tomorrow = today)", "{:+.0f}%"),
              ("hit_gain", "hit_ci", "Up/down calls: hit rate minus always-up", "{:+.0f} pts")]
    for ax, (key, ci, title, tick) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for k, (name, color) in enumerate(SERIES.items()):
            vals = [g[s][name][key] for s in symbols]
            errs = [g[s][name][ci] for s in symbols]
            offset = (0.5 - k) * (h + 0.04)
            ax.barh(y + offset, vals, height=h, color=color, label=model.MODEL_NAMES[name], zorder=3)
            ax.errorbar(vals, y + offset, xerr=errs, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2, zorder=4)
        ax.axvline(0, color=INK_2, linewidth=1, zorder=2)
        ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("left of 0: worse than the baseline    right of 0: better")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        lim = max(abs(v) for s in symbols for m in SERIES for v in
                  (g[s][m][key] - g[s][m][ci], g[s][m][key] + g[s][m][ci])) * 1.1
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _, t=tick: t.format(v).replace("+0", "0")))
    axes[0].set_yticks(y, symbols)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"Next-day forecasts vs simple baselines, walk-forward hold-out {period}",
                 x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Whiskers: rough 95% interval from day-by-day differences. "
             "Data: Yahoo Finance daily closes via yfinance.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write(results: Dict[str, pd.DataFrame], conn, folder: str) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    first = min(p.index[0] for p in results.values()).date()
    last = max(p.index[-1] for p in results.values()).date()
    starts = [store.load_prices(conn, s).index[0].date() for s in results]
    ends = [store.load_prices(conn, s).index[-1].date() for s in results]
    table = summary_table(results)
    text = (
        f"Daily bars from Yahoo Finance (yfinance, split and dividend adjusted), "
        f"{min(starts)} to {max(ends)}.\n"
        f"Hold-out: forecasts made at the close from {first} to {last}, "
        f"models refit every 21 trading days on all earlier days.\n\n"
        + to_markdown(table) + "\n"
    )
    (out / "results.md").write_text(text, encoding="utf-8")
    plot(results, out / "evaluation.png", f"{first:%b %Y} to {last:%b %Y}")
    print("\n" + text)
    print(f"wrote {out / 'results.md'} and {out / 'evaluation.png'}")
