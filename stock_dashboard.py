"""Streamlit dashboard: daily prices, indicators and a check of next-day forecasts against baselines.

    streamlit run stock_dashboard.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockdash import model, store
from stockdash.pipeline import DEFAULT_START, default_test_start, update_symbol

COLORS = {"ridge": "#2a78d6", "random_forest": "#eb6834"}
SMA_5, SMA_20, LINE, REF = "#6250d6", "#52514e", "#2a78d6", "#898781"

st.set_page_config(page_title="Stock analytics dashboard", layout="wide")
conn = store.connect()


def refresh(symbol: str) -> bool:
    """Download the latest bars for `symbol` and re-run the test. Show any error in the sidebar."""
    info = store.fetch_info(conn, symbol) or {}
    try:
        with st.spinner(f"Downloading {symbol} and re-running the walk-forward test..."):
            update_symbol(conn, symbol, info.get("start_date") or DEFAULT_START, None,
                          info.get("test_start") or default_test_start(None))
        return True
    except Exception as exc:  # bad ticker, network error or too little history
        st.sidebar.error(f"{symbol}: {exc}")
        return False


def verdict(gain: float, ci: float, better: str, worse: str, unit: str) -> str:
    text = f"{abs(gain):.1f}{unit} {better if gain > 0 else worse} (± {ci:.1f}{unit}"
    if abs(gain) <= ci:
        return text + ", within noise)"
    return text + (", **beats the baseline**)" if gain > 0 else ", **worse than the baseline**)")


# Sidebar
st.sidebar.title("Stock analytics")
symbols = store.symbols(conn)
new_symbol = st.sidebar.text_input("Add a symbol (Yahoo Finance ticker)").strip().upper()
if st.sidebar.button("Download and test", disabled=not new_symbol) and refresh(new_symbol):
    st.query_params["symbol"] = new_symbol
    st.rerun()

if not symbols:
    st.title("Stock analytics dashboard")
    st.info("No data yet. Run `python -m stockdash.pipeline` or add a symbol in the sidebar.")
    st.stop()

wanted = st.query_params.get("symbol", symbols[0])
symbol = st.sidebar.selectbox("Symbol", symbols, index=symbols.index(wanted) if wanted in symbols else 0)
st.query_params["symbol"] = symbol
days = st.sidebar.slider("Days to chart", 60, 750, 180, step=30)
if st.sidebar.button(f"Fetch latest {symbol} bars and re-test") and refresh(symbol):
    st.rerun()

prices = store.load_prices(conn, symbol)
preds = store.load_predictions(conn, symbol)
info = store.fetch_info(conn, symbol)
models = [m for m in model.MODEL_NAMES if m in preds.columns]
preds = preds[["close", "next_close"] + models]
hold = preds.dropna(subset=["next_close"])
live = preds[preds["next_close"].isna()]
last_day = prices.index[-1]

st.title(f"{symbol}: daily prices and a tested next-day forecast")
st.caption(
    f"Daily bars from Yahoo Finance via yfinance, adjusted for splits and dividends. "
    f"Last bar: **{last_day:%a %d %b %Y}**, fetched {info['fetched_at']}. "
    "End-of-day data, not real-time. Not investment advice."
)

# Key numbers for the last bar
last, prev = prices.iloc[-1], prices.iloc[-2]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Close", f"${last['close']:,.2f}", f"{(last['close'] / prev['close'] - 1) * 100:+.2f}% on the day")
c2.metric("Volume", f"{last['volume'] / 1e6:,.1f}M", f"{last['volume_ratio']:.2f}x the 20-day average",
          delta_color="off")
c3.metric("RSI (14 days)", f"{last['rsi']:.1f}")
c4.metric("Volatility (20 days)", f"{last['volatility']:.2f}% a day")

# Price and indicators
view = prices.iloc[-days:]
left, right = st.columns([3, 2])
with left:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view.index, open=view["open"], high=view["high"], low=view["low"],
                                 close=view["close"], name="Daily bar"))
    fig.add_trace(go.Scatter(x=view.index, y=view["sma_5"], name="SMA 5", line=dict(color=SMA_5, width=2)))
    fig.add_trace(go.Scatter(x=view.index, y=view["sma_20"], name="SMA 20", line=dict(color=SMA_20, width=2)))
    fig.update_layout(title="Price with 5- and 20-day moving averages", height=430, xaxis_rangeslider_visible=False,
                      yaxis_title="USD", margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.02, x=1,
                                                                             xanchor="right", yanchor="bottom"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, width="stretch")
with right:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["RSI (14 days)", "Volume / 20-day average", "Volatility, % a day"])
    for row, col, ref in [(1, "rsi", [30, 70]), (2, "volume_ratio", [1]), (3, "volatility", [])]:
        fig.add_trace(go.Scatter(x=view.index, y=view[col], line=dict(width=2, color=LINE),
                                 showlegend=False), row=row, col=1)
        for y in ref:
            fig.add_hline(y=y, line=dict(color=REF, width=1, dash="dot"), row=row, col=1)
    fig.update_layout(height=430, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

# Forecast check
st.header("Do the forecasts beat simple baselines?")
st.markdown(
    "Each model predicts tomorrow's return from what is known at today's close (returns, distance from the "
    "moving averages, RSI, volatility, volume ratio). It is refit every 21 trading days on all earlier days "
    f"and tested on the days that follow (walk-forward). Hold-out: **{hold.index[0]:%d %b %Y} to "
    f"{hold.index[-1]:%d %b %Y}, {len(hold)} days**. Baselines: *naive* says tomorrow's close equals "
    "today's; *always up* calls every day up."
)

scores = model.score(hold)
lines = []
for m in models:
    g = model.gains(hold, m)
    lines.append(
        f"- **{model.MODEL_NAMES[m]}**: price error "
        + verdict(g["mae_gain"], g["mae_ci"], "lower than naive", "higher than naive", "%")
        + "; up/down hit rate "
        + verdict(g["hit_gain"], g["hit_ci"], "above always-up", "below always-up", " points")
    )
st.markdown("\n".join(lines))
st.caption("± is a rough 95% interval from day-by-day differences. Within noise means this hold-out "
           "cannot tell the model from the baseline.")

left, right = st.columns(2)
with left:
    table = pd.DataFrame({
        "MAE, USD": scores["mae"].map("{:.2f}".format),
        "MAE, % of price": scores["mae_pct"].map("{:.2f}".format),
        "MAE / naive": scores["mae_vs_naive"].map("{:.3f}".format),
        "Up/down hit rate": scores["direction_acc"].map("{:.1f}%".format),
    }).replace({"nan": "n/a", "nan%": "n/a"})
    st.table(table)
with right:
    fig = go.Figure()
    naive_err = (hold["close"] - hold["next_close"]).abs()
    for m in models:
        saved = (naive_err - (hold[m] - hold["next_close"]).abs()).cumsum()
        fig.add_trace(go.Scatter(x=hold.index, y=saved, name=model.MODEL_NAMES[m],
                                 line=dict(color=COLORS[m], width=2)))
    fig.add_hline(y=0, line=dict(color=REF, width=1))
    fig.update_layout(title="Running total of error saved versus naive",
                      yaxis_title="USD per share (below 0: worse)", height=380, margin=dict(t=50, b=10),
                      legend=dict(orientation="h", y=-0.12, x=0, yanchor="top"))
    st.plotly_chart(fig, width="stretch")

# The forecast itself, with its track record next to it
st.subheader(f"Forecast for the next trading day after {last_day:%d %b %Y}")
cols = st.columns(len(models) + 1)
cols[0].metric("Naive (last close)", f"${last['close']:,.2f}")
for col, m in zip(cols[1:], models):
    value = live[m].iloc[0]
    col.metric(model.MODEL_NAMES[m], f"${value:,.2f}", f"{(value / last['close'] - 1) * 100:+.2f}%",
               delta_color="off")
st.caption(
    f"On the hold-out the naive forecast missed by ${scores.loc['Naive: tomorrow = today', 'mae']:.2f} "
    "on an average day. Read the model numbers with that in mind and with the test results above."
)
