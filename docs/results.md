Daily bars from Yahoo Finance (yfinance, split and dividend adjusted), 2016-01-04 to 2026-09-30.
Hold-out: forecasts made at the close from 2023-10-02 to 2026-09-29, models refit every 21 trading days on all earlier days.

| Symbol | Hold-out days | Naive MAE (% of price) | Ridge MAE / naive | Forest MAE / naive | Always-up hit rate (%) | Ridge hit rate (%) | Forest hit rate (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AAPL | 751 | 1.15 | 1.006 | 1.003 | 54.06 | 51.00 | 51.66 |
| GOOGL | 751 | 1.39 | 1.007 | 1.002 | 54.19 | 52.46 | 52.20 |
| MSFT | 751 | 1.13 | 1.015 | 1.018 | 53.40 | 53.53 | 52.33 |
| TSLA | 751 | 2.65 | 1.004 | 1.001 | 50.73 | 48.87 | 51.53 |
| AMZN | 751 | 1.45 | 1.000 | 1.002 | 52.06 | 52.46 | 50.73 |
| NVDA | 751 | 2.17 | 0.996 | 0.994 | 54.33 | 51.93 | 53.13 |
| Mean of 6 | 751 | 1.65 | 1.005 | 1.003 | 53.13 | 51.71 | 51.93 |
