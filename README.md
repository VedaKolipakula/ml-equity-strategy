# Cross-Sectional ML Equity Strategy

## Project goal
Learn how ML trading signals are built, backtested, and deployed by building a
cross-sectional momentum / mean-reversion strategy on S&P 500 stocks: each day,
rank stocks by predicted future return, go long the top, short the bottom.

## Success criteria (placeholders - fill in)
- [ ] Pipeline reproduces from raw data with one command
- [ ] Out-of-sample backtest Sharpe > ____ after transaction costs
- [ ] No look-ahead bias (verified by tests)
- [ ] Paper-trades on Alpaca for ____ weeks without errors

## Quick start
```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.11+ required
pip install -r requirements.txt
python -m src.data          # download + clean (~30 s) -> data/raw, data/processed
pytest                      # cleaning-logic tests
jupyter lab notebooks/01_data_check.ipynb
```

## Status
- Phase 0: setup - done
- Phase 1: data - done (500 tickers x ~4,200 days, 2010-01-04 to present)
- Phase 2+: features, models, backtest, live - not started

## Cleaning rules (src/data.py)
| Rule | Value | Why |
|---|---|---|
| Drop ticker if >5% of days missing after its first date | 5% | beyond ~200 days of holes the series is unreliable |
| Drop ticker with <252 valid days | 1 year | 12-month momentum features need it |
| Forward-fill gaps of at most 5 days (whole gap or nothing) | 1 week | longer gaps are real outages/halts |
| Prices <= 0 set to NaN | - | impossible values |
| One-day moves >50% flagged, NOT removed | 50% | real events exist; a human decides |
| Volume never forward-filled | - | would invent trading activity |

## Known limitations
- **Survivorship bias.** We use *today's* S&P 500 members for all of 2010-today.
  Companies that went bankrupt, were acquired, or were dropped from the index are
  missing, and today's members are by definition those that did well enough to
  stay. Backtests will look better than reality, especially for momentum.
  Fix later: point-in-time index membership (paid data, e.g. CRSP/Norgate).
- **Recent additions have short histories** (e.g. Q, SNDK, GEV): NaN before listing.
- **Spin-off artifacts.** Yahoo's adjustment for spin-offs is imperfect: e.g. DHR
  jumps $41.94 -> $67.61 on 2016-07-05 (Fortive spin-off). Flagged, not fixed.
- **Odd volume history** for stocks that recently moved their primary listing to
  the US (SW, FERG): near-zero volume early on. Apply a liquidity filter later.
- **yfinance is an unofficial scraper**: data can be revised or wrong, and the API
  can change or rate-limit without notice. Fine for learning, not for production.
- **Unadjusted for taxes/fees**; adjusted close assumes dividends are reinvested.
- Today's bar is excluded (incomplete intraday data).
