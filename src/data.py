"""Data pipeline: S&P 500 tickers -> raw OHLCV -> cleaned wide tables.

Pipeline stages (each is a separate function so each can be tested alone):
    get_sp500_tickers  -> list of tickers
    download_prices    -> raw long-format OHLCV (+ list of failed tickers)
    clean_prices       -> drop bad tickers, fill small gaps, flag bad values
    build_wide_tables  -> {adj close, volume} as dates x tickers DataFrames
    run_pipeline       -> glue: does all of the above and saves to disk

KNOWN LIMITATIONS (also in README):
    * SURVIVORSHIP BIAS: we use *today's* S&P 500 members for the whole
      2010-today history. Firms that failed or were removed are absent, which
      makes backtests look better than reality.
    * yfinance is an unofficial scraper of Yahoo Finance: data can have
      errors/revisions and the API can change without notice.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """Scrape the *current* S&P 500 constituents from Wikipedia.

    Returns tickers in yfinance/Yahoo format. Wikipedia writes share classes
    with a dot (BRK.B, BF.B) but Yahoo uses a dash (BRK-B, BF-B); passing the
    dotted form to yfinance silently fails to find the ticker.

    Survivorship bias: this is today's membership only (see module docstring).

    What can go wrong: Wikipedia changes the table layout (we look up the
    'Symbol' column by name rather than position to be a bit more robust), or
    blocks requests lacking a User-Agent (hence the explicit header).
    """
    # Wikipedia returns HTTP 403 to the default python-requests user agent.
    headers = {"User-Agent": "quant-learning-project/0.1 (student research)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    # The first table on the page is the constituents table.
    tables = pd.read_html(io.StringIO(resp.text))
    symbols = tables[0]["Symbol"].astype(str).str.strip()

    tickers = sorted({s.replace(".", "-") for s in symbols if s})
    logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    return tickers


# --------------------------------------------------------------------------
# Step 2: download + save raw data
# --------------------------------------------------------------------------

import time
from datetime import date
from pathlib import Path

import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_DATE = "2010-01-01"
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _download_batch(tickers: list[str], start: str, end: str,
                    max_retries: int = 4, base_delay: float = 2.0) -> pd.DataFrame | None:
    """Download one batch, retrying with exponential backoff on exceptions.

    Backoff = wait 2s, 4s, 8s, ... between attempts. Rate limiting is usually
    temporary; hammering the server immediately again just extends the ban.
    Returns None if every attempt raised.
    """
    for attempt in range(max_retries):
        try:
            # auto_adjust=False keeps BOTH 'Close' (split-adjusted only) and
            # 'Adj Close' (split + dividend adjusted), so raw data stays
            # transparent. group_by='ticker' puts the ticker on column level 0.
            return yf.download(
                tickers, start=start, end=end, auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as exc:  # yfinance raises various exception types
            delay = base_delay * 2**attempt
            logger.warning("Batch of %d failed (%s); retry %d/%d in %.0fs",
                           len(tickers), exc, attempt + 1, max_retries, delay)
            time.sleep(delay)
    return None


def _split_batch(df: pd.DataFrame | None, tickers: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
    """Turn a wide multi-ticker download into per-ticker long frames.

    A ticker counts as FAILED if it is absent or entirely NaN (yfinance
    returns all-NaN columns for delisted/unknown symbols rather than raising).
    """
    good: list[pd.DataFrame] = []
    failed: list[str] = []
    if df is None or df.empty:
        return good, list(tickers)
    top_level = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set()
    for t in tickers:
        if t not in top_level:
            failed.append(t)
            continue
        sub = df[t].dropna(how="all")
        if sub.empty:
            failed.append(t)
            continue
        sub = sub.reindex(columns=RAW_COLUMNS)
        sub.insert(0, "Ticker", t)
        sub.index.name = "Date"
        good.append(sub)
    return good, failed


def download_prices(tickers: list[str], start: str = START_DATE, end: str | None = None,
                    batch_size: int = 50, pause: float = 1.0) -> tuple[pd.DataFrame, list[str]]:
    """Download daily OHLCV for all tickers; never crash on bad tickers.

    Returns (long_df, failed_tickers). long_df has a Date index and columns
    Ticker, Open, High, Low, Close, Adj Close, Volume.

    `end` defaults to today. yfinance treats `end` as exclusive, so today's
    bar is not included -- good, because an intraday bar is incomplete and
    using it would be a subtle form of look-ahead/leakage.
    """
    end = end or date.today().isoformat()
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info("Downloading batch %d-%d of %d", i + 1, i + len(batch), len(tickers))
        good, bad = _split_batch(_download_batch(batch, start, end), batch)
        frames += good
        failed += bad
        time.sleep(pause)  # be polite between batches

    # Second chance: a ticker can fail inside a batch due to a transient hiccup.
    if failed:
        logger.info("Retrying %d failed tickers individually", len(failed))
        still_failed: list[str] = []
        for t in failed:
            good, bad = _split_batch(_download_batch([t], start, end, max_retries=2), [t])
            frames += good
            still_failed += bad
            time.sleep(pause)
        failed = still_failed

    if failed:
        logger.warning("FAILED tickers (%d): %s", len(failed), sorted(failed))
    long_df = pd.concat(frames) if frames else pd.DataFrame(columns=["Ticker"] + RAW_COLUMNS)
    return long_df, sorted(failed)


def save_raw(long_df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save raw long-format data as parquet (compressed, typed, fast to load).

    Parquet beats CSV here: it stores dtypes (no re-parsing dates), is
    columnar and compressed, and loads ~10x faster for this size of data.
    """
    path = path or RAW_DIR / "prices_raw.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(path)
    logger.info("Saved raw data: %s (%d rows)", path, len(long_df))
    return path


# --------------------------------------------------------------------------
# Step 3: cleaning
# --------------------------------------------------------------------------

from dataclasses import dataclass, field

import numpy as np

# Thresholds (all overridable). Reasoning:
MAX_MISSING_FRAC = 0.05   # >5% holes after a stock's first date => unreliable
MIN_HISTORY_DAYS = 252    # ~1 trading year; needed for 12-month momentum features
MAX_FILL_DAYS = 5         # ~1 trading week; longer gaps are real outages/halts
MAX_ABS_DAILY_RETURN = 0.5  # |1-day move| > 50% is suspicious (flag only)


@dataclass
class CleaningReport:
    """What the cleaner did -- so nothing is silently changed."""
    dropped: dict[str, str] = field(default_factory=dict)  # ticker -> reason
    non_positive_cells: int = 0     # price cells set to NaN because <= 0
    cells_filled: int = 0           # price cells filled by short-gap ffill
    big_jumps: pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged, NOT removed


def _fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Forward-fill NaN runs of length <= max_gap; leave longer runs entirely NaN.

    Why not s.ffill(limit=max_gap)? Because that fills the FIRST max_gap days
    of a long gap and leaves the rest NaN: a half-fabricated stretch of prices.
    Here we measure each NaN run's full length and fill it only if short.
    """
    na = s.isna()
    run_id = (na != na.shift()).cumsum()            # new id each time NaN-ness flips
    run_len = na.groupby(run_id).transform("sum")   # for NaN runs: the run's length
    fillable = na & (run_len <= max_gap)
    return s.where(~fillable, s.ffill())


def clean_prices(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    max_missing_frac: float = MAX_MISSING_FRAC,
    min_history_days: int = MIN_HISTORY_DAYS,
    max_fill_days: int = MAX_FILL_DAYS,
    max_abs_return: float = MAX_ABS_DAILY_RETURN,
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Clean wide (dates x tickers) adjusted-close and volume tables.

    Order matters:
      1. tidy index, 2. invalid values -> NaN, 3. drop bad tickers (judged on
      the data BEFORE filling, so we measure real data quality, not our own
      patches), 4. fill small gaps, 5. flag suspicious jumps.

    Look-ahead note: forward-fill only uses PAST values (yesterday's price
    stands in for today's), so it does not leak the future. Backward-fill
    would, and we never use it.
    """
    report = CleaningReport()

    # 1. Tidy: sorted unique dates, no all-empty rows (e.g. a bad calendar day).
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    volume = volume[~volume.index.duplicated(keep="last")].sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    volume = volume.reindex(index=prices.index, columns=prices.columns)

    # 2. Impossible values -> NaN. A price <= 0 is always a data error.
    bad = prices <= 0
    report.non_positive_cells = int(bad.sum().sum())
    prices = prices.mask(bad)
    volume = volume.mask(volume < 0)

    # 3. Drop tickers with too little / too patchy data.
    for ticker in prices.columns:
        col = prices[ticker]
        first = col.first_valid_index()
        if first is None:
            report.dropped[ticker] = "no valid prices"
            continue
        window = col.loc[first:]  # from first trade to END of calendar (stale tails count)
        if col.notna().sum() < min_history_days:
            report.dropped[ticker] = f"short history ({int(col.notna().sum())} days)"
        elif window.isna().mean() > max_missing_frac:
            report.dropped[ticker] = f"{window.isna().mean():.1%} missing since first date"
    keep = [t for t in prices.columns if t not in report.dropped]
    prices, volume = prices[keep], volume[keep]
    for t, why in report.dropped.items():
        logger.warning("Dropped %s: %s", t, why)

    # 4. Fill short gaps in PRICES only. Volume is never filled: yesterday's
    #    volume is not today's, so filling would invent trading activity.
    before = prices.isna().sum().sum()
    prices = prices.apply(_fill_short_gaps, max_gap=max_fill_days)
    report.cells_filled = int(before - prices.isna().sum().sum())

    # 5. Flag (don't delete) huge one-day moves. Real ones exist (earnings
    #    shocks), but so do un-adjusted splits and bad ticks; a human should look.
    ret = prices.pct_change(fill_method=None)
    jumps = ret.stack()
    jumps = jumps[jumps.abs() > max_abs_return]
    report.big_jumps = jumps.rename("return").reset_index().rename(
        columns={"level_0": "date", "level_1": "ticker", "Date": "date", "Ticker": "ticker"})
    if len(report.big_jumps):
        logger.warning("%d one-day moves exceed %.0f%% (flagged, kept)",
                       len(report.big_jumps), max_abs_return * 100)

    return prices, volume, report
