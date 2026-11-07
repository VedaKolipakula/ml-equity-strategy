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
