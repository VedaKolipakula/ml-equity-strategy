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
