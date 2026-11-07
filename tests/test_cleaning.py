"""Tests for src.data cleaning logic.

Strategy: build tiny synthetic price tables where we KNOW the right answer,
then assert the cleaner does exactly that. No network needed.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import _fill_short_gaps, clean_prices

N = 400  # > MIN_HISTORY_DAYS so history length isn't the reason a ticker is dropped
DATES = pd.bdate_range("2020-01-01", periods=N)


def make(**cols):
    """Build prices + volume frames. Each kwarg is a ticker's price series."""
    # REF is always fully populated, like the other ~500 tickers in real data.
    # Without it, a day where the only ticker is NaN makes the whole row empty
    # and the cleaner (correctly) deletes that date, hiding the gap under test.
    prices = pd.DataFrame({"REF": smooth(), **cols}, index=DATES, dtype=float)
    volume = pd.DataFrame(1_000.0, index=DATES, columns=prices.columns)
    return prices, volume


def smooth(n=N, start=100.0):
    """A gently rising price path with no jumps."""
    return pd.Series(start + np.arange(n) * 0.1, index=DATES[:n])


def test_drops_ticker_with_too_much_missing_data():
    bad = smooth().copy()
    bad.iloc[100:200] = np.nan            # 25% missing, in one big hole
    prices, vol = make(GOOD=smooth(), BAD=bad)
    out, _, report = clean_prices(prices, vol)
    assert "BAD" in report.dropped and "BAD" not in out.columns
    assert "GOOD" in out.columns


def test_drops_ticker_with_short_history():
    short = smooth().copy()
    short.iloc[:-100] = np.nan            # only 100 days of data
    prices, vol = make(GOOD=smooth(), SHORT=short)
    out, _, report = clean_prices(prices, vol)
    assert "SHORT" in report.dropped


def test_keeps_late_ipo_with_leading_nans():
    """A stock that listed late is NOT 'missing data' before its IPO."""
    ipo = smooth().copy()
    ipo.iloc[:50] = np.nan                # 350 real days afterwards
    prices, vol = make(GOOD=smooth(), IPO=ipo)
    out, _, report = clean_prices(prices, vol)
    assert "IPO" in out.columns
    assert out["IPO"].iloc[:50].isna().all()   # and we did not back-fill the past


def test_drops_stale_ticker_whose_data_stops():
    """Delisted/stale: data ends early. Trailing NaNs count as missing."""
    stale = smooth().copy()
    stale.iloc[-60:] = np.nan             # last ~15% missing
    prices, vol = make(GOOD=smooth(), STALE=stale)
    _, _, report = clean_prices(prices, vol)
    assert "STALE" in report.dropped


def test_fills_small_gap():
    s = smooth().copy()
    s.iloc[100:103] = np.nan              # 3-day gap <= 5
    prices, vol = make(A=s)
    out, _, report = clean_prices(prices, vol)
    assert out["A"].iloc[100:103].notna().all()
    assert (out["A"].iloc[100:103] == s.iloc[99]).all()   # filled with last known price
    assert report.cells_filled == 3


def test_does_not_fill_large_gap_even_partially():
    """The ffill(limit=) trap: a 10-day gap must stay ENTIRELY NaN."""
    s = smooth().copy()
    s.iloc[100:110] = np.nan              # 10-day gap > 5 (but only 2.5% missing: ticker kept)
    prices, vol = make(A=s)
    out, _, _ = clean_prices(prices, vol)
    assert "A" in out.columns
    assert out["A"].iloc[100:110].isna().all()


def test_fill_short_gaps_helper_directly():
    s = pd.Series([1, np.nan, 3, np.nan, np.nan, np.nan, 7.0])
    out = _fill_short_gaps(s, max_gap=2)
    assert out.tolist()[:3] == [1, 1, 3]            # 1-gap filled
    assert out.iloc[3:6].isna().all()                # 3-gap untouched


def test_never_fills_from_the_future():
    """Leading NaNs must stay NaN (no back-fill = no look-ahead)."""
    s = smooth().copy()
    s.iloc[:3] = np.nan
    prices, vol = make(A=s)
    out, _, _ = clean_prices(prices, vol)
    assert out["A"].iloc[:3].isna().all()


@pytest.mark.parametrize("bad_value", [0.0, -5.0])
def test_non_positive_prices_become_nan(bad_value):
    s = smooth().copy()
    s.iloc[200] = bad_value
    prices, vol = make(A=s)
    _, _, report = clean_prices(prices, vol, max_fill_days=0)
    assert report.non_positive_cells == 1


def test_flags_but_keeps_huge_jump():
    s = smooth().copy()
    s.iloc[300:] *= 3                     # +200% overnight, e.g. an un-adjusted split
    prices, vol = make(A=s)
    out, _, report = clean_prices(prices, vol)
    assert len(report.big_jumps) == 1
    assert out["A"].iloc[300] == s.iloc[300]      # data untouched


def test_volume_is_never_forward_filled():
    s = smooth().copy()
    s.iloc[100:102] = np.nan
    prices, vol = make(A=s)
    vol.iloc[100:102] = np.nan
    out_p, out_v, _ = clean_prices(prices, vol)
    assert out_p["A"].iloc[100:102].notna().all()   # price filled
    assert out_v["A"].iloc[100:102].isna().all()    # volume NOT filled
