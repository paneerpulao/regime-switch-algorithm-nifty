"""Data Loader module for fetching and cleaning NSE tickers and benchmark data."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
try:
    from .config import CONFIG
except (ImportError, ValueError):
    from regime_switching.config import CONFIG


def clean_outlier_glitches(df: pd.DataFrame) -> pd.DataFrame:
    """Detects and fixes erroneous bad-tick data spikes/drops via strictly causal backward rolling median."""
    cleaned = df.copy()
    for col in cleaned.columns:
        series = cleaned[col].dropna()
        if len(series) < 15:
            continue
        med = series.rolling(11, center=False, min_periods=3).median()
        ratio = series / med
        # Flag extreme outliers (drop > 50% or jump > 100% relative to local backward median)
        bad_mask = (ratio < 0.5) | (ratio > 2.0)
        if bad_mask.any():
            cleaned.loc[bad_mask.index[bad_mask], col] = med[bad_mask]
    return cleaned.ffill().bfill()


def fetch_all_market_data(
    config: Dict = CONFIG,
    end_date: Optional[str] = None
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fetches and aligns historical market data for:
    1. Benchmark (Nifty 50)
    2. Risk Parity ETF basket (NIFTYBEES, GOLDBEES, LTGILTBEES, LIQUIDBEES)
    3. Liquid Large-Cap Momentum universe.
    4. India VIX
    5. Point-in-Time Universe matrix
    """
    warmup_start = config["DATA_WARMUP_START"]
    end_date = end_date or config["END_DATE"]
    benchmark_ticker = config["BENCHMARK_TICKER"]

    print(f"[DataLoader] Fetching benchmark data for {benchmark_ticker} ({warmup_start} to {end_date})...")
    bench_raw = yf.download(benchmark_ticker, start=warmup_start, end=end_date, progress=False)["Close"]
    if isinstance(bench_raw, pd.DataFrame):
        bench_raw = bench_raw.iloc[:, 0]
    bench_series = bench_raw.dropna()

    # Risk Parity ETFs
    etf_tickers = list(config["RISK_PARITY_TICKERS"].values())
    print(f"[DataLoader] Fetching Risk Parity ETF basket: {etf_tickers}...")
    etf_raw = yf.download(etf_tickers, start=warmup_start, end=end_date, progress=False)["Close"]
    if isinstance(etf_raw, pd.Series):
        etf_raw = etf_raw.to_frame()

    # Clean outlier glitches (like Yahoo Finance NIFTYBEES 2019 misplaced decimal)
    etf_raw = clean_outlier_glitches(etf_raw)

    # Align with benchmark trading days
    etf_df = etf_raw.reindex(bench_series.index).ffill()

    # Handle LTGILTBEES backward fill if listed after start
    if "LTGILTBEES.NS" in etf_df.columns:
        gilt_series = etf_df["LTGILTBEES.NS"]
        first_valid = gilt_series.first_valid_index()
        if first_valid is not None and first_valid > bench_series.index[0]:
            # Backfill with 7.0% annual G-sec yield compounding backward
            first_val = gilt_series.loc[first_valid]
            pre_dates = etf_df.index[etf_df.index < first_valid]
            daily_factor = (1.0 + 0.07) ** (1.0 / 252.0)
            back_values = []
            for i, _ in enumerate(reversed(pre_dates)):
                back_values.append(first_val / (daily_factor ** (i + 1)))
            back_values.reverse()
            etf_df.loc[pre_dates, "LTGILTBEES.NS"] = back_values

    # Handle LIQUIDBEES.NS flat ₹1000 price peg on Yahoo Finance by compounding at 5.5% annual overnight repo yield
    if "LIQUIDBEES.NS" in etf_df.columns:
        liq_series = etf_df["LIQUIDBEES.NS"].dropna()
        if len(liq_series) > 10:
            total_raw_ret = abs((liq_series.iloc[-1] / liq_series.iloc[0]) - 1.0)
            if total_raw_ret < 0.03:  # Pegged at ~1000.0 on Yahoo Finance Close
                base_val = float(liq_series.iloc[0])
                daily_liq_factor = (1.0 + 0.055) ** (1.0 / 252.0)
                compounded_vals = [base_val * (daily_liq_factor ** i) for i in range(len(etf_df.index))]
                etf_df["LIQUIDBEES.NS"] = pd.Series(compounded_vals, index=etf_df.index)

    etf_df = etf_df.bfill().ffill()

    # Build Dynamic Point-in-Time (PIT) Universe Matrix
    base_2017 = config.get("NIFTY50_BASE_2017", config["LIQUID_UNIVERSE"])
    changes = config.get("NIFTY50_CHANGES", [])
    pit_universe_df, pit_all_tickers = build_point_in_time_universe_matrix(bench_series.index, base_2017, changes)

    # Momentum Stock Universe (downloads union of static liquid universe and all historical PIT constituents)
    all_stock_tickers = sorted(list(set(config["LIQUID_UNIVERSE"] + pit_all_tickers)))
    print(f"[DataLoader] Fetching Momentum universe ({len(all_stock_tickers)} combined stocks covering PIT and static universes)...")
    stocks_raw = yf.download(all_stock_tickers, start=warmup_start, end=end_date, progress=False)["Close"]
    if isinstance(stocks_raw, pd.Series):
        stocks_raw = stocks_raw.to_frame()

    stocks_raw = clean_outlier_glitches(stocks_raw)
    stocks_df = stocks_raw.reindex(bench_series.index).ffill().bfill()

    # India VIX (Volatility Circuit Breaker)
    vix_ticker = config.get("VIX_TICKER", "^INDIAVIX")
    print(f"[DataLoader] Fetching India VIX for volatility filter: {vix_ticker}...")
    vix_raw = yf.download(vix_ticker, start=warmup_start, end=end_date, progress=False)["Close"]
    if isinstance(vix_raw, pd.DataFrame):
        vix_raw = vix_raw.iloc[:, 0]
    vix_series = vix_raw.reindex(bench_series.index).ffill().bfill()

    print(f"[DataLoader] Data ready. Aligned trading bars: {len(bench_series)} from {bench_series.index[0].date()} to {bench_series.index[-1].date()}")
    return bench_series, etf_df, stocks_df, vix_series, pit_universe_df


def build_point_in_time_universe_matrix(
    trading_dates: pd.DatetimeIndex,
    base_constituents: list,
    changes_schedule: list
) -> Tuple[pd.DataFrame, list]:
    """Generates a daily binary DataFrame pit_universe_df indexed by trading date.

    Value is 1 if ticker was an active constituent on that date, 0 otherwise.
    Forward-propagates additions and removals based on effective rebalance dates.
    """
    all_tickers = sorted(list(set(
        base_constituents +
        [t for _, adds, _ in changes_schedule for t in adds] +
        [t for _, _, rems in changes_schedule for t in rems]
    )))

    pit_df = pd.DataFrame(0, index=trading_dates, columns=all_tickers, dtype=int)
    curr_constituents = set(base_constituents)
    changes_sorted = sorted(changes_schedule, key=lambda x: pd.Timestamp(x[0]))

    change_idx = 0
    for dt in trading_dates:
        while change_idx < len(changes_sorted) and dt >= pd.Timestamp(changes_sorted[change_idx][0]):
            eff_dt, adds, rems = changes_sorted[change_idx]
            for a in adds:
                curr_constituents.add(a)
            for r in rems:
                curr_constituents.discard(r)
            change_idx += 1
        for c in curr_constituents:
            if c in pit_df.columns:
                pit_df.loc[dt, c] = 1

    return pit_df, all_tickers

