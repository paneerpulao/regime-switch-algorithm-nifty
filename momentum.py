"""Momentum Component module (Bull Mode Component).

Implements 12-1 month cross-sectional momentum ranking (skipping recent 1-month reversal)
across liquid large/mid caps, or alternatively an ETF proxy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class MomentumModel:
    """Computes cross-sectional 12-1 month risk-adjusted momentum leaders with rank hysteresis."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_long: int = 252,   # 12-month lookback
        skip_short: int = 21,       # 1-month reversal skip
        top_n: int = 5,
        keep_rank: int = 8,         # Rank hysteresis buffer to prevent boundary churn
        vol_window: int = 126,      # 6-month realized volatility window for risk-adjusted momentum
        mode: str = "STOCK_BASKET",
        etf_ticker: str = "NIFTYBEES.NS",
        pit_universe_df: Optional[pd.DataFrame] = None,
        use_vol_norm: bool = False,
        require_stock_trend: bool = False
    ):
        self.prices_df = prices_df.copy().ffill()
        self.returns_df = self.prices_df.pct_change()
        self.lookback_long = lookback_long
        self.skip_short = skip_short
        self.top_n = top_n
        self.keep_rank = keep_rank
        self.vol_window = vol_window
        self.mode = mode
        self.etf_ticker = etf_ticker
        self.pit_universe_df = pit_universe_df
        self.use_vol_norm = use_vol_norm
        self.require_stock_trend = require_stock_trend

    def calculate_weights_at(
        self,
        date: pd.Timestamp,
        current_holdings: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """Calculates momentum weights as of date with rank hysteresis.

        Candidates are scored using risk-adjusted momentum:
            Score_i = (P_{i, t-21} / P_{i, t-252} - 1.0) / sigma_{i, 126d}
        Any currently held stock that remains within top keep_rank (Rank <= 8)
        retains its position to prevent costly boundary churn.
        """
        if self.mode == "ETF_PROXY":
            return {self.etf_ticker: 1.0}, [(self.etf_ticker, 0.0)]

        # Determine candidate universe for date
        if self.pit_universe_df is not None and date in self.pit_universe_df.index:
            row = self.pit_universe_df.loc[date]
            candidate_universe = [ticker for ticker in row.index if row[ticker] == 1 and ticker in self.prices_df.columns]
        else:
            candidate_universe = list(self.prices_df.columns)

        # Get historical slice up to date
        sub_df = self.prices_df.loc[:date]
        sub_rets = self.returns_df.loc[:date]
        if len(sub_df) < self.lookback_long:
            cols = candidate_universe[:self.top_n]
            return {c: 1.0 / len(cols) for c in cols}, [(c, 0.0) for c in cols]

        # Price at t - 21 (1 month ago) and Price at t - 252 (12 months ago)
        p_t_minus_skip = sub_df.iloc[-self.skip_short]
        p_t_minus_long = sub_df.iloc[-self.lookback_long]
        recent_rets = sub_rets.iloc[-self.vol_window:]

        mom_scores = {}
        fallback_candidates = {}
        raw_mom_returns = {}
        for col in candidate_universe:
            if col not in sub_df.columns:
                continue
            col_series = sub_df[col].dropna()
            if len(col_series) < self.lookback_long:
                continue

            p_past = float(p_t_minus_long[col])
            p_skip = float(p_t_minus_skip[col])
            p_curr = float(sub_df[col].iloc[-1])
            if p_past > 0 and not np.isnan(p_skip) and not np.isnan(p_past) and not np.isnan(p_curr):
                raw_ret = (p_skip / p_past) - 1.0
                if self.use_vol_norm:
                    vol = float(recent_rets[col].dropna().std() * np.sqrt(252)) if col in recent_rets.columns else 0.20
                    vol = max(vol, 0.12)  # Floor annualized volatility at 12%
                    score = raw_ret / vol
                else:
                    score = raw_ret

                raw_mom_returns[col] = raw_ret

                # 1-Month Crash Guard (-12% reversal circuit breaker):
                # Prevents buying or holding a stock that crashed >12% during the 21-day skip window
                # (e.g. rapid single-stock breakdown like ADANIENT in Feb 2023).
                ret_last_21d = (p_curr / p_skip) - 1.0 if p_skip > 0 else 0.0

                if ret_last_21d < -0.12:
                    fallback_candidates[col] = score
                else:
                    mom_scores[col] = score

        if len(mom_scores) < self.top_n and fallback_candidates:
            needed = self.top_n - len(mom_scores)
            sorted_fallback = sorted(fallback_candidates.items(), key=lambda x: x[1], reverse=True)
            for ticker, score in sorted_fallback[:needed]:
                mom_scores[ticker] = score

        if not mom_scores:
            cols = candidate_universe[:self.top_n]
            return {c: 1.0 / len(cols) for c in cols}, [(c, 0.0) for c in cols]

        # Sort candidates descending by risk-adjusted momentum score
        sorted_candidates = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
        top_keep_set = {ticker for ticker, _ in sorted_candidates[:self.keep_rank]}

        selected = []
        if current_holdings:
            # Retain existing holdings that are still within top keep_rank (Rank <= 8)
            for ticker, _ in sorted_candidates:
                if ticker in current_holdings and ticker in top_keep_set and len(selected) < self.top_n:
                    selected.append((ticker, raw_mom_returns[ticker]))

        # Fill remaining slots from highest ranked candidates
        selected_tickers = {t for t, _ in selected}
        for ticker, _ in sorted_candidates:
            if len(selected) >= self.top_n:
                break
            if ticker not in selected_tickers:
                selected.append((ticker, raw_mom_returns[ticker]))
                selected_tickers.add(ticker)

        # Equal weight among selected top momentum leaders (20% each for N=5)
        w_per_stock = 1.0 / len(selected) if selected else 0.0
        weights = {ticker: w_per_stock for ticker, _ in selected}

        return weights, selected

