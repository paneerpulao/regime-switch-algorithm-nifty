"""Cross-Asset Risk Parity module (Bear Mode Component).

Calculates inverse realized volatility weights over a rolling lookback window
with volatility cap (50%) and floor (10%).
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class RiskParityModel:
    """Computes Risk Parity asset weights across multi-asset ETF universe."""

    def __init__(
        self,
        prices_df: pd.DataFrame,
        lookback_days: int = 60,
        min_weight: float = 0.10,
        max_weight: float = 0.50,
        cash_ticker: str = "LIQUIDBEES.NS",
        cash_max_weight: float = 0.20
    ):
        self.prices_df = prices_df.copy().ffill()
        self.returns_df = self.prices_df.pct_change()
        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cash_ticker = cash_ticker
        self.cash_max_weight = cash_max_weight

    def calculate_weights_at(self, date: pd.Timestamp) -> Dict[str, float]:
        """Calculates volatility-constrained risk parity weights as of date."""
        # Find index location on or prior to date
        sub_returns = self.returns_df.loc[:date]
        if len(sub_returns) < self.lookback_days:
            # Equal weight fallback if insufficient lookback
            cols = list(self.prices_df.columns)
            return {c: 1.0 / len(cols) for c in cols}

        recent_returns = sub_returns.iloc[-self.lookback_days:]

        # Annualized standard deviation of daily returns
        vols = {}
        for col in self.prices_df.columns:
            s = recent_returns[col].dropna()
            std = float(s.std() * np.sqrt(252)) if len(s) > 10 else 0.15
            # Prevent zero or negative volatility
            vols[col] = max(std, 0.005)

        # Inverse volatility: 1 / sigma_i
        inv_vols = {col: 1.0 / vols[col] for col in vols}
        total_inv_vol = sum(inv_vols.values())

        raw_weights = {col: inv_vols[col] / total_inv_vol for col in inv_vols}

        # Apply Floor (10%) and Cap (50% for equity/gold/bonds, 20% for cash) with iterative re-weighting
        constrained_weights = self._apply_cap_floor(raw_weights)
        return constrained_weights

    def _apply_cap_floor(self, raw_weights: Dict[str, float], max_iter: int = 20) -> Dict[str, float]:
        """Iterative proportional fitting to enforce min_weight and per-asset max_weight bounds while summing to 1.0."""
        keys = list(raw_weights.keys())
        w = raw_weights.copy()
        max_caps = {
            k: (self.cash_max_weight if k == self.cash_ticker else self.max_weight)
            for k in keys
        }

        for _ in range(max_iter):
            clipped = {k: min(max_caps[k], max(self.min_weight, w[k])) for k in keys}
            tot = sum(clipped.values())
            if abs(tot - 1.0) < 1e-5:
                return clipped

            pinned = {k: (clipped[k] == self.min_weight or clipped[k] == max_caps[k]) for k in keys}
            if all(pinned.values()):
                return {k: clipped[k] / tot for k in keys}

            pinned_sum = sum(clipped[k] for k in keys if pinned[k])
            unpinned_keys = [k for k in keys if not pinned[k]]
            unpinned_sum = sum(w[k] for k in unpinned_keys)

            remaining_sum = 1.0 - pinned_sum
            if remaining_sum <= 0 or unpinned_sum <= 0:
                return {k: 1.0 / len(keys) for k in keys}

            scale = remaining_sum / unpinned_sum
            for k in unpinned_keys:
                w[k] = w[k] * scale
            for k in keys:
                if pinned[k]:
                    w[k] = clipped[k]

        final_tot = sum(clipped.values())
        return {k: clipped[k] / final_tot for k in keys}

