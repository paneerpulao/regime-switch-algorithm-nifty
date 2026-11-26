"""Regime filter module implementing:
1. Basic Binary 200-day SMA Trend Filter
2. Hysteresis Buffer Band (Sideways Chop Filter) + India VIX Circuit Breaker (Fast Shock Exit)
3. Continuous Allocation Sizing (Sigmoid / Smooth Allocation)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RegimeFilter:
    """Evaluates macroeconomic regime with optional Hysteresis Buffer, India VIX Circuit Breaker, and Sigmoid scaling."""

    def __init__(
        self,
        benchmark_series: pd.Series,
        vix_series: Optional[pd.Series] = None,
        sma_window: int = 200,
        mode: str = "HYSTERESIS_VIX",
        buffer_pct: float = 0.015,
        vix_abs_threshold: Optional[float] = None,
        vix_percentile_window: int = 252,
        vix_percentile_threshold: float = 0.95,
        vix_sma_window: int = 20,
        vix_rel_multiplier: Optional[float] = None,
        sigmoid_k: float = 35.0,
        # Dynamic Spatial Filtering (ATR bands)
        use_dynamic_atr_band: bool = False,
        atr_series: Optional[pd.Series] = None,
        atr_multiplier: float = 1.5,
        # Temporal Filtering (Time-delay confirmation)
        confirmation_days: int = 1,
        # Adaptive Sigmoid Steepness
        use_adaptive_sigmoid_k: bool = False,
        adaptive_k_constant: float = 5.25,
        sigmoid_d0: float = -0.010,
        upper_snap_dist: float = 0.020,
        lower_snap_dist: float = -0.045
    ):
        self.benchmark = benchmark_series.dropna()
        self.sma_window = sma_window
        self.sma = self.benchmark.rolling(window=sma_window).mean()
        self.mode = mode.upper()
        self.buffer_pct = buffer_pct
        self.vix_abs_threshold = vix_abs_threshold
        self.sigmoid_d0 = sigmoid_d0
        self.upper_snap_dist = upper_snap_dist
        self.lower_snap_dist = lower_snap_dist

        # Dynamic Volatility Bands (Spatial Filtering: ATR)
        self.use_dynamic_atr_band = use_dynamic_atr_band
        self.atr_series = atr_series.dropna() if atr_series is not None else None
        self.atr_multiplier = atr_multiplier

        # Time-Delay Confirmation Filters (Temporal Filtering)
        self.confirmation_days = max(1, confirmation_days)
        self._consecutive_bull_bars = 0
        self._consecutive_bear_bars = 0

        # Adaptive Sigmoid Steepness k_t = C / sigma_t
        self.use_adaptive_sigmoid_k = use_adaptive_sigmoid_k
        self.adaptive_k_constant = adaptive_k_constant
        self.benchmark_vol_series = self.benchmark.pct_change().rolling(window=60, min_periods=20).std() * np.sqrt(252)

        # Dynamic Rolling Empirical Percentile VIX parameters (Order Statistics / Non-parametric)
        self.vix = vix_series.dropna() if vix_series is not None else None
        self.vix_percentile_window = vix_percentile_window
        self.vix_percentile_threshold = vix_percentile_threshold

        if self.vix is not None and self.vix_abs_threshold is None:
            # Rolling 252-day window of prior closing values V = {v_{t-252}, ..., v_{t-1}}
            # Shifted by 1 to strictly enforce t-1 lag (zero lookahead)
            self.vix_percentile_series = self.vix.shift(1).rolling(
                window=vix_percentile_window, min_periods=60
            ).quantile(vix_percentile_threshold)
        else:
            self.vix_percentile_series = None

        self.vix_sma = self.vix.rolling(window=vix_sma_window).mean() if (self.vix is not None and vix_sma_window > 0) else None
        self.vix_rel_multiplier = vix_rel_multiplier
        self.sigmoid_k = sigmoid_k

        self.transitions: List[Dict] = []
        self._current_state = "BULL"
        self._prev_state = None

    def evaluate_at(self, date: pd.Timestamp) -> Tuple[str, float, float, Dict]:
        """Evaluates regime at date.

        Returns:
            regime: 'BULL' | 'BEAR' (or primary category)
            weight_momentum: float in [0.0, 1.0]
            weight_risk_parity: float in [0.0, 1.0]
            info: dict containing price, sma, vix, reason
        """
        price = float(self.benchmark.asof(date))
        sma_val = float(self.sma.asof(date))
        vix_val = float(self.vix.asof(date)) if self.vix is not None else 15.0
        vix_pct_val = float(self.vix_percentile_series.asof(date)) if self.vix_percentile_series is not None else (self.vix_abs_threshold if self.vix_abs_threshold is not None else 20.0)
        vix_sma_val = float(self.vix_sma.asof(date)) if self.vix_sma is not None else vix_val

        if np.isnan(sma_val):
            return "BULL", 1.0, 0.0, {"price": price, "sma": price, "vix": vix_val, "vix_threshold": vix_pct_val, "reason": "Warmup"}

        dist_pct = (price - sma_val) / sma_val

        # Dynamic Buffer Band Calculation: Either ATR-anchored or static % buffer
        if self.use_dynamic_atr_band and self.atr_series is not None and date in self.atr_series.index:
            atr_val = float(self.atr_series.asof(date))
            delta_val = self.atr_multiplier * atr_val
            lower_band = sma_val - delta_val
            upper_band = sma_val + delta_val
            effective_buffer_pct = delta_val / sma_val
        else:
            lower_band = sma_val * (1.0 - self.buffer_pct)
            upper_band = sma_val * (1.0 + self.buffer_pct)
            effective_buffer_pct = self.buffer_pct

        # Check VIX Circuit Breaker (Panic Shock Exit)
        vix_shock = False
        vix_reason = ""
        if self.vix is not None:
            if self.vix_abs_threshold is not None:
                # Static Hardcoded VIX mode (Previous approach)
                if vix_val >= self.vix_abs_threshold:
                    vix_shock = True
                    vix_reason = f"Hardcoded VIX Shock (VIX {vix_val:.1f} >= {self.vix_abs_threshold:.1f})"
                elif self.vix_rel_multiplier is not None and vix_sma_val > 0 and vix_val >= self.vix_rel_multiplier * vix_sma_val:
                    vix_shock = True
                    vix_reason = f"VIX Spike Shock (VIX {vix_val:.1f} >= {self.vix_rel_multiplier}x 20-SMA {vix_sma_val:.1f})"
            elif self.vix_percentile_series is not None and not np.isnan(vix_pct_val):
                # Dynamic Rolling 95th-Percentile mode (New institutional approach)
                if vix_val >= vix_pct_val:
                    vix_shock = True
                    vix_reason = f"Dynamic VIX 95th-Pct Shock (VIX {vix_val:.1f} >= {vix_pct_val:.1f})"
                elif self.vix_rel_multiplier is not None and vix_sma_val > 0 and vix_val >= self.vix_rel_multiplier * vix_sma_val:
                    vix_shock = True
                    vix_reason = f"VIX Spike Shock (VIX {vix_val:.1f} >= {self.vix_rel_multiplier}x 20-SMA {vix_sma_val:.1f})"

        # -----------------------------------------------------------------
        # 1. Basic Binary 200 SMA
        # -----------------------------------------------------------------
        if self.mode == "BINARY":
            regime = "BULL" if price >= sma_val else "BEAR"
            w_mom = 1.0 if regime == "BULL" else 0.0
            w_rp = 1.0 - w_mom
            reason = "Price >= 200 SMA" if regime == "BULL" else "Price < 200 SMA"

        # -----------------------------------------------------------------
        # 2. Hysteresis Buffer Band (Dynamic ATR or Static Buffer) + VIX Breakers
        # -----------------------------------------------------------------
        elif self.mode == "HYSTERESIS_VIX":
            if vix_shock:
                regime = "BEAR"
                reason = vix_reason
            else:
                recent_prices = self.benchmark.loc[:date].iloc[-self.confirmation_days:]
                recent_smas = self.sma.loc[:date].iloc[-self.confirmation_days:]
                if self.use_dynamic_atr_band and self.atr_series is not None:
                    recent_atrs = self.atr_series.reindex(recent_prices.index).ffill()
                    recent_upper = recent_smas + self.atr_multiplier * recent_atrs
                    recent_lower = recent_smas - self.atr_multiplier * recent_atrs
                else:
                    recent_upper = recent_smas * (1.0 + self.buffer_pct)
                    recent_lower = recent_smas * (1.0 - self.buffer_pct)

                bull_confirmed = bool((recent_prices > recent_upper).all()) and len(recent_prices) >= self.confirmation_days
                bear_confirmed = bool((recent_prices < recent_lower).all()) and len(recent_prices) >= self.confirmation_days

                if self._current_state == "BULL":
                    if bear_confirmed:
                        regime = "BEAR"
                        reason = f"Lower Band Breakout confirmed ({self.confirmation_days}d daily closes < {lower_band:.1f})"
                    else:
                        regime = "BULL"
                        reason = "Maintained BULL (Above/Within Buffer Band)"
                else:
                    if bull_confirmed:
                        regime = "BULL"
                        reason = f"Upper Band Breakout confirmed ({self.confirmation_days}d daily closes > {upper_band:.1f})"
                    else:
                        regime = "BEAR"
                        reason = "Maintained BEAR (Below/Within Buffer Band)"

            self._current_state = regime
            w_mom = 1.0 if regime == "BULL" else 0.0
            w_rp = 1.0 - w_mom

        # -----------------------------------------------------------------
        # 3. Continuous Allocation Sizing (Piecewise Sigmoid / Smooth Switch)
        # -----------------------------------------------------------------
        elif self.mode == "CONTINUOUS_SIGMOID":
            UPPER_SNAP_DIST = self.upper_snap_dist
            LOWER_SNAP_DIST = self.lower_snap_dist
            SIGMOID_D0 = self.sigmoid_d0

            if vix_shock:
                w_mom = 0.0
                w_rp = 1.0
                regime = "BEAR"
                reason = vix_reason
            else:
                d = dist_pct
                d_shifted = d - SIGMOID_D0

                if self.use_adaptive_sigmoid_k:
                    vol_val = float(self.benchmark_vol_series.asof(date)) if date in self.benchmark_vol_series.index else 0.15
                    vol_val = max(vol_val, 0.05)
                    k_const = 6.50 if d >= SIGMOID_D0 else self.adaptive_k_constant
                    eff_k = float(k_const / vol_val)
                else:
                    eff_k = self.sigmoid_k

                if d >= UPPER_SNAP_DIST:
                    w_mom = 1.0
                    w_rp = 0.0
                    regime = "BULL"
                    reason = f"Upper Snap (d={d*100:+.2f}% >= {UPPER_SNAP_DIST*100:+.1f}%, Full Momentum, k={eff_k:.1f})"
                elif d <= LOWER_SNAP_DIST:
                    w_mom = 0.0
                    w_rp = 1.0
                    regime = "BEAR"
                    reason = f"Lower Snap (d={d*100:+.2f}% <= {LOWER_SNAP_DIST*100:+.1f}%, Full Risk Parity, k={eff_k:.1f})"
                else:
                    w_mom = float(1.0 / (1.0 + np.exp(-eff_k * d_shifted)))
                    w_rp = 1.0 - w_mom
                    regime = "BULL" if w_mom >= 0.5 else "BEAR"
                    reason = f"Asymmetric Sigmoid (k={eff_k:.1f}, d={d*100:+.2f}%, Mom={w_mom*100:.1f}%, RP={w_rp*100:.1f}%)"

        else:
            raise ValueError(f"Unknown regime mode: {self.mode}")

        info = {
            "price": price,
            "sma": sma_val,
            "vix": vix_val,
            "vix_threshold": vix_pct_val,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "buffer_pct": effective_buffer_pct * 100.0,
            "distance_pct": dist_pct * 100.0,
            "vix_shock": vix_shock,
            "reason": reason,
        }
        return regime, w_mom, w_rp, info

    def check_and_log_transition(self, date: pd.Timestamp) -> Tuple[str, float, float, bool, Dict]:
        """Evaluates regime and logs if a state transition occurred."""
        regime, w_mom, w_rp, info = self.evaluate_at(date)
        transition_occurred = False

        if self._prev_state is not None and regime != self._prev_state:
            transition_occurred = True
            log_entry = {
                "date": date,
                "from_regime": self._prev_state,
                "to_regime": regime,
                "benchmark_price": info["price"],
                "sma_200": info["sma"],
                "vix": info["vix"],
                "vix_threshold": info["vix_threshold"],
                "distance_pct": info["distance_pct"],
                "reason": info["reason"]
            }
            self.transitions.append(log_entry)

        self._prev_state = regime
        return regime, w_mom, w_rp, transition_occurred, info

    def get_transition_log(self) -> List[Dict]:
        return self.transitions
