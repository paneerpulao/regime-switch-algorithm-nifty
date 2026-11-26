"""Performance metrics module for systematic portfolio evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_xirr(cash_flows: List[Tuple[pd.Timestamp, float]], guess: float = 0.15) -> float:
    """Calculates Extended Internal Rate of Return (XIRR) via Newton-Raphson."""
    if not cash_flows or len(cash_flows) < 2:
        return 0.0

    t0 = cash_flows[0][0]
    years = np.array([(cf[0] - t0).days / 365.25 for cf in cash_flows])
    amounts = np.array([cf[1] for cf in cash_flows])

    def npv(rate):
        if rate <= -0.9999:
            return 1e12
        return np.sum(amounts / ((1.0 + rate) ** years))

    def d_npv(rate):
        if rate <= -0.9999:
            return 1e12
        return np.sum(-years * amounts / ((1.0 + rate) ** (years + 1)))

    rate = guess
    for _ in range(100):
        val = npv(rate)
        deriv = d_npv(rate)
        if abs(deriv) < 1e-12:
            break
        new_rate = rate - val / deriv
        if abs(new_rate - rate) < 1e-6:
            return float(new_rate)
        rate = new_rate
        if rate < -0.99 or rate > 10.0:
            break

    # Bisection fallback
    low, high = -0.5, 3.0
    for _ in range(100):
        mid = (low + high) / 2.0
        val = npv(mid)
        if abs(val) < 1e-4:
            return float(mid)
        if val * npv(low) < 0:
            high = mid
        else:
            low = mid

    return float(rate)


def calculate_mdd(equity_series: pd.Series) -> Tuple[float, pd.Series]:
    """Calculates Maximum Drawdown (MDD %) and returns (mdd_pct, drawdown_series)."""
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    mdd = abs(float(drawdown.min())) * 100.0
    return mdd, drawdown


def calculate_sharpe(daily_returns: pd.Series, risk_free_annual: float = 0.065) -> float:
    """Calculates Annualized Sharpe Ratio relative to risk-free rate."""
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - rf_daily
    std = excess_returns.std()
    if std <= 1e-8 or np.isnan(std):
        return 0.0
    return float((excess_returns.mean() / std) * np.sqrt(252))


def compute_metrics_table(
    equity_series: pd.Series,
    cash_flows: List[Tuple[pd.Timestamp, float]],
    total_invested: float,
    trades_count: int,
    turnover: float,
    risk_free_rate: float = 0.065,
    nav_series: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Computes standard institutional performance statistics.

    Money-Weighted metrics (Final Value, Net Profit, Cumulative Return, XIRR) are
    computed from raw portfolio equity and external SIP cash flows.

    Time-Weighted metrics (Sharpe Ratio, Maximum Drawdown, Calmar Ratio) are
    computed strictly from unitized NAV (nav_series) when provided, preventing
    monthly SIP cash deposits from contaminating daily returns or masking drawdowns.
    """
    final_val = float(equity_series.iloc[-1])
    cumulative_return = ((final_val - total_invested) / total_invested) * 100.0

    # Calculate XIRR (Money-Weighted Internal Rate of Return)
    eval_cash_flows = list(cash_flows)
    eval_cash_flows.append((equity_series.index[-1], final_val))
    xirr = calculate_xirr(eval_cash_flows) * 100.0

    # Time-Weighted series for risk evaluation:
    eval_tw_series = nav_series if nav_series is not None else equity_series

    # Drawdown strictly from Time-Weighted unitized NAV
    mdd, _ = calculate_mdd(eval_tw_series)

    # Daily returns and Sharpe strictly from Time-Weighted unitized NAV
    daily_rets = eval_tw_series.pct_change().dropna()
    sharpe = calculate_sharpe(daily_rets, risk_free_annual=risk_free_rate)

    # Annualized Time-Weighted CAGR (True Geometric Compounding of NAV)
    total_days = max((eval_tw_series.index[-1] - eval_tw_series.index[0]).days, 1)
    years = max(total_days / 365.25, 1e-4)
    twr_cagr = (((float(eval_tw_series.iloc[-1]) / float(eval_tw_series.iloc[0])) ** (1.0 / years)) - 1.0) * 100.0

    # Calmar ratio (Time-Weighted CAGR / MDD)
    calmar = (twr_cagr / mdd) if mdd > 0 else 0.0

    return {
        "final_value": final_val,
        "total_invested": total_invested,
        "net_profit": final_val - total_invested,
        "cumulative_return_pct": cumulative_return,
        "xirr_cagr_pct": xirr,
        "twr_cagr_pct": twr_cagr,
        "max_drawdown_pct": mdd,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "trades_count": trades_count,
        "total_turnover": turnover,
    }

