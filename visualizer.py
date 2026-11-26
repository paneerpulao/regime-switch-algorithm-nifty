"""Visualization module comparing Anti-Whipsaw filters:
- Basic Binary 200 SMA
- Hysteresis + India VIX Circuit Breaker
- Continuous Sigmoid Sizing
- Baseline NIFTY 50 Buy & Hold
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
import shutil
from typing import Dict
try:
    from .performance_metrics import calculate_mdd
except (ImportError, ValueError):
    from regime_switching.performance_metrics import calculate_mdd


def plot_anti_whipsaw_comparison(
    results: Dict,
    output_path: str = "regime_switching/anti_whipsaw_comparison.png",
    artifact_path: str = None
):
    """Renders 3-subplot comparison highlighting Static vs PIT universes and crisis periods."""
    pit_s = results["dyn_vix_pit_series"]
    sta_s = results["dyn_vix_static_series"]
    sig_s = results["sigmoid_series"]
    nifty_s = results["nifty_series"]
    reg_hys = results["regime_hys_series"]

    # Drawdowns
    _, pit_dd = calculate_mdd(pit_s)
    _, sta_dd = calculate_mdd(sta_s)
    _, sig_dd = calculate_mdd(sig_s)
    _, nifty_dd = calculate_mdd(nifty_s)

    pit_dd_pct = pit_dd * 100.0
    sta_dd_pct = sta_dd * 100.0
    sig_dd_pct = sig_dd * 100.0
    nifty_dd_pct = nifty_dd * 100.0

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.8, 0.8]}
    )

    # Color Palette
    c_pit = "#00C853"     # Vivid Forest/Emerald Green (Point-in-Time True Reality)
    c_sta = "#2979FF"     # Royal Blue (Static Universe Baseline)
    c_sig = "#AA00FF"     # Purple (Adaptive Sigmoid)
    c_nifty = "#757575"   # Neutral Charcoal Grey

    # -------------------------------------------------------------
    # Subplot 1: Equity Curves
    # -------------------------------------------------------------
    ax1.plot(sta_s.index, sta_s / 1e5, label="Model 1: Static Universe (Lookahead Bias)", color=c_sta, linewidth=2.0, linestyle="--", alpha=0.9)
    ax1.plot(pit_s.index, pit_s / 1e5, label="Model 2: Point-in-Time Universe (Survivorship-Bias Free)", color=c_pit, linewidth=2.5)
    ax1.plot(sig_s.index, sig_s / 1e5, label="Adaptive Continuous Sigmoid (k_t = C/σ_t)", color=c_sig, linewidth=1.7, linestyle="-.", alpha=0.85)
    ax1.plot(nifty_s.index, nifty_s / 1e5, label="Benchmark: NIFTY 50 Index (Buy & Hold SIP)", color=c_nifty, linewidth=1.8, alpha=0.8)

    # Shaded Crisis Periods
    p0_start, p0_end = pd.Timestamp("2018-09-01"), pd.Timestamp("2019-10-31")
    p1_start, p1_end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")
    p2_start, p2_end = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-03-31")

    ax1.axvspan(p0_start, p0_end, color="#9C27B0", alpha=0.10, label="Period 0: IL&FS NBFC Contagion & Distressed Exits (2018-19)")
    ax1.axvspan(p1_start, p1_end, color="#FF5252", alpha=0.12, label="Period 1: COVID-19 V-Crash & Rebound (2020)")
    ax1.axvspan(p2_start, p2_end, color="#FFD740", alpha=0.14, label="Period 2: Ukraine War & Rate-Hike Consolidation (2022-23)")

    # Final Value Labels
    final_sta = sta_s.iloc[-1] / 1e5
    final_pit = pit_s.iloc[-1] / 1e5
    final_sig = sig_s.iloc[-1] / 1e5
    final_n = nifty_s.iloc[-1] / 1e5

    ax1.annotate(f"Static Univ: ₹{final_sta:.2f}L", xy=(sta_s.index[-1], final_sta), xytext=(8, 4),
                 textcoords="offset points", fontweight="bold", color=c_sta, fontsize=9.5)
    ax1.annotate(f"PIT Univ: ₹{final_pit:.2f}L", xy=(pit_s.index[-1], final_pit), xytext=(8, -8),
                 textcoords="offset points", fontweight="bold", color=c_pit, fontsize=9.5)
    ax1.annotate(f"NIFTY 50: ₹{final_n:.2f}L", xy=(nifty_s.index[-1], final_n), xytext=(8, -20),
                 textcoords="offset points", fontweight="bold", color=c_nifty, fontsize=9)

    ax1.set_title("Survivorship-Bias Elimination: Static Universe vs. Point-in-Time (PIT) Dynamic Reconstitution (2018 - 2023)\nQuantifying the Survivorship Premium and Testing Shock Exit under Real Historical Inclusions/Exclusions",
                  fontsize=13.0, fontweight="bold", pad=12)
    ax1.set_ylabel("Portfolio Value (₹ Lakhs)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.6)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 2: Underwater Drawdown Chart (%)
    # -------------------------------------------------------------
    ax2.plot(sta_dd_pct.index, sta_dd_pct, label="Model 1: Static Universe", color=c_sta, linewidth=1.5, linestyle="--", alpha=0.85)
    ax2.plot(pit_dd_pct.index, pit_dd_pct, label="Model 2: PIT Universe", color=c_pit, linewidth=1.8)
    ax2.plot(nifty_dd_pct.index, nifty_dd_pct, label="NIFTY 50", color=c_nifty, linewidth=1.4, alpha=0.8)

    ax2.axvspan(p0_start, p0_end, color="#9C27B0", alpha=0.10)
    ax2.axvspan(p1_start, p1_end, color="#FF5252", alpha=0.12)
    ax2.axvspan(p2_start, p2_end, color="#FFD740", alpha=0.14)
    ax2.fill_between(pit_dd_pct.index, pit_dd_pct, 0, color=c_pit, alpha=0.12)

    ax2.set_ylabel("Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Underwater Drawdown (% from Peak) - Stress Test During Real Distressed Periods", fontsize=11, fontweight="bold", pad=5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5, ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Subplot 3: Enhanced Regime Ribbon
    # -------------------------------------------------------------
    ax3.fill_between(reg_hys.index, 0, 1, where=(reg_hys == "BULL"), color="#00C853", alpha=0.75, label="BULL Mode (100% Momentum Basket)")
    ax3.fill_between(reg_hys.index, 0, 1, where=(reg_hys == "BEAR"), color="#FF6D00", alpha=0.85, label="BEAR Mode (100% Risk Parity Basket)")

    ax3.axvspan(p0_start, p0_end, color="#9C27B0", alpha=0.10)
    ax3.axvspan(p1_start, p1_end, color="#FF5252", alpha=0.12)
    ax3.axvspan(p2_start, p2_end, color="#FFD740", alpha=0.14)

    ax3.set_ylim([0, 1])
    ax3.set_yticks([])
    ax3.set_ylabel("Regime", fontsize=10, fontweight="bold")
    ax3.set_title("Active Macro Regime Ribbon (Spatial ATR Bands + 3-Day Temporal Confirmation + Dynamic 95th-Pct VIX)", fontsize=10, fontweight="bold", pad=4)
    ax3.legend(loc="center right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.set_xlabel("Year", fontsize=11, fontweight="bold")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[Visualizer] Plot saved to: {output_path}")

    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        shutil.copyfile(output_path, artifact_path)
        print(f"[Visualizer] Copied to artifact: {artifact_path}")

    plt.close(fig)
