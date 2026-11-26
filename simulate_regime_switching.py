"""Main execution script for Anti-Whipsaw Enhanced Regime-Switching Strategy (2018 - 2023).

Compares:
1. Basic Binary 200-day SMA
2. Enhanced Strategy with Hysteresis (1.5%) + India VIX Circuit Breakers
3. Enhanced Strategy with Continuous Sigmoid Sizing
4. Baseline NIFTY 50 Buy & Hold

Evaluates full period and specifically isolates:
- Period 1: Jan 2020 – Dec 2020 (COVID-19 V-crash & recovery)
- Period 2: Jan 2022 – Mar 2023 (Russia-Ukraine war & rate-hike consolidation)
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root and parent directories are on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_algo_root = os.path.abspath(os.path.join(_current_dir, ".."))
_code_root = os.path.abspath(os.path.join(_current_dir, "../.."))
for p in [_algo_root, _code_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from regime_switching.config import CONFIG
    from regime_switching.data_loader import fetch_all_market_data
    from regime_switching.backtest_engine import BacktestEngine
    from regime_switching.performance_metrics import compute_metrics_table, calculate_mdd
    from regime_switching.visualizer import plot_anti_whipsaw_comparison
except ImportError:
    from algo_trading_system.regime_switching.config import CONFIG
    from algo_trading_system.regime_switching.data_loader import fetch_all_market_data
    from algo_trading_system.regime_switching.backtest_engine import BacktestEngine
    from algo_trading_system.regime_switching.performance_metrics import compute_metrics_table, calculate_mdd
    from algo_trading_system.regime_switching.visualizer import plot_anti_whipsaw_comparison


def print_table(headers, rows):
    """Clean custom monospace table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)
    for row in rows:
        print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))


def evaluate_crisis_subperiod(equity_dict: dict, nav_dict: dict, start_date: str, end_date: str, period_name: str):
    """Evaluates sub-period equity changes and pure time-weighted NAV returns/drawdowns."""
    s_dt = pd.Timestamp(start_date)
    e_dt = pd.Timestamp(end_date)

    rows = []
    for name, s in equity_dict.items():
        sub_eq = s.loc[s_dt:e_dt]
        sub_nav = nav_dict.get(name, pd.Series()).loc[s_dt:e_dt] if name in nav_dict else pd.Series()
        if len(sub_eq) == 0:
            continue
        p_start = sub_eq.iloc[0]
        p_end = sub_eq.iloc[-1]

        if len(sub_nav) > 0 and sub_nav.iloc[0] > 0:
            nav_start = sub_nav.iloc[0]
            nav_end = sub_nav.iloc[-1]
            ret_pct = ((nav_end / nav_start) - 1.0) * 100.0
            mdd_pct, _ = calculate_mdd(sub_nav)
        else:
            ret_pct = ((p_end - p_start) / p_start) * 100.0 if p_start > 0 else 0.0
            mdd_pct, _ = calculate_mdd(sub_eq)

        rows.append([
            name,
            f"INR {p_start:,.2f}",
            f"INR {p_end:,.2f}",
            f"{ret_pct:+.2f}%",
            f"{mdd_pct:.2f}%"
        ])

    print("\n" + "=" * 95)
    print(f" CRISIS PERIOD EVALUATION: {period_name} ({start_date} to {end_date})")
    print("=" * 95)
    headers = ["Strategy / Model", "Start Equity (INR)", "End Equity (INR)", "TWR Return (%)", "TWR Max Drawdown (%)"]
    print_table(headers, rows)


def main():
    print("=" * 115)
    print(" ANTI-WHIPSAW ENHANCED REGIME-SWITCHING SYSTEM (2018 - 2023)")
    print(" DUAL BACKTEST: Model 1 (Static Universe) vs. Model 2 (Point-in-Time Universe)")
    print(f" Initial Lumpsum: INR {CONFIG['INITIAL_CAPITAL']:,.2f} | Monthly SIP: INR {CONFIG['SIP_AMOUNT']:,.2f}")
    print("=" * 115)

    # 1. Fetch Market Data including India VIX and Point-in-Time Universe Matrix through 2026
    bench_series, etf_df, stocks_df, vix_series, pit_universe_df = fetch_all_market_data(CONFIG, end_date="2026-10-15")

    # 2. Run Multi-Model Backtest (instantiating Static vs PIT universes side-by-side)
    engine = BacktestEngine(
        bench_series, etf_df, stocks_df,
        vix_series=vix_series,
        pit_universe_df=pit_universe_df,
        config=CONFIG
    )
    results = engine.run()

    # 3. Print Transition Log for Dynamic 95th-Pct VIX
    transitions = results["transitions_hys"]
    print("\n" + "=" * 115)
    print(" DYNAMIC 95TH-PERCENTILE VIX TRANSITIONS AUDIT (ADAPTIVE MODEL)")
    print("=" * 115)
    trans_headers = ["Date", "Transition", "NIFTY 50", "200 SMA", "India VIX", "95th-Pct", "Dist (%)", "Trigger Reason"]
    trans_rows = []
    for t in transitions:
        trans_rows.append([
            t["date"].strftime("%Y-%m-%d"),
            f"{t['from_regime']} -> {t['to_regime']}",
            f"INR {t['benchmark_price']:,.2f}",
            f"INR {t['sma_200']:,.2f}",
            f"{t['vix']:.1f}",
            f"{t.get('vix_threshold', 0.0):.1f}",
            f"{t['distance_pct']:+.2f}%",
            t["reason"][:42]
        ])
    print_table(trans_headers, trans_rows)

    # 4. Full Period Performance Metrics (Time-Weighted metrics strictly use unitized NAV)
    total_inv = results["acc_dyn_vix_pit"].total_invested

    m_bin = compute_metrics_table(
        results["binary_series"], results["acc_binary"].cash_flows, total_inv,
        results["acc_binary"].cost_ledger.total_trades_count,
        results["acc_binary"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["binary_nav_series"]
    )
    m_static_vix = compute_metrics_table(
        results["static_vix_series"], results["acc_static_vix"].cash_flows, total_inv,
        results["acc_static_vix"].cost_ledger.total_trades_count,
        results["acc_static_vix"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["static_vix_nav_series"]
    )
    m_dyn_static = compute_metrics_table(
        results["dyn_vix_static_series"], results["acc_dyn_vix_static"].cash_flows, total_inv,
        results["acc_dyn_vix_static"].cost_ledger.total_trades_count,
        results["acc_dyn_vix_static"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["dyn_vix_static_nav_series"]
    )
    m_dyn_pit = compute_metrics_table(
        results["dyn_vix_pit_series"], results["acc_dyn_vix_pit"].cash_flows, total_inv,
        results["acc_dyn_vix_pit"].cost_ledger.total_trades_count,
        results["acc_dyn_vix_pit"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["dyn_vix_pit_nav_series"]
    )
    m_sig = compute_metrics_table(
        results["sigmoid_series"], results["acc_sigmoid"].cash_flows, total_inv,
        results["acc_sigmoid"].cost_ledger.total_trades_count,
        results["acc_sigmoid"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["sigmoid_nav_series"]
    )
    m_nifty = compute_metrics_table(
        results["nifty_series"], results["acc_nifty"].cash_flows, total_inv,
        results["acc_nifty"].cost_ledger.total_trades_count,
        results["acc_nifty"].cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"],
        nav_series=results["nifty_nav_series"]
    )

    print("\n" + "=" * 115)
    print(" FULL HORIZON PERFORMANCE COMPARISON (2018 - 2023)")
    num_sips = len(results["acc_dyn_vix_pit"].cash_flows) - 1
    print(f" Total Principal Invested: INR {total_inv:,.2f} (Initial INR {CONFIG['INITIAL_CAPITAL']:,.2f} + {num_sips} x INR {CONFIG['SIP_AMOUNT']:,.2f} SIP)")
    print("=" * 115)

    comp_headers = [
        "Metric",
        "NIFTY 50 Index",
        "Model 1: Static Univ",
        "Model 2: PIT Univ",
        "Hardcoded VIX (>=24)",
        "Flagship: PIT+Sigmoid+VIX"
    ]
    comp_table = [
        ["Final Portfolio Value", f"INR {m_nifty['final_value']:,.2f}", f"INR {m_dyn_static['final_value']:,.2f}", f"INR {m_dyn_pit['final_value']:,.2f}", f"INR {m_static_vix['final_value']:,.2f}", f"INR {m_sig['final_value']:,.2f}"],
        ["Net Realized Profit", f"INR {m_nifty['net_profit']:,.2f}", f"INR {m_dyn_static['net_profit']:,.2f}", f"INR {m_dyn_pit['net_profit']:,.2f}", f"INR {m_static_vix['net_profit']:,.2f}", f"INR {m_sig['net_profit']:,.2f}"],
        ["Cumulative Return (%)", f"{m_nifty['cumulative_return_pct']:.2f}%", f"{m_dyn_static['cumulative_return_pct']:.2f}%", f"{m_dyn_pit['cumulative_return_pct']:.2f}%", f"{m_static_vix['cumulative_return_pct']:.2f}%", f"{m_sig['cumulative_return_pct']:.2f}%"],
        ["Annualized TWR CAGR (%) [NAV]", f"{m_nifty['twr_cagr_pct']:.2f}%", f"{m_dyn_static['twr_cagr_pct']:.2f}%", f"{m_dyn_pit['twr_cagr_pct']:.2f}%", f"{m_static_vix['twr_cagr_pct']:.2f}%", f"{m_sig['twr_cagr_pct']:.2f}%"],
        ["Annualized XIRR (%) [Money-Weighted]", f"{m_nifty['xirr_cagr_pct']:.2f}%", f"{m_dyn_static['xirr_cagr_pct']:.2f}%", f"{m_dyn_pit['xirr_cagr_pct']:.2f}%", f"{m_static_vix['xirr_cagr_pct']:.2f}%", f"{m_sig['xirr_cagr_pct']:.2f}%"],
        ["Maximum Drawdown (MDD %)", f"{m_nifty['max_drawdown_pct']:.2f}%", f"{m_dyn_static['max_drawdown_pct']:.2f}%", f"{m_dyn_pit['max_drawdown_pct']:.2f}%", f"{m_static_vix['max_drawdown_pct']:.2f}%", f"{m_sig['max_drawdown_pct']:.2f}%"],
        ["Calmar Ratio (CAGR / MDD)", f"{m_nifty['calmar_ratio']:.2f}", f"{m_dyn_static['calmar_ratio']:.2f}", f"{m_dyn_pit['calmar_ratio']:.2f}", f"{m_static_vix['calmar_ratio']:.2f}", f"{m_sig['calmar_ratio']:.2f}"],
        ["Sharpe Ratio (Rf = 6.5%)", f"{m_nifty['sharpe_ratio']:.2f}", f"{m_dyn_static['sharpe_ratio']:.2f}", f"{m_dyn_pit['sharpe_ratio']:.2f}", f"{m_static_vix['sharpe_ratio']:.2f}", f"{m_sig['sharpe_ratio']:.2f}"],
        ["Total Trades Count", f"{m_nifty['trades_count']}", f"{m_dyn_static['trades_count']}", f"{m_dyn_pit['trades_count']}", f"{m_static_vix['trades_count']}", f"{m_sig['trades_count']}"],
        ["Total Turnover", f"INR {m_nifty['total_turnover']:,.2f}", f"INR {m_dyn_static['total_turnover']:,.2f}", f"INR {m_dyn_pit['total_turnover']:,.2f}", f"INR {m_static_vix['total_turnover']:,.2f}", f"INR {m_sig['total_turnover']:,.2f}"],
    ]
    print_table(comp_headers, comp_table)

    # 5. Survivorship Bias Elimination Delta
    cagr_drop = m_dyn_static['xirr_cagr_pct'] - m_dyn_pit['xirr_cagr_pct']
    mdd_delta = m_dyn_pit['max_drawdown_pct'] - m_dyn_static['max_drawdown_pct']
    profit_diff = m_dyn_static['net_profit'] - m_dyn_pit['net_profit']

    print("\n" + "=" * 115)
    print(" SURVIVORSHIP BIAS QUANTIFICATION DELTA (MODEL 1 STATIC vs. MODEL 2 PIT)")
    print("=" * 115)
    bias_headers = ["Metric", "Model 1: Static Universe", "Model 2: PIT Universe", "Survivorship Premium Delta"]
    bias_rows = [
        ["Annualized CAGR / XIRR", f"{m_dyn_static['xirr_cagr_pct']:.2f}%", f"{m_dyn_pit['xirr_cagr_pct']:.2f}%", f"{cagr_drop:+.2f}% CAGR drop (Survivorship Premium)"],
        ["Full Maximum Drawdown", f"{m_dyn_static['max_drawdown_pct']:.2f}%", f"{m_dyn_pit['max_drawdown_pct']:.2f}%", f"{mdd_delta:+.2f}% MDD increase"],
        ["Net Realized Profit", f"INR {m_dyn_static['net_profit']:,.2f}", f"INR {m_dyn_pit['net_profit']:,.2f}", f"-INR {profit_diff:,.2f}"],
        ["Calmar Ratio", f"{m_dyn_static['calmar_ratio']:.2f}", f"{m_dyn_pit['calmar_ratio']:.2f}", f"{m_dyn_pit['calmar_ratio'] - m_dyn_static['calmar_ratio']:+.2f}"],
        ["Sharpe Ratio", f"{m_dyn_static['sharpe_ratio']:.2f}", f"{m_dyn_pit['sharpe_ratio']:.2f}", f"{m_dyn_pit['sharpe_ratio'] - m_dyn_static['sharpe_ratio']:+.2f}"],
    ]
    print_table(bias_headers, bias_rows)

    # 6. Crisis Sub-Periods Evaluation (including 2018-2020 NBFC / YESBANK / IL&FS collapse)
    series_map = {
        "Flagship Gen 7": results["flagship_series"],
        "Model 2: PIT Universe": results["dyn_vix_pit_series"],
        "Model 1: Static Universe": results["dyn_vix_static_series"],
        "Hardcoded VIX (>= 24)": results["static_vix_series"],
        "NIFTY 50 Index": results["nifty_series"],
    }
    nav_map = {
        "Flagship Gen 7": results["flagship_nav_series"],
        "Model 2: PIT Universe": results["dyn_vix_pit_nav_series"],
        "Model 1: Static Universe": results["dyn_vix_static_nav_series"],
        "Hardcoded VIX (>= 24)": results["static_vix_nav_series"],
        "NIFTY 50 Index": results["nifty_nav_series"],
    }

    # Crisis Period 0: Jan 2018 – Mar 2020 (IL&FS Crisis, NBFC contagion, YESBANK & IBULHSGFIN collapse)
    evaluate_crisis_subperiod(series_map, nav_map, "2018-01-01", "2020-03-31", "Period 0: 2018-2020 IL&FS NBFC Contagion & YESBANK/IBULHSGFIN Collapse")

    # Period 1: Jan 2020 – Dec 2020 (COVID-19 V-shape crash and recovery)
    evaluate_crisis_subperiod(series_map, nav_map, "2020-01-01", "2020-12-31", "Period 1: COVID-19 V-Crash & Recovery")

    # Period 2: Jan 2022 – March 2023 (Russia-Ukraine war and rate-hike consolidation)
    evaluate_crisis_subperiod(series_map, nav_map, "2022-01-01", "2023-03-31", "Period 2: Ukraine War & Rate-Hike Consolidation")

    # 7. Audit Table: Historical Exited Constituents Traded Under PIT
    exited_trades = results.get("historical_exited_trades", [])
    print("\n" + "=" * 115)
    print(" LOGGED AUDIT: TRADES EXECUTED UNDER PIT INVOLVING EXITED/DISTRESSED CONSTITUENTS")
    print("=" * 115)
    if exited_trades:
        exit_headers = ["Date", "Ticker", "Action", "Weight (%)", "Price (INR)", "12-1m Mom Return", "Constituent Audit Context"]
        exit_rows = []
        for tr in exited_trades:
            exit_rows.append([
                tr["date"].strftime("%Y-%m-%d"),
                tr["ticker"],
                tr["action"],
                f"{tr['weight']*100:.1f}%",
                f"INR {tr['price']:,.2f}",
                f"{tr['mom_12_1_score']*100:+.2f}%",
                tr["note"]
            ])
        print_table(exit_headers, exit_rows)
    else:
        print(" [Audit] No historical distressed constituents qualified for top 5 momentum during Bull regimes in PIT simulation.")

    # 8. Render Comparison Visualizer
    out_img = os.path.join(_algo_root, "regime_switching", "anti_whipsaw_comparison.png")
    print(f"\n[Visualizer] Generating crisis-highlighted comparison chart at {out_img}...")
    plot_anti_whipsaw_comparison(results, output_path=out_img)

    # 9. Optimization Impact Comparison Table (Gen 5 vs Gen 6 vs Gen 7)
    GEN5 = {
        "label":      "Gen 5 Baseline",
        "cagr":       13.33,
        "mdd":        17.13,
        "calmar":     0.78,
        "sharpe":     1.95,
        "trades":     731,
        "turnover_l": 286.73,   # Lakhs
    }
    GEN6 = {
        "label":      "Gen 6 (Deadband + Snap + HIFO)",
        "cagr":       14.66,
        "mdd":        15.27,
        "calmar":     0.96,
        "sharpe":     1.99,
        "trades":     417,
        "turnover_l": 212.88,   # Lakhs
    }

    sig_turnover_lakhs = m_sig["total_turnover"] / 1e5
    sig_calmar = m_sig["calmar_ratio"]
    sig_sharpe = m_sig["sharpe_ratio"]

    print("\n" + "=" * 115)
    print(" OPTIMIZATION IMPACT ACROSS GENERATIONS: Gen 5 -> Gen 6 -> Gen 7 (Full Architecture)")
    print(" Targets: CAGR > Nifty (16.5%)  |  Trades < 280  |  Turnover < INR 150 Lakhs  |  MDD ~14-16%")
    print("=" * 115)
    opt_headers = ["Metric", "Nifty 50", "Gen 5 Baseline", "Gen 6", "Gen 7 (Current)", "Gen 7 vs Nifty Delta"]
    opt_rows = [
        [
            "Annualized CAGR / XIRR (%)",
            f"{m_nifty['xirr_cagr_pct']:.2f}%",
            f"{GEN5['cagr']:.2f}%",
            f"{GEN6['cagr']:.2f}%",
            f"{m_sig['xirr_cagr_pct']:.2f}%",
            f"{m_sig['xirr_cagr_pct'] - m_nifty['xirr_cagr_pct']:+.2f}%"
        ],
        [
            "Maximum Drawdown (MDD %)",
            f"{m_nifty['max_drawdown_pct']:.2f}%",
            f"{GEN5['mdd']:.2f}%",
            f"{GEN6['mdd']:.2f}%",
            f"{m_sig['max_drawdown_pct']:.2f}%",
            f"{m_sig['max_drawdown_pct'] - m_nifty['max_drawdown_pct']:+.2f}% (Risk Cut)"
        ],
        [
            "Calmar Ratio (CAGR / MDD)",
            f"{m_nifty['calmar_ratio']:.2f}",
            f"{GEN5['calmar']:.2f}",
            f"{GEN6['calmar']:.2f}",
            f"{sig_calmar:.2f}",
            f"{sig_calmar - m_nifty['calmar_ratio']:+.2f}"
        ],
        [
            "Sharpe Ratio (Rf = 6.5%)",
            f"{m_nifty['sharpe_ratio']:.2f}",
            f"{GEN5['sharpe']:.2f}",
            f"{GEN6['sharpe']:.2f}",
            f"{sig_sharpe:.2f}",
            f"{sig_sharpe - m_nifty['sharpe_ratio']:+.2f}"
        ],
        [
            "Total Trades Executed",
            f"{m_nifty['trades_count']}",
            f"{GEN5['trades']}",
            f"{GEN6['trades']}",
            f"{m_sig['trades_count']}",
            f"{m_sig['trades_count'] - GEN5['trades']:+d} vs Gen 5"
        ],
        [
            "Total Gross Turnover (INR Lakhs)",
            f"INR {m_nifty['total_turnover']/1e5:.2f} L",
            f"INR {GEN5['turnover_l']:.2f} L",
            f"INR {GEN6['turnover_l']:.2f} L",
            f"INR {sig_turnover_lakhs:.2f} L",
            f"{sig_turnover_lakhs - GEN5['turnover_l']:+.2f} L vs Gen 5"
        ],
    ]
    print_table(opt_headers, opt_rows)

    # 10. Run Signal-Layer 9-Scenario Sensitivity Grid (In-Sample 2018-2023)
    grid_results = run_sensitivity_grid(
        bench_series, etf_df, stocks_df, vix_series, pit_universe_df, CONFIG
    )

    # 11. Run Out-of-Sample (OOS) Validation (2024-2026 Frozen Parameters)
    m_flagship_oos, m_nifty_oos = run_out_of_sample_validation(
        bench_series, etf_df, stocks_df, vix_series, pit_universe_df, CONFIG
    )

    # 12. Run HNI & Family Office Capital Scale Benchmark (Nifty 50 vs. Gen 6 vs. Flagship Gen 7)
    run_hni_gen6_gen7_nifty_benchmark(
        bench_series, etf_df, stocks_df, vix_series, pit_universe_df, CONFIG
    )

    # 13. Run Retail ETF Proxy vs. HNI 5-Stock Basket Scalability Demonstration
    run_retail_etf_vs_hni_stocks(
        bench_series, etf_df, stocks_df, vix_series, pit_universe_df, CONFIG
    )

    print("\n" + "=" * 115)
    print(" ALL SIMULATIONS & AUDIT VALIDATIONS COMPLETE.")
    print("=" * 115)


def run_sensitivity_grid(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, base_config):
    """Evaluates 9 combinations of (d_0, upper_snap_dist) across the In-Sample period (2018 - 2023)."""
    d0_values = [-0.015, -0.010, 0.000]
    upper_snap_values = [0.015, 0.020, 0.030]

    grid_results = []
    print("\n" + "=" * 115)
    print(" SIGNAL-LAYER SENSITIVITY GRID (9 SCENARIOS: d_0 x UPPER_SNAP) [IN-SAMPLE 2018-2023]")
    print("=" * 115)

    for d0 in d0_values:
        for u_snap in upper_snap_values:
            cfg = dict(base_config)
            cfg["START_DATE"] = "2018-01-01"
            cfg["END_DATE"] = "2023-12-31"
            cfg["SIGMOID_D0"] = d0
            cfg["UPPER_SNAP_DIST"] = u_snap
            eng = BacktestEngine(
                bench_series, etf_df, stocks_df,
                vix_series=vix_series,
                pit_universe_df=pit_universe_df,
                config=cfg
            )
            res = eng.run()
            m = compute_metrics_table(
                res["sigmoid_series"], res["acc_sigmoid"].cash_flows,
                res["acc_sigmoid"].total_invested,
                res["acc_sigmoid"].cost_ledger.total_trades_count,
                res["acc_sigmoid"].cost_ledger.total_turnover,
                cfg["RISK_FREE_RATE"],
                nav_series=res["sigmoid_nav_series"]
            )
            is_baseline = (d0 == -0.010 and u_snap == 0.020)
            tag = " (Baseline)" if is_baseline else ""
            grid_results.append({
                "d0": d0,
                "upper_snap": u_snap,
                "cagr": m["xirr_cagr_pct"],
                "mdd": m["max_drawdown_pct"],
                "calmar": m["calmar_ratio"],
                "sharpe": m["sharpe_ratio"],
                "trades": m["trades_count"],
                "turnover_l": m["total_turnover"] / 1e5,
                "tag": tag
            })

    headers = [
        "d_0 (Offset)", "Delta_Upper (Snap)", "CAGR (%)", "MDD (%) [TWR]",
        "Calmar Ratio", "Sharpe Ratio [TWR]", "Trades", "Turnover (INR L)", "Notes"
    ]
    rows = []
    for r in grid_results:
        rows.append([
            f"{r['d0']*100:+.1f}%",
            f"{r['upper_snap']*100:+.1f}%",
            f"{r['cagr']:.2f}%",
            f"{r['mdd']:.2f}%",
            f"{r['calmar']:.2f}",
            f"{r['sharpe']:.2f}",
            f"{r['trades']}",
            f"INR {r['turnover_l']:.2f} L",
            r["tag"]
        ])
    print_table(headers, rows)
    return grid_results


def run_out_of_sample_validation(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, base_config):
    """Runs Out-of-Sample (OOS) validation from 2024-01-01 to 2026-10-14 with frozen parameters."""
    oos_cfg = dict(base_config)
    oos_cfg["START_DATE"] = "2024-01-01"
    oos_cfg["END_DATE"] = "2026-10-14"

    print("\n" + "=" * 115)
    print(" OUT-OF-SAMPLE (OOS) VALIDATION: 2024-01-01 TO 2026-10-14 (FROZEN PARAMETERS)")
    print(f" Initial Lumpsum: INR {oos_cfg['INITIAL_CAPITAL']:,.2f} | Monthly SIP: INR {oos_cfg['SIP_AMOUNT']:,.2f}")
    print("=" * 115)

    eng_oos = BacktestEngine(
        bench_series, etf_df, stocks_df,
        vix_series=vix_series,
        pit_universe_df=pit_universe_df,
        config=oos_cfg
    )
    res_oos = eng_oos.run()

    total_inv_oos = res_oos["acc_flagship"].total_invested

    m_flagship_oos = compute_metrics_table(
        res_oos["flagship_series"], res_oos["acc_flagship"].cash_flows, total_inv_oos,
        res_oos["acc_flagship"].cost_ledger.total_trades_count,
        res_oos["acc_flagship"].cost_ledger.total_turnover, oos_cfg["RISK_FREE_RATE"],
        nav_series=res_oos["flagship_nav_series"]
    )
    m_nifty_oos = compute_metrics_table(
        res_oos["nifty_series"], res_oos["acc_nifty"].cash_flows, total_inv_oos,
        res_oos["acc_nifty"].cost_ledger.total_trades_count,
        res_oos["acc_nifty"].cost_ledger.total_turnover, oos_cfg["RISK_FREE_RATE"],
        nav_series=res_oos["nifty_nav_series"]
    )

    oos_headers = ["Metric", "Nifty 50 Buy & Hold (OOS)", "Flagship Gen 7 (OOS)", "Alpha / Delta"]
    oos_rows = [
        ["Total Principal Invested", f"INR {total_inv_oos:,.2f}", f"INR {total_inv_oos:,.2f}", "-"],
        ["Final Portfolio Value", f"INR {m_nifty_oos['final_value']:,.2f}", f"INR {m_flagship_oos['final_value']:,.2f}", f"{'+' if m_flagship_oos['final_value']>=m_nifty_oos['final_value'] else ''}INR {m_flagship_oos['final_value']-m_nifty_oos['final_value']:,.2f}"],
        ["Net Realized Profit", f"INR {m_nifty_oos['net_profit']:,.2f}", f"INR {m_flagship_oos['net_profit']:,.2f}", f"{'+' if m_flagship_oos['net_profit']>=m_nifty_oos['net_profit'] else ''}INR {m_flagship_oos['net_profit']-m_nifty_oos['net_profit']:,.2f}"],
        ["Cumulative Return (%)", f"{m_nifty_oos['cumulative_return_pct']:.2f}%", f"{m_flagship_oos['cumulative_return_pct']:.2f}%", f"{m_flagship_oos['cumulative_return_pct']-m_nifty_oos['cumulative_return_pct']:+.2f}%"],
        ["Annualized TWR CAGR (%) [NAV]", f"{m_nifty_oos['twr_cagr_pct']:.2f}%", f"{m_flagship_oos['twr_cagr_pct']:.2f}%", f"{m_flagship_oos['twr_cagr_pct']-m_nifty_oos['twr_cagr_pct']:+.2f}%"],
        ["Annualized XIRR (%) [Money-Weighted]", f"{m_nifty_oos['xirr_cagr_pct']:.2f}%", f"{m_flagship_oos['xirr_cagr_pct']:.2f}%", f"{m_flagship_oos['xirr_cagr_pct']-m_nifty_oos['xirr_cagr_pct']:+.2f}%"],
        ["Maximum Drawdown (MDD %) [TWR]", f"{m_nifty_oos['max_drawdown_pct']:.2f}%", f"{m_flagship_oos['max_drawdown_pct']:.2f}%", f"{m_flagship_oos['max_drawdown_pct']-m_nifty_oos['max_drawdown_pct']:+.2f}% (Risk Cut)"],
        ["Calmar Ratio (CAGR / MDD)", f"{m_nifty_oos['calmar_ratio']:.2f}", f"{m_flagship_oos['calmar_ratio']:.2f}", f"{m_flagship_oos['calmar_ratio']-m_nifty_oos['calmar_ratio']:+.2f}"],
        ["Sharpe Ratio (Rf = 6.5%) [TWR]", f"{m_nifty_oos['sharpe_ratio']:.2f}", f"{m_flagship_oos['sharpe_ratio']:.2f}", f"{m_flagship_oos['sharpe_ratio']-m_nifty_oos['sharpe_ratio']:+.2f}"],
        ["Total Trades Count", f"{m_nifty_oos['trades_count']}", f"{m_flagship_oos['trades_count']}", f"{m_flagship_oos['trades_count']-m_nifty_oos['trades_count']:+d}"],
        ["Total Turnover", f"INR {m_nifty_oos['total_turnover']/1e5:.2f} L", f"INR {m_flagship_oos['total_turnover']/1e5:.2f} L", f"INR {(m_flagship_oos['total_turnover']-m_nifty_oos['total_turnover'])/1e5:+.2f} L"],
    ]
    print_table(oos_headers, oos_rows)
    return m_flagship_oos, m_nifty_oos


def run_gen6_account_simulation(
    bench_series: pd.Series,
    etf_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    vix_series: pd.Series,
    pit_universe_df: pd.DataFrame,
    config: dict
):
    """Simulates exact Generation 6 architecture on the same Unitized TWR NAV and statutory tax ledger."""
    try:
        from regime_switching.backtest_engine import PortfolioAccount
        from regime_switching.regime_filter import RegimeFilter
        from regime_switching.risk_parity import RiskParityModel
        from regime_switching.momentum import MomentumModel
    except ImportError:
        from algo_trading_system.regime_switching.backtest_engine import PortfolioAccount
        from algo_trading_system.regime_switching.regime_filter import RegimeFilter
        from algo_trading_system.regime_switching.risk_parity import RiskParityModel
        from algo_trading_system.regime_switching.momentum import MomentumModel

    cfg = dict(config)
    acc_gen6 = PortfolioAccount(cfg, name="Generation 6 (Deadband + 3.5% Snap)")

    filter_gen6 = RegimeFilter(
        bench_series, vix_series=vix_series, sma_window=cfg["TREND_SMA_DAYS"],
        mode="CONTINUOUS_SIGMOID", vix_abs_threshold=None,
        vix_percentile_window=cfg.get("VIX_PERCENTILE_WINDOW", 252),
        vix_percentile_threshold=cfg.get("VIX_PERCENTILE", 0.95),
        vix_sma_window=cfg.get("VIX_SMA_WINDOW", 20),
        vix_rel_multiplier=cfg.get("VIX_REL_MULTIPLIER", None),
        use_adaptive_sigmoid_k=True, adaptive_k_constant=5.25,
        sigmoid_d0=0.0, upper_snap_dist=0.035, lower_snap_dist=-1.0
    )
    rp_model = RiskParityModel(etf_df, lookback_days=cfg["VOL_LOOKBACK"], min_weight=cfg["MIN_WEIGHT"], max_weight=cfg["MAX_WEIGHT"])
    mom_pit = MomentumModel(stocks_df, lookback_long=cfg["MOMENTUM_LOOKBACK_LONG"], skip_short=cfg["MOMENTUM_SKIP_SHORT"], top_n=cfg["MOMENTUM_TOP_N"], keep_rank=5, mode=cfg["MOMENTUM_MODE"], etf_ticker=cfg["MOMENTUM_ETF"], pit_universe_df=pit_universe_df)

    start_date, end_date = pd.Timestamp(cfg["START_DATE"]), pd.Timestamp(cfg["END_DATE"])
    sim_dates = [d for d in bench_series.index if start_date <= d <= end_date]
    first_date = sim_dates[0]

    monthly_dates = []
    date_series = pd.Series(sim_dates, index=sim_dates)
    for _, group in date_series.groupby([date_series.index.year, date_series.index.month]):
        monthly_dates.append(group.iloc[-1])
    rebalance_dates = sorted(list(set(monthly_dates) | {first_date}))

    prev_fy = first_date.year if first_date.month >= 4 else first_date.year - 1
    rp_etf_tickers = set(cfg["RISK_PARITY_TICKERS"].values())

    for dt in sim_dates:
        curr_fy = dt.year if dt.month >= 4 else dt.year - 1
        if curr_fy != prev_fy:
            acc_gen6.process_year_end_tax()
            prev_fy = curr_fy

        prices_snap = {col: float(etf_df[col].asof(dt)) for col in etf_df.columns}
        for col in stocks_df.columns:
            prices_snap[col] = float(stocks_df[col].asof(dt))
        prices_snap["^NSEI"] = float(bench_series.asof(dt))

        if dt in rebalance_dates:
            inflow = cfg["INITIAL_CAPITAL"] if dt == first_date else cfg["SIP_AMOUNT"]
            if inflow > 0:
                acc_gen6.inject_cash(dt, inflow, prices=prices_snap)

            _, w_mom_s, w_rp_s, _, _ = filter_gen6.check_and_log_transition(dt)
            rp_weights = rp_model.calculate_weights_at(dt)
            mom_weights_pit, _ = mom_pit.calculate_weights_at(dt, current_holdings=None)

            gen6_equity = acc_gen6.get_equity(prices_snap)
            if gen6_equity > 0:
                gen6_mom_val = sum(u * prices_snap.get(s, 0.0) for s, u in acc_gen6.holdings.items() if s not in rp_etf_tickers and u > 1e-4)
                curr_w_mom = gen6_mom_val / gen6_equity
            else:
                curr_w_mom = 0.0

            if (dt == first_date) or abs(w_mom_s - curr_w_mom) >= 0.05:
                all_syms = set(list(mom_weights_pit.keys()) + list(rp_weights.keys()))
                blended = {s: (mom_weights_pit.get(s, 0.0) * w_mom_s) + (rp_weights.get(s, 0.0) * w_rp_s) for s in all_syms}
                tot = sum(v for v in blended.values() if v > 1e-5)
                if tot > 0:
                    acc_gen6.rebalance(dt, {k: v / tot for k, v in blended.items() if v > 1e-5}, prices_snap)

        eq = acc_gen6.get_equity(prices_snap)
        acc_gen6.daily_equity[dt] = eq
        acc_gen6.daily_nav[dt] = (eq / acc_gen6.total_units) if acc_gen6.total_units > 1e-9 else 100.0

    acc_gen6.process_year_end_tax()
    return pd.Series(acc_gen6.daily_equity), pd.Series(acc_gen6.daily_nav), acc_gen6


def run_hni_gen6_gen7_nifty_benchmark(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, base_config):
    """Runs head-to-head comparison of ONLY Nifty 50, Generation 6, and Flagship Generation 7."""
    tiers = [
        {"name": "Tier 1: Retail SIP", "init": 100000.0, "sip": 10000.0},
        {"name": "Tier 2: HNI SIP", "init": 2500000.0, "sip": 250000.0},
        {"name": "Tier 3: Lumpsum", "init": 10000000.0, "sip": 0.0},
    ]
    periods = [
        {"label": "IN-SAMPLE (2018-01-01 to 2023-12-31)", "start": "2018-01-01", "end": "2023-12-31"},
        {"label": "OUT-OF-SAMPLE (2024-01-01 to 2026-10-14)", "start": "2024-01-01", "end": "2026-10-14"},
    ]

    for p in periods:
        print("\n" + "=" * 130)
        print(f" HNI & FAMILY OFFICE BENCHMARK (NIFTY 50 vs. GEN 6 vs. GEN 7) — {p['label']}")
        print("=" * 130)
        headers = ["Capital Tier & Strategy", "Principal", "Final Value", "Net Profit", "TWR CAGR", "XIRR", "TWR MDD", "Calmar", "Sharpe", "Trades", "DP Fees", "Total Friction"]
        rows = []
        for t in tiers:
            cfg = dict(base_config)
            cfg["START_DATE"] = p["start"]
            cfg["END_DATE"] = p["end"]
            cfg["INITIAL_CAPITAL"] = t["init"]
            cfg["SIP_AMOUNT"] = t["sip"]

            eng = BacktestEngine(bench_series, etf_df, stocks_df, vix_series=vix_series, pit_universe_df=pit_universe_df, config=cfg)
            res = eng.run()
            gen6_eq, gen6_nav, acc_gen6 = run_gen6_account_simulation(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, cfg)

            for label, eq_s, nav_s, acc in [
                (f"{t['name']} | NIFTY 50", res["nifty_series"], res["nifty_nav_series"], res["acc_nifty"]),
                (f"{t['name']} | Gen 6", gen6_eq, gen6_nav, acc_gen6),
                (f"{t['name']} | Flagship Gen 7", res["flagship_series"], res["flagship_nav_series"], res["acc_flagship"]),
            ]:
                m = compute_metrics_table(eq_s, acc.cash_flows, acc.total_invested, acc.cost_ledger.total_trades_count, acc.cost_ledger.total_turnover, cfg["RISK_FREE_RATE"], nav_series=nav_s)
                f_sum = acc.cost_ledger.get_friction_summary()
                rows.append([
                    label, f"INR {acc.total_invested/1e5:,.2f} L", f"INR {m['final_value']/1e5:,.2f} L", f"INR {m['net_profit']/1e5:+,.2f} L",
                    f"{m['twr_cagr_pct']:+.2f}%", f"{m['xirr_cagr_pct']:+.2f}%", f"{m['max_drawdown_pct']:.2f}%",
                    f"{m['calmar_ratio']:.2f}", f"{m['sharpe_ratio']:.2f}", f"{m['trades_count']}",
                    f"INR {f_sum['total_dp_charges']:,.0f}", f"INR {f_sum['total_frictions']:,.0f}"
                ])
        print_table(headers, rows)


def run_retail_etf_vs_hni_stocks(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, base_config):
    """Demonstrates that small accounts must use an ETF proxy to survive transaction friction,
    while ₹1 Crore institutional accounts have the scale to run the 5-stock momentum basket.
    """
    print("\n" + "=" * 130)
    print(" RETAIL ETF PROXY vs. HNI 5-STOCK BASKET SCALABILITY DEMONSTRATION (2018-2023)")
    print("=" * 130)
    
    headers = [
        "Account Type & Setup", "Principal", "Final Value", "TWR CAGR", "TWR MDD", 
        "Sharpe", "Trades", "DP Fees", "Total Friction"
    ]
    rows = []
    
    # 1. Retail SIP (₹1 Lakh + ₹10k/m) running 5-Stock Basket (High Friction Drag)
    cfg_retail_stocks = dict(base_config)
    cfg_retail_stocks["INITIAL_CAPITAL"] = 100000.0
    cfg_retail_stocks["SIP_AMOUNT"] = 10000.0
    cfg_retail_stocks["MOMENTUM_MODE"] = "STOCK_BASKET"
    
    eng_retail_stocks = BacktestEngine(bench_series, etf_df, stocks_df, vix_series=vix_series, pit_universe_df=pit_universe_df, config=cfg_retail_stocks)
    res_retail_stocks = eng_retail_stocks.run()
    acc_rs = res_retail_stocks["acc_flagship"]
    m_rs = compute_metrics_table(res_retail_stocks["flagship_series"], acc_rs.cash_flows, acc_rs.total_invested, acc_rs.cost_ledger.total_trades_count, acc_rs.cost_ledger.total_turnover, cfg_retail_stocks["RISK_FREE_RATE"], nav_series=res_retail_stocks["flagship_nav_series"])
    f_rs = acc_rs.cost_ledger.get_friction_summary()
    
    rows.append([
        "Retail SIP (1L+10k) - 5 Stocks", f"INR {acc_rs.total_invested/1e5:.2f} L", f"INR {m_rs['final_value']/1e5:.2f} L",
        f"{m_rs['twr_cagr_pct']:+.2f}%", f"{m_rs['max_drawdown_pct']:.2f}%", f"{m_rs['sharpe_ratio']:.2f}",
        f"{m_rs['trades_count']}", f"INR {f_rs['total_dp_charges']:,.0f}", f"INR {f_rs['total_frictions']:,.0f}"
    ])

    # 2. Retail SIP (₹1 Lakh + ₹10k/m) running ETF Proxy (Low Friction)
    cfg_retail_etf = dict(base_config)
    cfg_retail_etf["INITIAL_CAPITAL"] = 100000.0
    cfg_retail_etf["SIP_AMOUNT"] = 10000.0
    cfg_retail_etf["MOMENTUM_MODE"] = "ETF_PROXY"
    cfg_retail_etf["MOMENTUM_ETF"] = "NIFTYBEES.NS" # Using NiftyBees as proxy since specific Mom ETF might lack 2018 history
    
    eng_retail_etf = BacktestEngine(bench_series, etf_df, stocks_df, vix_series=vix_series, pit_universe_df=pit_universe_df, config=cfg_retail_etf)
    res_retail_etf = eng_retail_etf.run()
    acc_re = res_retail_etf["acc_flagship"]
    m_re = compute_metrics_table(res_retail_etf["flagship_series"], acc_re.cash_flows, acc_re.total_invested, acc_re.cost_ledger.total_trades_count, acc_re.cost_ledger.total_turnover, cfg_retail_etf["RISK_FREE_RATE"], nav_series=res_retail_etf["flagship_nav_series"])
    f_re = acc_re.cost_ledger.get_friction_summary()
    
    rows.append([
        "Retail SIP (1L+10k) - ETF Proxy", f"INR {acc_re.total_invested/1e5:.2f} L", f"INR {m_re['final_value']/1e5:.2f} L",
        f"{m_re['twr_cagr_pct']:+.2f}%", f"{m_re['max_drawdown_pct']:.2f}%", f"{m_re['sharpe_ratio']:.2f}",
        f"{m_re['trades_count']}", f"INR {f_re['total_dp_charges']:,.0f}", f"INR {f_re['total_frictions']:,.0f}"
    ])

    # 3. Family Office Lumpsum (₹1 Crore) running 5-Stock Basket (Scale absorbs friction)
    cfg_hni_stocks = dict(base_config)
    cfg_hni_stocks["INITIAL_CAPITAL"] = 10000000.0
    cfg_hni_stocks["SIP_AMOUNT"] = 0.0
    cfg_hni_stocks["MOMENTUM_MODE"] = "STOCK_BASKET"
    
    eng_hni_stocks = BacktestEngine(bench_series, etf_df, stocks_df, vix_series=vix_series, pit_universe_df=pit_universe_df, config=cfg_hni_stocks)
    res_hni_stocks = eng_hni_stocks.run()
    acc_hs = res_hni_stocks["acc_flagship"]
    m_hs = compute_metrics_table(res_hni_stocks["flagship_series"], acc_hs.cash_flows, acc_hs.total_invested, acc_hs.cost_ledger.total_trades_count, acc_hs.cost_ledger.total_turnover, cfg_hni_stocks["RISK_FREE_RATE"], nav_series=res_hni_stocks["flagship_nav_series"])
    f_hs = acc_hs.cost_ledger.get_friction_summary()
    
    rows.append([
        "Family Office Lumpsum (1Cr) - 5 Stocks", f"INR {acc_hs.total_invested/1e5:.2f} L", f"INR {m_hs['final_value']/1e5:.2f} L",
        f"{m_hs['twr_cagr_pct']:+.2f}%", f"{m_hs['max_drawdown_pct']:.2f}%", f"{m_hs['sharpe_ratio']:.2f}",
        f"{m_hs['trades_count']}", f"INR {f_hs['total_dp_charges']:,.0f}", f"INR {f_hs['total_frictions']:,.0f}"
    ])

    print_table(headers, rows)


if __name__ == "__main__":
    main()
