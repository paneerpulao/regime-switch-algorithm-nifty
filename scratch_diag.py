import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, r"c:\Users\Paneer\code\algo_trading_system")
from regime_switching.config import CONFIG
from regime_switching.data_loader import fetch_all_market_data
from regime_switching.backtest_engine import BacktestEngine
from regime_switching.performance_metrics import compute_metrics_table

bench_series, etf_df, stocks_df, vix_series, pit_universe_df = fetch_all_market_data(CONFIG)

start_date = pd.Timestamp(CONFIG["START_DATE"])
end_date = pd.Timestamp(CONFIG["END_DATE"])
sim_dates = [d for d in bench_series.index if start_date <= d <= end_date]
first_date = sim_dates[0]
monthly_dates = []
date_series = pd.Series(sim_dates, index=sim_dates)
for _, group in date_series.groupby([date_series.index.year, date_series.index.month]):
    monthly_dates.append(group.iloc[-1])
rebalance_dates = sorted(list(set(monthly_dates) | {first_date}))

def run_custom(use_vol_norm=False, keep_rank=8, d0=-0.010, upper_snap=0.020, lower_snap=-0.045, k_up=6.50, macro_deadband=0.05, min_trim=0.025):
    engine = BacktestEngine(bench_series, etf_df, stocks_df, vix_series, pit_universe_df, CONFIG)
    
    # Custom rebalance with min_trim
    def custom_rebalance(self_acc, date, target_weights, prices):
        current_equity = self_acc.get_equity(prices)
        if current_equity <= 0: return
        symbols_to_check = set(list(self_acc.holdings.keys()) + list(target_weights.keys()))
        sells_plan = []
        buys_plan = []
        MIN_EXIT_VALUE = current_equity * 0.002
        MIN_TRIM_VALUE = current_equity * min_trim
        for sym in symbols_to_check:
            px = prices.get(sym, 0.0)
            if px <= 0: continue
            curr_units = self_acc.holdings.get(sym, 0.0)
            curr_val = curr_units * px
            tgt_w = target_weights.get(sym, 0.0)
            tgt_val = tgt_w * current_equity
            diff = tgt_val - curr_val
            threshold = MIN_EXIT_VALUE if (tgt_w < 1e-5 or curr_units <= 1e-4) else MIN_TRIM_VALUE
            if diff < -threshold:
                sells_plan.append((sym, -diff, px, curr_units))
            elif diff > threshold:
                buys_plan.append((sym, diff, px))
        
        # Execute sells
        for sym, sell_val_raw, px, curr_units in sells_plan:
            units_to_sell = min(curr_units, sell_val_raw / px)
            if units_to_sell <= 0: continue
            effective_px = px * (1.0 - self_acc.config["SLIPPAGE_PER_SIDE"])
            turnover = units_to_sell * effective_px
            sym_lots = self_acc.lot_book.get(sym, [])
            sym_lots_sorted = sorted(sym_lots, key=lambda t: t[0], reverse=True)
            remaining_to_sell = units_to_sell
            total_cost_sold = 0.0
            total_units_consumed = 0.0
            new_lots = []
            for lot_price, lot_units in sym_lots_sorted:
                if remaining_to_sell <= 1e-9:
                    new_lots.append((lot_price, lot_units))
                    continue
                consume = min(lot_units, remaining_to_sell)
                total_cost_sold += lot_price * consume
                total_units_consumed += consume
                remaining_to_sell -= consume
                leftover = lot_units - consume
                if leftover > 1e-9: new_lots.append((lot_price, leftover))
            hifo_cost_basis = (total_cost_sold / total_units_consumed) if total_units_consumed > 1e-9 else effective_px
            realized_pnl = (effective_px - hifo_cost_basis) * units_to_sell
            self_acc.lot_book[sym] = new_lots
            self_acc._recompute_avg_cost(sym)
            frictions = self_acc.cost_ledger.record_sell(turnover, realized_pnl)
            self_acc.cash += turnover - (frictions["stt"] + frictions["dp_charge"])
            rem_u = sum(u for _, u in new_lots)
            if rem_u <= 1e-4:
                self_acc.holdings[sym] = 0.0
                self_acc.avg_cost[sym] = 0.0
                self_acc.lot_book[sym] = []
            else:
                self_acc.holdings[sym] = rem_u

        # Execute buys
        total_buy_desired = sum(b[1] for b in buys_plan)
        if total_buy_desired > 0 and self_acc.cash > 0:
            capital_scaler = min(1.0, (self_acc.cash * 0.999) / total_buy_desired)
            for sym, buy_val_desired, px in buys_plan:
                alloc_cash = buy_val_desired * capital_scaler
                effective_px = px * (1.0 + self_acc.config["SLIPPAGE_PER_SIDE"])
                unit_cost = effective_px * (1.0 + self_acc.config["STT_RATE"])
                units_to_buy = alloc_cash / unit_cost
                if units_to_buy <= 0: continue
                turnover = units_to_buy * effective_px
                frictions = self_acc.cost_ledger.record_buy(turnover)
                self_acc.cash -= (turnover + frictions["stt"])
                if sym not in self_acc.lot_book: self_acc.lot_book[sym] = []
                self_acc.lot_book[sym].append((effective_px, units_to_buy))
                self_acc._recompute_avg_cost(sym)
                self_acc.holdings[sym] = sum(u for _, u in self_acc.lot_book[sym])
        self_acc.holdings = {k: v for k, v in self_acc.holdings.items() if v > 1e-4}
        self_acc.lot_book = {k: v for k, v in self_acc.lot_book.items() if sum(u for _, u in v) > 1e-4}

    # Bind method to acc_flagship
    engine.acc_flagship.rebalance = lambda dt, tgt, px: custom_rebalance(engine.acc_flagship, dt, tgt, px)

    prev_year = first_date.year
    for dt in sim_dates:
        if dt.year != prev_year:
            engine.acc_flagship.process_year_end_tax()
            prev_year = dt.year

        prices_snap = {col: float(etf_df[col].asof(dt)) for col in etf_df.columns}
        for col in stocks_df.columns:
            prices_snap[col] = float(stocks_df[col].asof(dt))
        prices_snap["^NSEI"] = float(bench_series.asof(dt))

        if dt in rebalance_dates:
            inflow = CONFIG["INITIAL_CAPITAL"] if (dt == first_date) else CONFIG["SIP_AMOUNT"]
            engine.acc_flagship.inject_cash(dt, inflow)
            p_nifty = bench_series.loc[dt]
            sma_200 = float(engine.filter_sigmoid.sma.asof(dt)) if dt in engine.filter_sigmoid.sma.index else p_nifty
            d = (p_nifty / sma_200) - 1.0
            d_shifted = d - d0
            vix_val = float(vix_series.asof(dt)) if vix_series is not None else 15.0
            vix_pct_val = float(engine.filter_sigmoid.vix_percentile_series.asof(dt)) if engine.filter_sigmoid.vix_percentile_series is not None else 20.0
            vix_shock = (vix_val >= vix_pct_val)

            if vix_shock:
                w_mom_s = 0.0
                w_rp_s = 1.0
            elif upper_snap is not None and d >= upper_snap:
                w_mom_s = 1.0
                w_rp_s = 0.0
            elif lower_snap is not None and d <= lower_snap:
                w_mom_s = 0.0
                w_rp_s = 1.0
            else:
                vol_val = float(engine.filter_sigmoid.benchmark_vol_series.asof(dt)) if dt in engine.filter_sigmoid.benchmark_vol_series.index else 0.15
                vol_val = max(vol_val, 0.05)
                eff_k = float((k_up if d >= d0 else 5.25) / vol_val)
                w_mom_s = float(1.0 / (1.0 + np.exp(-eff_k * d_shifted)))
                w_rp_s = 1.0 - w_mom_s

            rp_weights = engine.risk_parity_model.calculate_weights_at(dt)
            flagship_curr_stocks = [s for s in engine.acc_flagship.holdings.keys() if s not in CONFIG["RISK_PARITY_TICKERS"].values()]
            
            if not use_vol_norm:
                sub_df = stocks_df.loc[:dt]
                row = pit_universe_df.loc[dt] if pit_universe_df is not None and dt in pit_universe_df.index else None
                cands = [t for t in row.index if row[t] == 1 and t in stocks_df.columns] if row is not None else list(stocks_df.columns)
                if len(sub_df) < 252:
                    mom_weights_pit = {c: 0.20 for c in cands[:5]}
                else:
                    mom_scores = {}
                    p_skip = sub_df.iloc[-21]
                    p_long = sub_df.iloc[-252]
                    for c in cands:
                        if len(sub_df[c].dropna()) >= 252:
                            p0 = float(p_long[c])
                            p1 = float(p_skip[c])
                            if p0 > 0 and not np.isnan(p1) and not np.isnan(p0):
                                mom_scores[c] = (p1 / p0) - 1.0
                    sorted_c = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                    top_keep = {t for t, _ in sorted_c[:keep_rank]}
                    sel = []
                    for t, _ in sorted_c:
                        if t in flagship_curr_stocks and t in top_keep and len(sel) < 5:
                            sel.append(t)
                    for t, _ in sorted_c:
                        if len(sel) >= 5: break
                        if t not in sel: sel.append(t)
                    mom_weights_pit = {t: 0.20 for t in sel}
            else:
                mom_weights_pit, _ = engine.momentum_pit.calculate_weights_at(dt, current_holdings=flagship_curr_stocks)

            # Deadband check
            flagship_equity = engine.acc_flagship.get_equity(prices_snap)
            rp_etf_tickers = set(CONFIG["RISK_PARITY_TICKERS"].values())
            if flagship_equity > 0:
                mom_val = sum(u * prices_snap.get(s, 0.0) for s, u in engine.acc_flagship.holdings.items() if s not in rp_etf_tickers and u > 1e-4)
                current_w_mom = mom_val / flagship_equity
            else:
                current_w_mom = 0.0

            if abs(w_mom_s - current_w_mom) >= macro_deadband:
                tgt = engine._blend_weights(mom_weights_pit, rp_weights, w_mom_s, w_rp_s)
                engine.acc_flagship.rebalance(dt, tgt, prices_snap)
            else:
                if engine.acc_flagship.holdings:
                    tot_h = sum(u * prices_snap.get(s, 0.0) for s, u in engine.acc_flagship.holdings.items() if u > 1e-4)
                    if tot_h > 0:
                        chw = {s: (u * prices_snap.get(s, 0.0)) / tot_h for s, u in engine.acc_flagship.holdings.items() if u > 1e-4}
                        engine.acc_flagship.deploy_idle_cash(dt, chw, prices_snap)

        engine.acc_flagship.daily_equity[dt] = engine.acc_flagship.get_equity(prices_snap)

    engine.acc_flagship.process_year_end_tax()
    series = pd.Series(engine.acc_flagship.daily_equity)
    m = compute_metrics_table(series, engine.acc_flagship.cash_flows, engine.acc_flagship.total_invested, engine.acc_flagship.cost_ledger.total_trades_count, engine.acc_flagship.cost_ledger.total_turnover, CONFIG["RISK_FREE_RATE"])
    return m

print("Running parameter tests...")
for db in [0.05, 0.06, 0.07]:
    for mt in [0.025, 0.035, 0.045]:
        m = run_custom(use_vol_norm=False, keep_rank=8, d0=-0.010, upper_snap=0.020, lower_snap=-0.045, k_up=6.50, macro_deadband=db, min_trim=mt)
        label = f"Deadband={db*100:.0f}% | MinTrim={mt*100:.1f}%"
        print(f"{label} => CAGR: {m['xirr_cagr_pct']:.2f}% | MDD: {m['max_drawdown_pct']:.2f}% | Trades: {m['trades_count']} | TO: {m['total_turnover']/1e5:.1f}L")

