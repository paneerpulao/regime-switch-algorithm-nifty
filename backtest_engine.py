"""Modular backtest simulation engine comparing:
1. Strategy with Basic Binary 200-day SMA
2. Strategy with Hysteresis Buffer + India VIX Circuit Breakers
3. Strategy with Continuous Sigmoid Allocation Sizing
4. Baseline NIFTY 50 Buy & Hold
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .config import CONFIG
    from .regime_filter import RegimeFilter
    from .risk_parity import RiskParityModel
    from .momentum import MomentumModel
    from .cost_ledger import CostLedger
except (ImportError, ValueError):
    from regime_switching.config import CONFIG
    from regime_switching.regime_filter import RegimeFilter
    from regime_switching.risk_parity import RiskParityModel
    from regime_switching.momentum import MomentumModel
    from regime_switching.cost_ledger import CostLedger


class PortfolioAccount:
    """Manages cash, asset positions, friction deductions, and trade settlement.

    Tax Lot Accounting Method: HIFO (Highest-In, First-Out).
    When reducing a position, lots are sorted descending by purchase price and
    consumed from the top. This maximises the cost basis of sold units, minimising
    Net Realized PnL and therefore STCG tax crystallisation in any given year.
    The avg_cost dict is kept as a derived scalar (for legacy compatibility with
    equity valuation helpers) and is recomputed after every lot-book mutation.
    """

    def __init__(self, config: Dict, name: str = "Portfolio"):
        self.config = config
        self.name = name
        self.cash = 0.0
        self.holdings: Dict[str, float] = {}   # symbol -> total units (derived from lot_book)
        self.avg_cost: Dict[str, float] = {}   # symbol -> weighted avg cost (derived scalar)
        # Lot Book: symbol -> list of (purchase_date, purchase_price_per_unit, units_held)
        # Supports statutory FIFO (sorted by purchase_date) and HIFO (sorted by purchase_price).
        self.lot_book: Dict[str, List[Tuple[pd.Timestamp, float, float]]] = {}
        self.cost_ledger = CostLedger(config)
        self.cash_flows: List[Tuple[pd.Timestamp, float]] = []
        self.total_invested = 0.0
        self.daily_equity: Dict[pd.Timestamp, float] = {}
        # Unitized NAV tracking (Time-Weighted Return) to eliminate SIP cash-flow contamination
        self.total_units = 0.0
        self.current_nav = 100.0
        self.daily_nav: Dict[pd.Timestamp, float] = {}

    def update_nav_before_inflow(self, prices: Dict[str, float]):
        """Calculates instantaneous marked-to-market equity before cash inflow and updates current_nav."""
        if self.total_units > 1e-9:
            self.current_nav = self.get_equity(prices) / self.total_units

    def inject_cash(self, date: pd.Timestamp, amount: float, prices: Optional[Dict[str, float]] = None):
        """Deposits capital into portfolio cash and issues units at current NAV."""
        if amount <= 0:
            return
        if self.total_units > 1e-9 and prices is not None:
            self.update_nav_before_inflow(prices)

        new_units = amount / self.current_nav
        self.total_units += new_units

        self.cash += amount
        self.cash_flows.append((date, -amount))
        self.total_invested += amount

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculates instantaneous marked-to-market equity."""
        equity = self.cash
        for sym, units in self.holdings.items():
            if units > 0 and sym in prices:
                equity += units * prices[sym]
        return equity

    def _recompute_avg_cost(self, sym: str):
        """Recomputes the derived avg_cost scalar from the lot book for symbol sym."""
        lots = self.lot_book.get(sym, [])
        total_units = sum(u for _, _, u in lots)
        if total_units > 1e-9:
            total_cost = sum(p * u for _, p, u in lots)
            self.avg_cost[sym] = total_cost / total_units
        else:
            self.avg_cost[sym] = 0.0

    def rebalance(self, date: pd.Timestamp, target_weights: Dict[str, float], prices: Dict[str, float]):
        """Executes a complete rebalance to target weights with realistic slippage, STT, and DP charges.

        Sell execution uses HIFO (Highest-In, First-Out) lot selection to minimise
        crystallised gains and STCG tax liability.
        """
        current_equity = self.get_equity(prices)
        if current_equity <= 0:
            return

        # 1. Determine sells first to raise cash
        symbols_to_check = set(list(self.holdings.keys()) + list(target_weights.keys()))
        sells_plan = []
        buys_plan = []

        # Differentiate between full exits (0.2% floor) and partial trims/top-ups (0.8% floor)
        MIN_EXIT_VALUE = current_equity * 0.002  # 0.2% floor for complete exits
        MIN_TRIM_VALUE = current_equity * 0.008  # 0.8% floor (< 5% macro deadband / 5 stocks = 1.0%)

        cash_deficit = max(0.0, -self.cash)

        for sym in symbols_to_check:
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue
            curr_units = self.holdings.get(sym, 0.0)
            curr_val = curr_units * px
            tgt_w = target_weights.get(sym, 0.0)
            tgt_val = tgt_w * current_equity

            diff = tgt_val - curr_val
            threshold = MIN_EXIT_VALUE if (tgt_w < 1e-5 or curr_units <= 1e-4) else MIN_TRIM_VALUE
            if diff < -threshold:
                sells_plan.append((sym, -diff, px, curr_units))
            elif diff > threshold:
                buys_plan.append((sym, diff, px))

        # Execute Sells — Statutory FIFO or HIFO lot selection
        for sym, sell_val_raw, px, curr_units in sells_plan:
            units_to_sell = min(curr_units, sell_val_raw / px)
            if units_to_sell <= 0:
                continue

            effective_px = px * (1.0 - self.config["SLIPPAGE_PER_SIDE"])
            turnover = units_to_sell * effective_px

            # --- Tax Lot Selection: FIFO (Statutory Indian Demat) vs HIFO ---
            sym_lots = self.lot_book.get(sym, [])
            tax_method = self.config.get("TAX_LOT_METHOD", "FIFO")
            if tax_method == "FIFO":
                # First-In, First-Out: sort ascending by purchase date
                sym_lots_sorted = sorted(sym_lots, key=lambda t: t[0])
            else:
                # Highest-In, First-Out: sort descending by purchase price
                sym_lots_sorted = sorted(sym_lots, key=lambda t: t[1], reverse=True)

            remaining_to_sell = units_to_sell
            realized_stcg = 0.0
            realized_ltcg = 0.0
            new_lots = []

            for lot_date, lot_price, lot_units in sym_lots_sorted:
                if remaining_to_sell <= 1e-9:
                    new_lots.append((lot_date, lot_price, lot_units))
                    continue
                consume = min(lot_units, remaining_to_sell)
                holding_days = (date - lot_date).days if isinstance(lot_date, pd.Timestamp) else 0
                lot_pnl = (effective_px - lot_price) * consume
                if holding_days > 365:
                    realized_ltcg += lot_pnl
                else:
                    realized_stcg += lot_pnl

                remaining_to_sell -= consume
                leftover = lot_units - consume
                if leftover > 1e-9:
                    new_lots.append((lot_date, lot_price, leftover))

            # Persist updated lot book
            self.lot_book[sym] = new_lots
            self._recompute_avg_cost(sym)

            frictions = self.cost_ledger.record_sell(
                turnover=turnover,
                realized_pnl=(realized_stcg + realized_ltcg),
                realized_stcg=realized_stcg,
                realized_ltcg=realized_ltcg
            )
            net_proceeds = turnover - (frictions["stt"] + frictions["dp_charge"])
            self.cash += net_proceeds

            remaining_units = sum(u for _, _, u in new_lots)
            if remaining_units <= 1e-4:
                self.holdings[sym] = 0.0
                self.avg_cost[sym] = 0.0
                self.lot_book[sym] = []
            else:
                self.holdings[sym] = remaining_units

        # If cash is still negative after planned sells, trim largest holdings to clear deficit
        if self.cash < 0:
            deficit_to_clear = -self.cash + 100.0
            sorted_holdings = sorted(
                [(s, u) for s, u in self.holdings.items() if u > 1e-4],
                key=lambda x: x[1] * prices.get(x[0], 0.0),
                reverse=True
            )
            for sym, curr_units in sorted_holdings:
                if self.cash >= 0:
                    break
                px = prices.get(sym, 0.0)
                if px <= 0:
                    continue
                units_needed = min(curr_units, (deficit_to_clear / px))
                if units_needed <= 0:
                    continue
                effective_px = px * (1.0 - self.config["SLIPPAGE_PER_SIDE"])
                turnover = units_needed * effective_px
                sym_lots = self.lot_book.get(sym, [])
                tax_method = self.config.get("TAX_LOT_METHOD", "FIFO")
                sym_lots_sorted = sorted(sym_lots, key=lambda t: t[0]) if tax_method == "FIFO" else sorted(sym_lots, key=lambda t: t[1], reverse=True)
                rem = units_needed
                realized_stcg, realized_ltcg = 0.0, 0.0
                new_lots = []
                for lot_date, lot_price, lot_units in sym_lots_sorted:
                    if rem <= 1e-9:
                        new_lots.append((lot_date, lot_price, lot_units))
                        continue
                    c = min(lot_units, rem)
                    h_days = (date - lot_date).days if isinstance(lot_date, pd.Timestamp) else 0
                    pnl = (effective_px - lot_price) * c
                    if h_days > 365:
                        realized_ltcg += pnl
                    else:
                        realized_stcg += pnl
                    rem -= c
                    leftover = lot_units - c
                    if leftover > 1e-9:
                        new_lots.append((lot_date, lot_price, leftover))
                self.lot_book[sym] = new_lots
                self._recompute_avg_cost(sym)
                frictions = self.cost_ledger.record_sell(
                    turnover=turnover,
                    realized_pnl=(realized_stcg + realized_ltcg),
                    realized_stcg=realized_stcg,
                    realized_ltcg=realized_ltcg
                )
                net_proceeds = turnover - (frictions["stt"] + frictions["dp_charge"])
                self.cash += net_proceeds
                deficit_to_clear -= net_proceeds
                self.holdings[sym] = sum(u for _, _, u in new_lots)

        # Execute Buys — add new lot entry per purchase
        total_buy_desired = sum(b[1] for b in buys_plan)
        if total_buy_desired > 0 and self.cash > 0:
            capital_scaler = min(1.0, (self.cash * 0.999) / total_buy_desired)
            for sym, buy_val_desired, px in buys_plan:
                alloc_cash = buy_val_desired * capital_scaler
                effective_px = px * (1.0 + self.config["SLIPPAGE_PER_SIDE"])
                unit_cost = effective_px * (1.0 + self.config["STT_RATE"])
                units_to_buy = alloc_cash / unit_cost
                if units_to_buy <= 0:
                    continue

                turnover = units_to_buy * effective_px
                frictions = self.cost_ledger.record_buy(turnover)
                total_outflow = turnover + frictions["stt"]

                self.cash -= total_outflow

                # Add new lot to lot book: (date, price, units)
                if sym not in self.lot_book:
                    self.lot_book[sym] = []
                self.lot_book[sym].append((date, effective_px, units_to_buy))

                # Update derived scalars
                self._recompute_avg_cost(sym)
                self.holdings[sym] = sum(u for _, _, u in self.lot_book[sym])

        # Clean zero holdings
        self.holdings = {k: v for k, v in self.holdings.items() if v > 1e-4}
        self.lot_book = {k: v for k, v in self.lot_book.items() if sum(u for _, _, u in v) > 1e-4}

    def deploy_idle_cash(self, date: pd.Timestamp, target_weights: Dict[str, float], prices: Dict[str, float]):
        """Deploys idle cash (e.g. monthly SIP inflows) via buy-only orders during deadband-skipped months.

        Because no sells are executed, this operation incurs zero CDSL DP debit charges
        and triggers zero STCG tax realization, keeping cash fully invested and compounding.
        """
        if self.cash <= 500.0 or not target_weights:
            return
        investable_cash = self.cash * 0.999
        for sym, weight in target_weights.items():
            if weight <= 0:
                continue
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue
            alloc_cash = investable_cash * weight
            effective_px = px * (1.0 + self.config["SLIPPAGE_PER_SIDE"])
            unit_cost = effective_px * (1.0 + self.config["STT_RATE"])
            units_to_buy = alloc_cash / unit_cost
            if units_to_buy <= 0:
                continue
            turnover = units_to_buy * effective_px
            frictions = self.cost_ledger.record_buy(turnover)
            self.cash -= (turnover + frictions["stt"])
            if sym not in self.lot_book:
                self.lot_book[sym] = []
            self.lot_book[sym].append((date, effective_px, units_to_buy))
            self._recompute_avg_cost(sym)
            self.holdings[sym] = sum(u for _, _, u in self.lot_book[sym])

    def swap_dropped_stocks(
        self,
        date: pd.Timestamp,
        dropped_stocks: List[str],
        entering_stocks: List[str],
        prices: Dict[str, float]
    ):
        """Surgically exits only momentum stocks that fell out of Top-8 keep_rank and
        reinvests ONLY their net sale proceeds into the newly entering Top-5 leaders.
        Any remaining idle cash (the monthly SIP inflow) is then deployed proportionally
        across all holdings so neither single-stock weights nor macro weights are distorted.
        """
        swap_proceeds = 0.0

        # 1. Sell 100% of dropped stocks using statutory FIFO / HIFO lot accounting
        for sym in dropped_stocks:
            curr_units = self.holdings.get(sym, 0.0)
            px = prices.get(sym, 0.0)
            if curr_units <= 1e-4 or px <= 0:
                continue

            effective_px = px * (1.0 - self.config["SLIPPAGE_PER_SIDE"])
            turnover = curr_units * effective_px

            sym_lots = self.lot_book.get(sym, [])
            tax_method = self.config.get("TAX_LOT_METHOD", "FIFO")
            if tax_method == "FIFO":
                sym_lots_sorted = sorted(sym_lots, key=lambda t: t[0])
            else:
                sym_lots_sorted = sorted(sym_lots, key=lambda t: t[1], reverse=True)

            realized_stcg = 0.0
            realized_ltcg = 0.0
            for lot_date, lot_price, lot_units in sym_lots_sorted:
                holding_days = (date - lot_date).days if isinstance(lot_date, pd.Timestamp) else 0
                lot_pnl = (effective_px - lot_price) * lot_units
                if holding_days > 365:
                    realized_ltcg += lot_pnl
                else:
                    realized_stcg += lot_pnl

            self.lot_book[sym] = []
            self.holdings[sym] = 0.0
            self.avg_cost[sym] = 0.0

            frictions = self.cost_ledger.record_sell(
                turnover=turnover,
                realized_pnl=(realized_stcg + realized_ltcg),
                realized_stcg=realized_stcg,
                realized_ltcg=realized_ltcg
            )
            net_proceeds = turnover - (frictions["stt"] + frictions["dp_charge"])
            self.cash += net_proceeds
            swap_proceeds += net_proceeds

        # Clean zero holdings
        self.holdings = {k: v for k, v in self.holdings.items() if v > 1e-4}
        self.lot_book = {k: v for k, v in self.lot_book.items() if sum(u for _, _, u in v) > 1e-4}

        # 2. Reinvest ONLY swap_proceeds equally into entering_stocks (strictly dollar-neutral swap)
        if swap_proceeds > 500.0 and entering_stocks:
            alloc_per_stock = (swap_proceeds * 0.999) / len(entering_stocks)
            for sym in entering_stocks:
                px = prices.get(sym, 0.0)
                if px <= 0:
                    continue
                effective_px = px * (1.0 + self.config["SLIPPAGE_PER_SIDE"])
                unit_cost = effective_px * (1.0 + self.config["STT_RATE"])
                units_to_buy = alloc_per_stock / unit_cost
                if units_to_buy <= 0:
                    continue
                turnover = units_to_buy * effective_px
                frictions = self.cost_ledger.record_buy(turnover)
                self.cash -= (turnover + frictions["stt"])
                if sym not in self.lot_book:
                    self.lot_book[sym] = []
                self.lot_book[sym].append((date, effective_px, units_to_buy))
                self._recompute_avg_cost(sym)
                self.holdings[sym] = sum(u for _, _, u in self.lot_book[sym])

        # 3. Deploy remaining idle cash (the monthly SIP) proportionally across all current holdings
        if self.cash > 500.0 and self.holdings:
            tot_h = sum(u * prices.get(s, 0.0) for s, u in self.holdings.items() if u > 1e-4)
            if tot_h > 0:
                chw = {s: (u * prices.get(s, 0.0)) / tot_h for s, u in self.holdings.items() if u > 1e-4}
                self.deploy_idle_cash(date, chw, prices)

    def process_year_end_tax(self):
        """Deducts STCG tax liability from cash at year end."""
        tax = self.cost_ledger.process_financial_year_tax()
        if tax > 0:
            self.cash -= tax


class BacktestEngine:
    """Coordinates multi-strategy backtest simulation from 2018 to present."""

    def __init__(
        self,
        benchmark_series: pd.Series,
        etf_df: pd.DataFrame,
        stocks_df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        pit_universe_df: Optional[pd.DataFrame] = None,
        config: Dict = CONFIG
    ):
        self.config = config
        self.benchmark = benchmark_series
        self.etf_df = etf_df
        self.stocks_df = stocks_df
        self.vix_series = vix_series
        self.pit_universe_df = pit_universe_df

        # Compute benchmark ATR for Dynamic Spatial Bands:
        # TR_t = max(H_t - L_t, |H_t - P_{t-1}|, |L_t - P_{t-1}|)
        # Using benchmark close volatility as robust proxy or direct daily range
        ret_daily = self.benchmark.pct_change().abs()
        self.atr_series = (ret_daily * self.benchmark).rolling(window=20, min_periods=5).mean()

        # Regime Filters
        self.filter_binary = RegimeFilter(
            self.benchmark,
            vix_series=self.vix_series,
            sma_window=config["TREND_SMA_DAYS"],
            mode="BINARY"
        )
        # Previous hardcoded VIX approach (VIX >= 24.0, 1.35x 20-SMA)
        self.filter_static_vix = RegimeFilter(
            self.benchmark,
            vix_series=self.vix_series,
            sma_window=config["TREND_SMA_DAYS"],
            mode="HYSTERESIS_VIX",
            buffer_pct=config.get("BUFFER_PCT", 0.015),
            vix_abs_threshold=24.0,
            vix_sma_window=20,
            vix_rel_multiplier=1.35
        )
        # New dynamic rolling 252-day 95th percentile VIX approach with Static Universe
        self.filter_dyn_vix_static = RegimeFilter(
            self.benchmark,
            vix_series=self.vix_series,
            sma_window=config["TREND_SMA_DAYS"],
            mode="HYSTERESIS_VIX",
            buffer_pct=config.get("BUFFER_PCT", 0.015),
            vix_abs_threshold=None,
            vix_percentile_window=config.get("VIX_PERCENTILE_WINDOW", 252),
            vix_percentile_threshold=config.get("VIX_PERCENTILE", 0.95),
            vix_sma_window=config.get("VIX_SMA_WINDOW", 20),
            vix_rel_multiplier=config.get("VIX_REL_MULTIPLIER", None),
            use_dynamic_atr_band=True,
            atr_series=self.atr_series,
            atr_multiplier=1.5,
            confirmation_days=3
        )
        # New dynamic rolling 252-day 95th percentile VIX approach with Point-in-Time Universe
        self.filter_dyn_vix_pit = RegimeFilter(
            self.benchmark,
            vix_series=self.vix_series,
            sma_window=config["TREND_SMA_DAYS"],
            mode="HYSTERESIS_VIX",
            buffer_pct=config.get("BUFFER_PCT", 0.015),
            vix_abs_threshold=None,
            vix_percentile_window=config.get("VIX_PERCENTILE_WINDOW", 252),
            vix_percentile_threshold=config.get("VIX_PERCENTILE", 0.95),
            vix_sma_window=config.get("VIX_SMA_WINDOW", 20),
            vix_rel_multiplier=config.get("VIX_REL_MULTIPLIER", None),
            use_dynamic_atr_band=True,
            atr_series=self.atr_series,
            atr_multiplier=1.5,
            confirmation_days=3
        )
        self.filter_sigmoid = RegimeFilter(
            self.benchmark,
            vix_series=self.vix_series,
            sma_window=config["TREND_SMA_DAYS"],
            mode="CONTINUOUS_SIGMOID",
            sigmoid_k=config.get("SIGMOID_K", 35.0),
            vix_abs_threshold=None,
            vix_percentile_window=config.get("VIX_PERCENTILE_WINDOW", 252),
            vix_percentile_threshold=config.get("VIX_PERCENTILE", 0.95),
            vix_sma_window=config.get("VIX_SMA_WINDOW", 20),
            vix_rel_multiplier=config.get("VIX_REL_MULTIPLIER", None),
            use_adaptive_sigmoid_k=True,
            adaptive_k_constant=config.get("ADAPTIVE_K_CONSTANT", 5.25),
            sigmoid_d0=config.get("SIGMOID_D0", -0.010),
            upper_snap_dist=config.get("UPPER_SNAP_DIST", 0.020),
            lower_snap_dist=config.get("LOWER_SNAP_DIST", -0.045)
        )

        # Asset Class Models
        self.risk_parity_model = RiskParityModel(
            self.etf_df,
            lookback_days=config["VOL_LOOKBACK"],
            min_weight=config["MIN_WEIGHT"],
            max_weight=config["MAX_WEIGHT"]
        )
        # Dual Momentum Models: Static vs. Point-in-Time (PIT)
        self.momentum_static = MomentumModel(
            self.stocks_df[config["LIQUID_UNIVERSE"]],
            lookback_long=config["MOMENTUM_LOOKBACK_LONG"],
            skip_short=config["MOMENTUM_SKIP_SHORT"],
            top_n=config["MOMENTUM_TOP_N"],
            mode=config["MOMENTUM_MODE"],
            etf_ticker=config["MOMENTUM_ETF"],
            pit_universe_df=None
        )
        self.momentum_pit = MomentumModel(
            self.stocks_df,
            lookback_long=config["MOMENTUM_LOOKBACK_LONG"],
            skip_short=config["MOMENTUM_SKIP_SHORT"],
            top_n=config["MOMENTUM_TOP_N"],
            mode=config["MOMENTUM_MODE"],
            etf_ticker=config["MOMENTUM_ETF"],
            pit_universe_df=self.pit_universe_df
        )

        # Accounts
        self.acc_binary = PortfolioAccount(config, name="Basic Binary 200 SMA")
        self.acc_static_vix = PortfolioAccount(config, name="Hardcoded VIX >= 24")
        self.acc_dyn_vix_static = PortfolioAccount(config, name="Model 1: Static Universe (95th-Pct VIX)")
        self.acc_dyn_vix_pit = PortfolioAccount(config, name="Model 2: PIT Discrete (95th-Pct VIX + ATR)")
        self.acc_flagship = PortfolioAccount(config, name="Flagship Combined (PIT + Adaptive Sigmoid + 95th-Pct VIX)")
        self.acc_sigmoid = self.acc_flagship  # Backward-compatible alias
        self.acc_nifty = PortfolioAccount(config, name="Nifty 50 Buy & Hold")

        self.regime_binary_hist: Dict[pd.Timestamp, str] = {}
        self.regime_static_hist: Dict[pd.Timestamp, str] = {}
        self.regime_dyn_hist: Dict[pd.Timestamp, str] = {}
        self.regime_sigmoid_hist: Dict[pd.Timestamp, float] = {}

        # Logged table of every trade executed under PIT that involved an exited historical constituent
        self.distressed_tickers = set(config.get("DISTRESSED_HISTORICAL_CONSTITUENTS", []))
        self.historical_exited_trades: List[Dict] = []

    def _blend_weights(
        self,
        mom_weights: Dict[str, float],
        rp_weights: Dict[str, float],
        w_mom: float,
        w_rp: float
    ) -> Dict[str, float]:
        """Blends Momentum and Risk Parity weights continuously."""
        all_syms = set(list(mom_weights.keys()) + list(rp_weights.keys()))
        blended = {}
        for s in all_syms:
            wt = (mom_weights.get(s, 0.0) * w_mom) + (rp_weights.get(s, 0.0) * w_rp)
            if wt > 1e-5:
                blended[s] = wt
        # Normalize sum to 1.0
        tot = sum(blended.values())
        if tot > 0:
            blended = {k: v / tot for k, v in blended.items()}
        return blended

    def run(self) -> Dict:
        """Executes the monthly multi-model backtest."""
        start_date = pd.Timestamp(self.config["START_DATE"])
        end_date = pd.Timestamp(self.config["END_DATE"])

        sim_dates = [d for d in self.benchmark.index if start_date <= d <= end_date]
        if not sim_dates:
            raise ValueError("No trading dates found in specified date window!")

        first_date = sim_dates[0]
        monthly_dates = []
        date_series = pd.Series(sim_dates, index=sim_dates)
        for _, group in date_series.groupby([date_series.index.year, date_series.index.month]):
            monthly_dates.append(group.iloc[-1])
        rebalance_dates = sorted(list(set(monthly_dates) | {first_date}))

        print(f"[BacktestEngine] Simulating 6 strategies across {len(sim_dates)} daily bars and {len(rebalance_dates)} monthly rebalances...")

        # Indian Financial Year tracking (April 1 to March 31)
        prev_fy = first_date.year if first_date.month >= 4 else first_date.year - 1
        nifty_etf = self.config["RISK_PARITY_TICKERS"]["EQUITY"]

        for dt in sim_dates:
            # Indian Fiscal Year transition tax handling (sweep on March 31 / April 1 transition)
            curr_fy = dt.year if dt.month >= 4 else dt.year - 1
            if curr_fy != prev_fy:
                self.acc_binary.process_year_end_tax()
                self.acc_static_vix.process_year_end_tax()
                self.acc_dyn_vix_static.process_year_end_tax()
                self.acc_dyn_vix_pit.process_year_end_tax()
                self.acc_flagship.process_year_end_tax()
                self.acc_nifty.process_year_end_tax()
                prev_fy = curr_fy

            # Market prices snapshot
            prices_snap = {}
            for col in self.etf_df.columns:
                prices_snap[col] = float(self.etf_df[col].asof(dt))
            for col in self.stocks_df.columns:
                prices_snap[col] = float(self.stocks_df[col].asof(dt))
            prices_snap["^NSEI"] = float(self.benchmark.asof(dt))

            if dt in rebalance_dates:
                is_first_day = (dt == first_date)
                inflow = self.config["INITIAL_CAPITAL"] if is_first_day else self.config["SIP_AMOUNT"]

                # 1. Cash Inflow injection with prices passed for unitized NAV calculation
                if inflow > 0:
                    self.acc_binary.inject_cash(dt, inflow, prices=prices_snap)
                    self.acc_static_vix.inject_cash(dt, inflow, prices=prices_snap)
                    self.acc_dyn_vix_static.inject_cash(dt, inflow, prices=prices_snap)
                    self.acc_dyn_vix_pit.inject_cash(dt, inflow, prices=prices_snap)
                    self.acc_flagship.inject_cash(dt, inflow, prices=prices_snap)
                    self.acc_nifty.inject_cash(dt, inflow, prices=prices_snap)

                # 2. Evaluate Strategy Models
                # A. Binary 200 SMA
                reg_bin, w_mom_b, w_rp_b, _, _ = self.filter_binary.check_and_log_transition(dt)
                self.regime_binary_hist[dt] = reg_bin

                # B. Static VIX (>= 24.0)
                reg_sta, w_mom_sta, w_rp_sta, _, info_sta = self.filter_static_vix.check_and_log_transition(dt)
                self.regime_static_hist[dt] = reg_sta

                # C. Dynamic 95th-Pct VIX with ATR bands and 3-day confirmation
                reg_dyn_s, w_mom_ds, w_rp_ds, _, info_ds = self.filter_dyn_vix_static.check_and_log_transition(dt)
                reg_dyn_p, w_mom_dp, w_rp_dp, _, info_dp = self.filter_dyn_vix_pit.check_and_log_transition(dt)
                self.regime_dyn_hist[dt] = reg_dyn_p

                # D. Continuous Sigmoid Sizing (with adaptive k_t = C / sigma_t)
                _, w_mom_s, w_rp_s, _, _ = self.filter_sigmoid.check_and_log_transition(dt)
                self.regime_sigmoid_hist[dt] = w_mom_s

                # 3. Component Weights
                rp_etf_tickers = set(self.config["RISK_PARITY_TICKERS"].values())
                rp_weights = self.risk_parity_model.calculate_weights_at(dt)

                # Independent current equity holdings per account (prevents cross-account contamination)
                static_curr_stocks = [s for s, u in self.acc_dyn_vix_static.holdings.items() if s not in rp_etf_tickers and u > 1e-4]
                pit_curr_stocks = [s for s, u in self.acc_dyn_vix_pit.holdings.items() if s not in rp_etf_tickers and u > 1e-4]
                flagship_curr_stocks = [s for s, u in self.acc_flagship.holdings.items() if s not in rp_etf_tickers and u > 1e-4]

                mom_weights_static_raw, _ = self.momentum_static.calculate_weights_at(dt, current_holdings=None)
                mom_weights_static_keep, _ = self.momentum_static.calculate_weights_at(dt, current_holdings=static_curr_stocks)
                mom_weights_pit_model2, leaders_pit_m2 = self.momentum_pit.calculate_weights_at(dt, current_holdings=pit_curr_stocks)
                mom_weights_pit, leaders_pit = self.momentum_pit.calculate_weights_at(dt, current_holdings=flagship_curr_stocks)

                target_binary = mom_weights_static_raw if reg_bin == "BULL" else rp_weights
                target_static_vix = mom_weights_static_raw if reg_sta == "BULL" else rp_weights
                target_dyn_static = mom_weights_static_keep if reg_dyn_s == "BULL" else rp_weights
                target_dyn_pit = mom_weights_pit_model2 if reg_dyn_p == "BULL" else rp_weights

                # --- Deadband Check for Flagship Sigmoid Allocation ---
                dropped_stocks = [s for s in flagship_curr_stocks if s not in mom_weights_pit]
                entering_stocks = [s for s in mom_weights_pit if s not in flagship_curr_stocks]

                flagship_invested_val = sum(
                    units * prices_snap.get(sym, 0.0)
                    for sym, units in self.acc_flagship.holdings.items()
                    if units > 1e-4
                )
                if flagship_invested_val > 0:
                    flagship_mom_value = sum(
                        units * prices_snap.get(sym, 0.0)
                        for sym, units in self.acc_flagship.holdings.items()
                        if sym not in rp_etf_tickers and units > 1e-4
                    )
                    current_w_mom_flagship = flagship_mom_value / flagship_invested_val
                else:
                    current_w_mom_flagship = 0.0

                DEADBAND_THRESHOLD = 0.05
                if flagship_invested_val == 0 or abs(w_mom_s - current_w_mom_flagship) >= DEADBAND_THRESHOLD:
                    target_sigmoid = self._blend_weights(mom_weights_pit, rp_weights, w_mom_s, w_rp_s)
                else:
                    target_sigmoid = None

                # Audit check for Model 2 PIT:
                if reg_dyn_p == "BULL":
                    for sym, wt in target_dyn_pit.items():
                        if sym in self.distressed_tickers and wt > 0:
                            px = prices_snap.get(sym, 0.0)
                            mom_score = dict(leaders_pit_m2).get(sym, 0.0)
                            self.historical_exited_trades.append({
                                "date": dt,
                                "ticker": sym,
                                "action": "HOLD/BUY",
                                "weight": wt,
                                "price": px,
                                "mom_12_1_score": mom_score,
                                "note": "Historical constituent selected under PIT"
                            })

                # 4. Rebalance Portfolios
                self.acc_binary.rebalance(dt, target_binary, prices_snap)
                self.acc_static_vix.rebalance(dt, target_static_vix, prices_snap)
                self.acc_dyn_vix_static.rebalance(dt, target_dyn_static, prices_snap)
                self.acc_dyn_vix_pit.rebalance(dt, target_dyn_pit, prices_snap)

                if target_sigmoid is not None:
                    self.acc_flagship.rebalance(dt, target_sigmoid, prices_snap)
                elif len(dropped_stocks) > 0 and w_mom_s >= 0.95 and current_w_mom_flagship >= 0.90:
                    self.acc_flagship.swap_dropped_stocks(dt, dropped_stocks, entering_stocks, prices_snap)
                else:
                    if self.acc_flagship.holdings:
                        tot_h = sum(u * prices_snap.get(s, 0.0) for s, u in self.acc_flagship.holdings.items() if u > 1e-4)
                        if tot_h > 0:
                            chw = {s: (u * prices_snap.get(s, 0.0)) / tot_h for s, u in self.acc_flagship.holdings.items() if u > 1e-4}
                            self.acc_flagship.deploy_idle_cash(dt, chw, prices_snap)
                    else:
                        fallback_deploy_weights = mom_weights_pit if w_mom_s >= 0.5 else rp_weights
                        self.acc_flagship.deploy_idle_cash(dt, fallback_deploy_weights, prices_snap)

                self.acc_nifty.rebalance(dt, {nifty_etf: 1.0}, prices_snap)

            else:
                self.regime_binary_hist[dt] = self.regime_binary_hist.get(rebalance_dates[0], "BULL")
                self.regime_static_hist[dt] = self.regime_static_hist.get(rebalance_dates[0], "BULL")
                self.regime_dyn_hist[dt] = self.regime_dyn_hist.get(rebalance_dates[0], "BULL")
                self.regime_sigmoid_hist[dt] = self.regime_sigmoid_hist.get(rebalance_dates[0], 1.0)

            # Record Daily Equity and Unitized Time-Weighted NAV
            for acc in [
                self.acc_binary, self.acc_static_vix, self.acc_dyn_vix_static,
                self.acc_dyn_vix_pit, self.acc_flagship, self.acc_nifty
            ]:
                eq = acc.get_equity(prices_snap)
                acc.daily_equity[dt] = eq
                acc.daily_nav[dt] = (eq / acc.total_units) if acc.total_units > 1e-9 else 100.0

        # Final fiscal year tax sweep
        self.acc_binary.process_year_end_tax()
        self.acc_static_vix.process_year_end_tax()
        self.acc_dyn_vix_static.process_year_end_tax()
        self.acc_dyn_vix_pit.process_year_end_tax()
        self.acc_flagship.process_year_end_tax()
        self.acc_nifty.process_year_end_tax()

        return {
            "binary_series": pd.Series(self.acc_binary.daily_equity),
            "static_vix_series": pd.Series(self.acc_static_vix.daily_equity),
            "dyn_vix_static_series": pd.Series(self.acc_dyn_vix_static.daily_equity),
            "dyn_vix_pit_series": pd.Series(self.acc_dyn_vix_pit.daily_equity),
            "flagship_series": pd.Series(self.acc_flagship.daily_equity),
            "dyn_vix_series": pd.Series(self.acc_dyn_vix_pit.daily_equity), # default alias
            "hys_vix_series": pd.Series(self.acc_dyn_vix_pit.daily_equity), # visualizer alias
            "sigmoid_series": pd.Series(self.acc_flagship.daily_equity),
            "nifty_series": pd.Series(self.acc_nifty.daily_equity),
            "binary_nav_series": pd.Series(self.acc_binary.daily_nav),
            "static_vix_nav_series": pd.Series(self.acc_static_vix.daily_nav),
            "dyn_vix_static_nav_series": pd.Series(self.acc_dyn_vix_static.daily_nav),
            "dyn_vix_pit_nav_series": pd.Series(self.acc_dyn_vix_pit.daily_nav),
            "flagship_nav_series": pd.Series(self.acc_flagship.daily_nav),
            "dyn_vix_nav_series": pd.Series(self.acc_dyn_vix_pit.daily_nav),
            "hys_vix_nav_series": pd.Series(self.acc_dyn_vix_pit.daily_nav),
            "sigmoid_nav_series": pd.Series(self.acc_flagship.daily_nav),
            "nifty_nav_series": pd.Series(self.acc_nifty.daily_nav),
            "regime_binary_series": pd.Series(self.regime_binary_hist).reindex(sim_dates).ffill(),
            "regime_static_series": pd.Series(self.regime_static_hist).reindex(sim_dates).ffill(),
            "regime_hys_series": pd.Series(self.regime_dyn_hist).reindex(sim_dates).ffill(),
            "regime_sigmoid_series": pd.Series(self.regime_sigmoid_hist).reindex(sim_dates).ffill(),
            "acc_binary": self.acc_binary,
            "acc_static_vix": self.acc_static_vix,
            "acc_dyn_vix_static": self.acc_dyn_vix_static,
            "acc_dyn_vix_pit": self.acc_dyn_vix_pit,
            "acc_flagship": self.acc_flagship,
            "acc_hys_vix": self.acc_dyn_vix_pit,
            "acc_dyn_vix": self.acc_dyn_vix_pit,
            "acc_sigmoid": self.acc_flagship,
            "acc_nifty": self.acc_nifty,
            "transitions_binary": self.filter_binary.get_transition_log(),
            "transitions_static": self.filter_static_vix.get_transition_log(),
            "transitions_hys": self.filter_dyn_vix_pit.get_transition_log(),
            "historical_exited_trades": self.historical_exited_trades,
            "sim_dates": sim_dates,
        }
