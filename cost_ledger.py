"""Cost and Friction Ledger for Indian Equity & ETF Delivery Execution."""

from typing import Dict


class CostLedger:
    """Tracks brokerage, DP charges, STT, slippage, and statutory STCG / LTCG tax liabilities."""

    def __init__(self, config: Dict):
        self.stt_rate = config.get("STT_RATE", 0.001)
        self.dp_charge_per_sell = config.get("DP_CHARGE", 15.34)
        self.slippage_rate = config.get("SLIPPAGE_PER_SIDE", 0.0005)
        self.stcg_tax_rate = config.get("TAX_STCG_RATE", 0.20)
        self.ltcg_tax_rate = config.get("TAX_LTCG_RATE", 0.125)
        self.ltcg_exemption_limit = config.get("LTCG_EXEMPTION_LIMIT", 125000.0)

        # Cumulative totals
        self.total_stt = 0.0
        self.total_dp_charges = 0.0
        self.total_slippage = 0.0
        self.total_stcg_tax = 0.0
        self.total_ltcg_tax = 0.0
        self.total_turnover = 0.0
        self.total_trades_count = 0

        # Annual tracking for current Indian Financial Year (April 1 - March 31)
        self.current_fy_stcg_gains = 0.0
        self.current_fy_ltcg_gains = 0.0
        self.stcl_carried_forward = 0.0
        self.ltcl_carried_forward = 0.0

    def record_sell(
        self,
        turnover: float,
        realized_pnl: float = 0.0,
        realized_stcg: float = 0.0,
        realized_ltcg: float = 0.0
    ) -> Dict[str, float]:
        """Calculates and records frictions for a sell order, segregating STCG and LTCG."""
        slippage = turnover * self.slippage_rate
        stt = turnover * self.stt_rate
        dp = self.dp_charge_per_sell
        total_drag = slippage + stt + dp

        self.total_slippage += slippage
        self.total_stt += stt
        self.total_dp_charges += dp
        self.total_turnover += turnover
        self.total_trades_count += 1

        if abs(realized_stcg) > 1e-9 or abs(realized_ltcg) > 1e-9:
            tot_gain_mag = abs(realized_stcg) + abs(realized_ltcg)
            stcg_drag = total_drag * (abs(realized_stcg) / tot_gain_mag)
            ltcg_drag = total_drag - stcg_drag
            self.current_fy_stcg_gains += (realized_stcg - stcg_drag)
            self.current_fy_ltcg_gains += (realized_ltcg - ltcg_drag)
        else:
            self.current_fy_stcg_gains += (realized_pnl - total_drag)

        return {
            "slippage": slippage,
            "stt": stt,
            "dp_charge": dp,
            "total_frictions": total_drag
        }

    def record_buy(self, turnover: float) -> Dict[str, float]:
        """Calculates and records frictions for a buy order."""
        slippage = turnover * self.slippage_rate
        stt = turnover * self.stt_rate
        total_drag = slippage + stt

        self.total_slippage += slippage
        self.total_stt += stt
        self.total_turnover += turnover
        self.total_trades_count += 1

        return {
            "slippage": slippage,
            "stt": stt,
            "dp_charge": 0.0,
            "total_frictions": total_drag
        }

    def process_financial_year_tax(self) -> float:
        """Computes STCG & LTCG taxes at March 31 under Sections 111A, 112A, and 74.

        Statutory Rules Applied:
        1. STCL carried forward offsets STCG, and any excess STCL can offset LTCG.
        2. LTCL carried forward can only offset LTCG.
        3. Remaining positive STCG is taxed at 20.0% (Section 111A).
        4. Remaining positive LTCG receives ₹1,25,000 annual exemption; excess taxed at 12.5% (Section 112A).
        5. Unabsorbed losses are carried forward to future fiscal years under Section 74.
        """
        # 1. Net against prior carried-forward losses
        net_stcg = self.current_fy_stcg_gains - self.stcl_carried_forward
        net_ltcg = self.current_fy_ltcg_gains - self.ltcl_carried_forward

        # 2. Section 74 cross-offset: STCL can offset LTCG
        if net_stcg < 0 and net_ltcg > 0:
            offset = min(abs(net_stcg), net_ltcg)
            net_ltcg -= offset
            net_stcg += offset

        total_tax = 0.0

        # 3. STCG Tax Calculation
        if net_stcg > 0:
            stcg_tax = net_stcg * self.stcg_tax_rate
            total_tax += stcg_tax
            self.total_stcg_tax += stcg_tax
            self.stcl_carried_forward = 0.0
        else:
            self.stcl_carried_forward = abs(net_stcg)

        # 4. LTCG Tax Calculation (with ₹1,25,000 annual exemption limit)
        if net_ltcg > 0:
            taxable_ltcg = max(0.0, net_ltcg - self.ltcg_exemption_limit)
            ltcg_tax = taxable_ltcg * self.ltcg_tax_rate
            total_tax += ltcg_tax
            self.total_ltcg_tax += ltcg_tax
            self.ltcl_carried_forward = 0.0
        else:
            self.ltcl_carried_forward = abs(net_ltcg)

        # Reset annual accumulators for next fiscal year
        self.current_fy_stcg_gains = 0.0
        self.current_fy_ltcg_gains = 0.0

        return total_tax

    def get_friction_summary(self) -> Dict[str, float]:
        """Returns comprehensive breakdown of all incurred frictions and taxes."""
        return {
            "total_stt": self.total_stt,
            "total_dp_charges": self.total_dp_charges,
            "total_slippage": self.total_slippage,
            "total_stcg_tax": self.total_stcg_tax,
            "total_ltcg_tax": self.total_ltcg_tax,
            "total_frictions": (
                self.total_stt + self.total_dp_charges + self.total_slippage +
                self.total_stcg_tax + self.total_ltcg_tax
            ),
            "total_turnover": self.total_turnover,
            "trades_count": self.total_trades_count,
        }

