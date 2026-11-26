"""Configuration module for Low-Frequency Regime-Switching Strategy."""

CONFIG = {
    # --- Capital & Inflow Parameters ---
    "INITIAL_CAPITAL": 100000.0,      # ₹1,00,000 initial capital
    "SIP_AMOUNT": 10000.0,            # ₹10,000 added every month

    # --- Realistic Frictions & Costs (Indian Market / Zerodha Delivery) ---
    "STT_RATE": 0.001,                # 0.1% STT on turnover
    "DP_CHARGE": 15.34,               # Flat ₹15.34 per scrip sold (CDSL DP charge incl. 18% GST)
    "SLIPPAGE_PER_SIDE": 0.0005,      # 0.05% per leg (bid-ask spread / market impact)
    "TAX_STCG_RATE": 0.20,            # 20% on realized short-term capital gains (Section 111A)
    "TAX_LTCG_RATE": 0.125,           # 12.5% on realized long-term capital gains > 365 days (Section 112A)
    "LTCG_EXEMPTION_LIMIT": 125000.0, # ₹1,25,000 annual LTCG exemption limit
    "TAX_LOT_METHOD": "FIFO",         # Statutory default "FIFO" per Section 45(2A); "HIFO" also supported

    # --- Benchmark & Risk Metrics ---
    "RISK_FREE_RATE": 0.065,          # 6.5% Indian 10Y G-Sec / MIBOR rate for Sharpe ratio
    "BENCHMARK_TICKER": "^NSEI",      # Nifty 50 Index

    # --- Date Boundaries ---
    "START_DATE": "2018-01-01",
    "END_DATE": "2023-12-31",
    "DATA_WARMUP_START": "2017-01-01", # Warmup period for 200 SMA & 12-1m momentum lookback

    # --- Regime Filter Parameters ---
    "TREND_SMA_DAYS": 200,            # 200-day Simple Moving Average on Nifty 50
    "BUFFER_PCT": 0.015,              # 1.5% Hysteresis Buffer Band around 200 SMA
    "VIX_TICKER": "^INDIAVIX",        # India VIX ticker for volatility circuit breaker
    "VIX_PERCENTILE_WINDOW": 252,     # Rolling 1 calendar year (252 trading days) empirical window
    "VIX_PERCENTILE": 0.95,           # 95th empirical percentile threshold (Order Statistic index k=240)
    "VIX_SMA_WINDOW": 20,             # Optional 20-day SMA window
    "VIX_REL_MULTIPLIER": None,       # Set to None for pure dynamic 95th percentile Shock_t
    "SIGMOID_K": 35.0,                # Slope parameter for continuous allocation sizing
    "SIGMOID_D0": -0.010,             # -1.0% trend distance offset
    "UPPER_SNAP_DIST": 0.020,         # +2.0% upper snap to 100% Momentum
    "LOWER_SNAP_DIST": -0.045,        # -4.5% lower snap to 100% Risk Parity

    # --- Cross-Asset Risk Parity Basket (Bear Mode) ---
    "RISK_PARITY_TICKERS": {
        "EQUITY": "NIFTYBEES.NS",
        "GOLD": "GOLDBEES.NS",
        "BOND": "LTGILTBEES.NS",
        "CASH": "LIQUIDBEES.NS",
    },
    "VOL_LOOKBACK": 60,               # 60-day rolling lookback window for realized volatility
    "MIN_WEIGHT": 0.10,               # 10% minimum weight floor per asset class
    "MAX_WEIGHT": 0.50,               # 50% maximum weight ceiling per asset class

    # --- Momentum Component (Bull Mode) ---
    "MOMENTUM_MODE": "STOCK_BASKET",  # 'STOCK_BASKET' or 'ETF_PROXY'
    "MOMENTUM_ETF": "NIFTYBEES.NS",   # ETF proxy fallback if MOMENTUM_MODE == 'ETF_PROXY'
    "MOMENTUM_TOP_N": 5,              # Number of momentum leaders to hold in Bull mode
    "MOMENTUM_LOOKBACK_LONG": 252,    # 12-month lookback (approx 252 trading days)
    "MOMENTUM_SKIP_SHORT": 21,        # Skip most recent 1-month reversal (approx 21 trading days)

    # Liquid Large-Cap Universe for Momentum Stock Basket (NSE) - Static Baseline
    "LIQUID_UNIVERSE": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS",
        "BHARTIARTL.NS", "LT.NS", "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "SBIN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "SUNPHARMA.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
        "HCLTECH.NS", "ULTRACEMCO.NS", "WIPRO.NS", "COALINDIA.NS", "JSWSTEEL.NS",
        "NESTLEIND.NS", "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "CIPLA.NS"
    ],

    # --- Dynamic Point-in-Time (PIT) Reconstitution Schedule (NSE Nifty 50) ---
    # Format: (Effective Date, Tickers Added, Tickers Removed)
    # Ticker symbol renames mapped for Yahoo Finance continuity:
    # IBULHSGFIN -> SAMMAANCAP.NS, BHARTIINFRATEL -> INDUSTOWER.NS, HPCL -> HINDPETRO.NS
    "NIFTY50_CHANGES": [
        ("2018-09-28", ["JSWSTEEL.NS"], ["LUPIN.NS"]),
        ("2019-03-29", ["BRITANNIA.NS"], ["HINDPETRO.NS"]),
        ("2019-09-27", ["NESTLEIND.NS"], ["SAMMAANCAP.NS"]),
        ("2020-03-27", ["SHREECEM.NS"], ["YESBANK.NS"]),
        ("2020-09-25", ["DIVISLAB.NS", "SBILIFE.NS"], ["INDUSTOWER.NS", "ZEEL.NS"]),
        ("2021-03-31", ["TATACONSUM.NS"], ["GAIL.NS"]),
        ("2021-09-30", ["APOLLOHOSP.NS"], ["IOC.NS"]),
        ("2022-09-30", ["ADANIENT.NS"], ["SHREECEM.NS"]),
        ("2023-03-31", ["LTIM.NS"], ["BPCL.NS"]),
        ("2024-03-28", ["SHRIRAMFIN.NS"], ["UPL.NS"]),
        ("2024-09-30", ["BEL.NS", "TRENT.NS"], ["DIVISLAB.NS"]),
    ],

    # Base Active Constituents as of 2017-01-01 (incorporating historical and distressed constituents)
    "NIFTY50_BASE_2017": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "HDFCBANK.NS",
        "BHARTIARTL.NS", "LT.NS", "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "SBIN.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
        "SUNPHARMA.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
        "HCLTECH.NS", "ULTRACEMCO.NS", "WIPRO.NS", "COALINDIA.NS", "GRASIM.NS",
        "TECHM.NS", "BAJAJFINSV.NS", "CIPLA.NS",
        # Historical constituents present in 2017:
        "YESBANK.NS", "SAMMAANCAP.NS", "ZEEL.NS", "VEDL.NS", "LUPIN.NS",
        "HINDPETRO.NS", "INDUSTOWER.NS", "GAIL.NS", "IOC.NS", "BPCL.NS", "UPL.NS"
    ],

    # Historical constituents that eventually were excluded or collapsed (for audit tracking)
    "DISTRESSED_HISTORICAL_CONSTITUENTS": [
        "YESBANK.NS", "SAMMAANCAP.NS", "ZEEL.NS", "VEDL.NS", "LUPIN.NS",
        "HINDPETRO.NS", "INDUSTOWER.NS", "GAIL.NS", "IOC.NS", "BPCL.NS", "UPL.NS"
    ]
}

