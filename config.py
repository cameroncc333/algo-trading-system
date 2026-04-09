"""
Algorithmic Trading System — Configuration
Sector rotation strategy using RSI, momentum, and risk-adjusted signals.
Built by Cameron Camarotti | github.com/cameroncc333
"""

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}

BENCHMARK = "SPY"

BACKTEST_START = "2019-01-01"
BACKTEST_END = "2026-04-01"
BACKTEST_WARMUP_DAYS = 252
INITIAL_CAPITAL = 10_000
TRANSACTION_COST_BPS = 10
REBALANCE_FREQUENCY = "monthly"
MAX_POSITIONS = 3
POSITION_SIZING = "equal_weight"

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MOMENTUM_LOOKBACK_DAYS = 63
MOMENTUM_WEIGHT = 0.40
RSI_WEIGHT = 0.30
SHARPE_WEIGHT = 0.30

SHARPE_LOOKBACK_DAYS = 126
RISK_FREE_RATE = 0.045

MA_LONG = 200
MA_SHORT = 50

MAX_DRAWDOWN_LIMIT = 0.15
STOP_LOSS_PCT = 0.08
VOLATILITY_LOOKBACK = 21

JOURNAL_FILE = "paper_trading_journal.json"
