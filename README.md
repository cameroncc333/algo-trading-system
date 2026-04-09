# Algorithmic Trading System: Cross-Sectional Sector Rotation via Composite Factor Scoring

> **Can a systematic, rules-based allocation model using relative momentum, mean-reversion, and risk-adjusted return signals generate statistically significant alpha over a passive S&P 500 benchmark — net of transaction costs, with realistic risk constraints?**

Built by [Cameron Camarotti](https://github.com/cameroncc333)

---

## Abstract

This project constructs and backtests a sector rotation strategy that ranks the 11 S&P 500 GICS sector ETFs using a cross-sectionally normalized composite of three signals: 63-day price momentum (40% weight), inverted 14-day RSI as a mean-reversion overlay (30%), and 126-day rolling Sharpe ratio (30%). A 200-day moving average golden cross filter gates entries, and a risk management framework includes per-position stop losses, a portfolio-level drawdown circuit breaker, and configurable transaction costs. The strategy is backtested over 6+ years (2020-2026) with a 252-day indicator warm-up window to eliminate look-ahead bias. Results are computed against SPY and presented in an interactive Streamlit dashboard with walk-forward validation and a structured paper trading journal.

---

## Key Findings

| Metric | Strategy | SPY Benchmark |
|--------|----------|---------------|
| **Total Return** | +64.5% | +119% |
| **CAGR** | +8.3% | +13.4% |
| **Sharpe Ratio** | 0.30 | 0.50 |
| **Max Drawdown** | -23.5% | -33.7% |
| **CAPM Alpha** | -0.8% | — |
| **Beta** | 0.52 | 1.00 |
| **Win Rate** | 70% | — |

The strategy delivered roughly half the market's return but with significantly lower volatility (beta 0.52) and a shallower maximum drawdown (-23.5% vs -33.7%). The 70% win rate and low beta indicate the composite signal successfully identifies lower-risk sector rotations, though the current static weight configuration does not generate positive alpha over this period. The walk-forward validation tab and paper trading journal document ongoing parameter iteration.

---

## Methodology

### Signal Generation

Each sector receives a composite score from three cross-sectionally z-score normalized signals:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| **Price Momentum** (63-day) | 40% | Trend-following: sectors in motion tend to stay in motion (Jegadeesh & Titman, 1993) |
| **RSI** (14-day, inverted) | 30% | Mean-reversion overlay: oversold sectors score higher |
| **Rolling Sharpe** (126-day) | 30% | Risk-adjusted return: reward consistency, not just magnitude |

### Trend Filter

Sectors must satisfy both conditions for long eligibility:
1. Price > 200-day simple moving average
2. 50-day SMA > 200-day SMA (golden cross)

### Risk Management

| Control | Parameter | Rationale |
|---------|-----------|-----------|
| **Stop Loss** | 8% per position | Limits single-sector tail risk |
| **Circuit Breaker** | 15% portfolio drawdown | Regime-change protection; peak resets after liquidation |
| **Warm-Up Window** | 252 trading days | Eliminates look-ahead bias from 200-day MA |

### Walk-Forward Validation

To address overfitting, the system includes expanding-window walk-forward validation. Each fold grid-searches 81 parameter combinations on training data, then tests the best configuration on unseen future data. Parameter stability analysis tracks robustness across folds.

---

## Architecture

    algo-trading-system/
    config.py            - All strategy parameters (single source of truth)
    data.py              - Yahoo Finance data pipeline
    signals.py           - RSI, momentum, rolling Sharpe, composite scoring
    backtest.py          - Event-driven portfolio simulation engine
    walk_forward.py      - Walk-forward validation (expanding-window, grid search)
    journal.py           - Paper trading journal (JSON persistence)
    dashboard_app.py     - Streamlit dashboard (6 interactive tabs)
    requirements.txt     - Python 3.9+ dependencies

---

## Dashboard Tabs

1. **Overview** — Strategy explanation, sector snapshot, trailing returns heatmap
2. **Signal Analysis** — Current composite rankings, per-sector RSI/momentum/Sharpe charts
3. **Backtest Results** — Equity curve vs SPY, drawdown analysis, monthly returns heatmap
4. **Trade Log** — Every trade with filters by action, reason, and sector
5. **Paper Trading Journal** — Monthly decision documentation
6. **Walk-Forward Validation** — Out-of-sample equity curve, fold-by-fold results, parameter stability

---

## Limitations and Future Work

### Known Limitations
1. **Survivorship bias**: GICS sector ETFs have not changed, but underlying holdings have
2. **Static risk-free rate**: Using fixed 4.5% Rf; future version should pull time-varying rates from FRED
3. **No fundamental data**: Signals are purely price-derived
4. **Transaction cost assumption**: 10 bps is reasonable for retail but ignores market impact

### Planned Extensions
- FinBERT integration from fomc-sentiment-analyzer as a fourth signal
- Monte Carlo simulation for confidence intervals on Sharpe and alpha
- Regime detection using VIX or yield curve slope

---

## Connection to Research Pipeline

| Project | Role in Pipeline |
|---------|-----------------|
| [AAS-Pricing-Model](https://github.com/cameroncc333/AAS-Pricing-Model) | Calculus-based optimization (partial derivatives, Monte Carlo) |
| [fed-rate-sector-analysis](https://github.com/cameroncc333/fed-rate-sector-analysis) | FOMC rate decision impact on sector returns |
| [equity-sector-analyzer](https://github.com/cameroncc333/equity-sector-analyzer) | Live sector dashboard: Sharpe, RSI, beta, momentum, options pricing |
| [fomc-sentiment-analyzer](https://github.com/cameroncc333/fomc-sentiment-analyzer) | FinBERT transformer NLP across 90+ FOMC meetings |
| **algo-trading-system** (this repo) | **Tests whether the signals produce alpha as trade triggers** |

---

## Run Locally

    pip install streamlit yfinance pandas numpy plotly
    streamlit run dashboard_app.py

Requires Python 3.9+ and internet connection for Yahoo Finance data.

---

*Not financial advice. Educational and analytical tool only.*
