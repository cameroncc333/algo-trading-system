"""
Backtesting engine — simulates the sector rotation strategy.
"""

import numpy as np
import pandas as pd
from config import (
    SECTOR_ETFS, BENCHMARK, INITIAL_CAPITAL, TRANSACTION_COST_BPS,
    REBALANCE_FREQUENCY, MAX_POSITIONS, STOP_LOSS_PCT,
    MAX_DRAWDOWN_LIMIT, VOLATILITY_LOOKBACK, POSITION_SIZING,
    BACKTEST_WARMUP_DAYS, MOMENTUM_WEIGHT, RSI_WEIGHT, SHARPE_WEIGHT
)
from signals import generate_composite_scores


def get_rebalance_dates(dates, freq):
    if freq == "monthly":
        grouped = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    elif freq == "weekly":
        grouped = pd.Series(dates, index=dates).groupby([dates.year, dates.isocalendar().week]).first()
    else:
        raise ValueError(f"Unknown frequency: {freq}")
    return set(pd.DatetimeIndex(grouped.values))


def compute_risk_parity_weights(prices, tickers, date, lookback=VOLATILITY_LOOKBACK):
    loc = prices.index.get_loc(date)
    if loc < lookback:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    window = prices.iloc[loc - lookback: loc][tickers]
    vol = window.pct_change().std() * np.sqrt(252)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    inv_vol = inv_vol.dropna()
    if inv_vol.sum() == 0:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


def run_backtest(
    prices,
    initial_capital=INITIAL_CAPITAL,
    max_positions=MAX_POSITIONS,
    tx_cost_bps=TRANSACTION_COST_BPS,
    rebal_freq=REBALANCE_FREQUENCY,
    max_dd_limit=MAX_DRAWDOWN_LIMIT,
    stop_loss_pct=STOP_LOSS_PCT,
    position_sizing=POSITION_SIZING,
    mom_weight=MOMENTUM_WEIGHT,
    rsi_weight=RSI_WEIGHT,
    sharpe_weight=SHARPE_WEIGHT,
):
    sector_tickers = list(SECTOR_ETFS.keys())
    composite = generate_composite_scores(
        prices, mom_weight=mom_weight, rsi_weight=rsi_weight, sharpe_weight=sharpe_weight
    )

    warmup = min(BACKTEST_WARMUP_DAYS, len(composite) - 60)
    composite = composite.iloc[warmup:]

    rebalance_dates = get_rebalance_dates(composite.index, rebal_freq)

    cash = initial_capital
    holdings = {}
    portfolio_values = []
    trades = []
    holdings_history = {}
    in_cash_mode = False
    peak_value = initial_capital

    benchmark_prices = prices[BENCHMARK].reindex(composite.index).dropna()
    bench_shares = initial_capital / benchmark_prices.iloc[0]
    tx_cost_pct = tx_cost_bps / 10_000

    for i, date in enumerate(composite.index):
        current_prices = prices.loc[date, sector_tickers]

        port_value = cash
        for ticker, pos in holdings.items():
            if ticker in current_prices.index and not np.isnan(current_prices[ticker]):
                port_value += pos["shares"] * current_prices[ticker]

        peak_value = max(peak_value, port_value)
        drawdown = (peak_value - port_value) / peak_value

        if drawdown > max_dd_limit and not in_cash_mode:
            for ticker, pos in list(holdings.items()):
                sell_price = current_prices.get(ticker, pos["entry_price"])
                if not np.isnan(sell_price):
                    proceeds = pos["shares"] * sell_price * (1 - tx_cost_pct)
                    cash += proceeds
                    trades.append({
                        "date": date, "ticker": ticker, "action": "SELL",
                        "reason": "CIRCUIT_BREAKER",
                        "shares": pos["shares"], "price": sell_price,
                        "pnl_pct": (sell_price / pos["entry_price"]) - 1
                    })
            holdings = {}
            in_cash_mode = True
            port_value = cash
            peak_value = cash

        if in_cash_mode and drawdown < max_dd_limit / 2:
            in_cash_mode = False

        for ticker in list(holdings.keys()):
            pos = holdings[ticker]
            curr_p = current_prices.get(ticker, np.nan)
            if not np.isnan(curr_p):
                loss = (curr_p / pos["entry_price"]) - 1
                if loss < -stop_loss_pct:
                    proceeds = pos["shares"] * curr_p * (1 - tx_cost_pct)
                    cash += proceeds
                    trades.append({
                        "date": date, "ticker": ticker, "action": "SELL",
                        "reason": "STOP_LOSS",
                        "shares": pos["shares"], "price": curr_p,
                        "pnl_pct": loss
                    })
                    del holdings[ticker]

        if date in rebalance_dates and not in_cash_mode:
            scores = composite.loc[date, sector_tickers].dropna()
            valid = scores[scores > -999].sort_values(ascending=False)
            target_tickers = list(valid.head(max_positions).index)

            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    curr_p = current_prices.get(ticker, np.nan)
                    if not np.isnan(curr_p):
                        proceeds = holdings[ticker]["shares"] * curr_p * (1 - tx_cost_pct)
                        pnl = (curr_p / holdings[ticker]["entry_price"]) - 1
                        cash += proceeds
                        trades.append({
                            "date": date, "ticker": ticker, "action": "SELL",
                            "reason": "REBALANCE",
                            "shares": holdings[ticker]["shares"], "price": curr_p,
                            "pnl_pct": pnl
                        })
                        del holdings[ticker]

            if len(target_tickers) > 0:
                if position_sizing == "risk_parity":
                    weights = compute_risk_parity_weights(prices, target_tickers, date)
                else:
                    w = 1.0 / len(target_tickers)
                    weights = {t: w for t in target_tickers}

                total_val = cash
                for ticker, pos in holdings.items():
                    curr_p = current_prices.get(ticker, np.nan)
                    if not np.isnan(curr_p):
                        total_val += pos["shares"] * curr_p

                for ticker in target_tickers:
                    target_val = total_val * weights.get(ticker, 0)
                    curr_p = current_prices.get(ticker, np.nan)
                    if np.isnan(curr_p) or curr_p <= 0:
                        continue

                    current_val = 0
                    if ticker in holdings:
                        current_val = holdings[ticker]["shares"] * curr_p

                    delta_val = target_val - current_val

                    if abs(delta_val) < 50:
                        continue

                    if delta_val > 0:
                        cost = delta_val * (1 + tx_cost_pct)
                        if cost > cash:
                            cost = cash
                            delta_val = cost / (1 + tx_cost_pct)
                        shares_to_buy = delta_val / curr_p
                        if ticker in holdings:
                            old = holdings[ticker]
                            total_shares = old["shares"] + shares_to_buy
                            avg_price = (
                                (old["shares"] * old["entry_price"] + shares_to_buy * curr_p)
                                / total_shares
                            )
                            holdings[ticker] = {"shares": total_shares, "entry_price": avg_price}
                        else:
                            holdings[ticker] = {"shares": shares_to_buy, "entry_price": curr_p}
                        cash -= cost
                        trades.append({
                            "date": date, "ticker": ticker, "action": "BUY",
                            "reason": "REBALANCE",
                            "shares": shares_to_buy, "price": curr_p,
                            "pnl_pct": 0.0
                        })
                    elif delta_val < 0:
                        shares_to_sell = abs(delta_val) / curr_p
                        current_shares = holdings.get(ticker, {}).get("shares", 0)
                        shares_to_sell = min(shares_to_sell, current_shares)
                        if shares_to_sell > 0 and ticker in holdings:
                            entry_p = holdings[ticker]["entry_price"]
                            pnl = (curr_p / entry_p) - 1
                            proceeds = shares_to_sell * curr_p * (1 - tx_cost_pct)
                            cash += proceeds
                            holdings[ticker]["shares"] -= shares_to_sell
                            if holdings[ticker]["shares"] < 0.001:
                                del holdings[ticker]
                            trades.append({
                                "date": date, "ticker": ticker, "action": "SELL",
                                "reason": "REBALANCE_TRIM",
                                "shares": shares_to_sell, "price": curr_p,
                                "pnl_pct": pnl
                            })

        daily_val = cash
        for ticker, pos in holdings.items():
            curr_p = current_prices.get(ticker, np.nan)
            if not np.isnan(curr_p):
                daily_val += pos["shares"] * curr_p

        portfolio_values.append({"date": date, "portfolio": daily_val})
        holdings_history[date] = {
            t: {"shares": p["shares"], "value": p["shares"] * current_prices.get(t, 0)}
            for t, p in holdings.items()
        }

    pv = pd.DataFrame(portfolio_values).set_index("date")["portfolio"]
    bv = bench_shares * benchmark_prices.reindex(pv.index)
    metrics = compute_performance_metrics(pv, bv, trades)

    return {
        "portfolio_values": pv,
        "benchmark_values": bv,
        "trades": trades,
        "holdings_history": holdings_history,
        "metrics": metrics,
    }


def compute_performance_metrics(portfolio, benchmark, trades):
    port_returns = portfolio.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    aligned = pd.DataFrame({"port": port_returns, "bench": bench_returns}).dropna()
    n_years = len(portfolio) / 252

    total_return = (portfolio.iloc[-1] / portfolio.iloc[0]) - 1
    bench_total_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1

    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    bench_cagr = (1 + bench_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = aligned["port"].std() * np.sqrt(252)
    bench_vol = aligned["bench"].std() * np.sqrt(252)

    rf_daily = (1 + 0.045) ** (1 / 252) - 1
    port_std = aligned["port"].std()
    bench_std = aligned["bench"].std()
    sharpe = ((aligned["port"].mean() - rf_daily) / port_std) * np.sqrt(252) if port_std > 0 else 0
    bench_sharpe = ((aligned["bench"].mean() - rf_daily) / bench_std) * np.sqrt(252) if bench_std > 0 else 0

    downside = aligned["port"][aligned["port"] < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1
    sortino = (cagr - 0.045) / downside_std if downside_std > 0 else 0

    cummax = portfolio.cummax()
    drawdowns = (portfolio - cummax) / cummax
    max_dd = drawdowns.min()

    bench_cummax = benchmark.cummax()
    bench_dd = ((benchmark - bench_cummax) / bench_cummax).min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    if len(aligned) > 30:
        cov = np.cov(aligned["port"], aligned["bench"])
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1
        alpha_annual = cagr - (0.045 + beta * (bench_cagr - 0.045))
    else:
        beta = 1.0
        alpha_annual = 0.0

    sells = [t for t in trades if t["action"] == "SELL" and t["reason"] != "CIRCUIT_BREAKER"]
    if sells:
        wins = [t for t in sells if t["pnl_pct"] > 0]
        losses = [t for t in sells if t["pnl_pct"] <= 0]
        win_rate = len(wins) / len(sells)
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        gross_profit = sum(t["pnl_pct"] for t in wins)
        gross_loss = abs(sum(t["pnl_pct"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0

    return {
        "total_return": total_return, "benchmark_return": bench_total_return,
        "cagr": cagr, "benchmark_cagr": bench_cagr,
        "annualized_vol": ann_vol, "benchmark_vol": bench_vol,
        "sharpe": sharpe, "benchmark_sharpe": bench_sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "benchmark_max_dd": bench_dd,
        "calmar": calmar, "alpha": alpha_annual, "beta": beta,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "total_trades": len(trades),
        "n_years": n_years, "final_value": portfolio.iloc[-1],
        "benchmark_final": benchmark.iloc[-1],
    }
