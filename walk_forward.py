"""
Walk-Forward Validation — eliminates overfitting by never testing on training data.
"""

import numpy as np
import pandas as pd
from itertools import product
from config import (
    SECTOR_ETFS, BENCHMARK, INITIAL_CAPITAL, TRANSACTION_COST_BPS,
    REBALANCE_FREQUENCY, MAX_POSITIONS, STOP_LOSS_PCT,
    MAX_DRAWDOWN_LIMIT, POSITION_SIZING, BACKTEST_WARMUP_DAYS
)
from backtest import run_backtest


DEFAULT_PARAM_GRID = {
    "mom_weight":  [0.30, 0.40, 0.50],
    "rsi_weight":  [0.20, 0.30, 0.40],
    "sharpe_weight": [0.20, 0.30, 0.40],
    "max_positions": [2, 3, 4],
}


def run_walk_forward(
    prices,
    train_months=24,
    test_months=6,
    param_grid=None,
    initial_capital=INITIAL_CAPITAL,
    tx_cost_bps=TRANSACTION_COST_BPS,
    rebal_freq=REBALANCE_FREQUENCY,
    max_dd_limit=MAX_DRAWDOWN_LIMIT,
    stop_loss_pct=STOP_LOSS_PCT,
):
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combos = [dict(zip(param_names, combo)) for combo in product(*param_values)]

    weight_params = {"mom_weight", "rsi_weight", "sharpe_weight"}
    if weight_params.issubset(set(param_names)):
        param_combos = [
            p for p in param_combos
            if abs(p["mom_weight"] + p["rsi_weight"] + p["sharpe_weight"] - 1.0) < 0.15
        ]

    warmup_start = BACKTEST_WARMUP_DAYS
    valid_dates = prices.index[warmup_start:]

    folds = []
    fold_num = 0

    while True:
        train_end_date = valid_dates[0] + pd.DateOffset(months=train_months + (fold_num * test_months))
        test_end_date = train_end_date + pd.DateOffset(months=test_months)

        train_mask = valid_dates <= train_end_date
        test_mask = (valid_dates > train_end_date) & (valid_dates <= test_end_date)

        train_dates = valid_dates[train_mask]
        test_dates = valid_dates[test_mask]

        if len(test_dates) < 20:
            break

        folds.append({
            "fold": fold_num + 1,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "test_start": test_dates[0],
            "test_end": test_dates[-1],
            "train_days": len(train_dates),
            "test_days": len(test_dates),
        })
        fold_num += 1

    if not folds:
        return {"error": "Not enough data for walk-forward validation"}

    fold_results = []
    oos_returns_list = []

    for fold in folds:
        train_start_with_warmup = prices.index[
            max(0, prices.index.get_loc(fold["train_start"]) - BACKTEST_WARMUP_DAYS)
        ]
        train_prices = prices.loc[train_start_with_warmup:fold["train_end"]]

        best_sharpe = -999
        best_params = param_combos[0]
        best_train_metrics = {}

        for params in param_combos:
            try:
                result = run_backtest(
                    train_prices,
                    initial_capital=initial_capital,
                    max_positions=params.get("max_positions", MAX_POSITIONS),
                    tx_cost_bps=tx_cost_bps,
                    rebal_freq=rebal_freq,
                    max_dd_limit=max_dd_limit,
                    stop_loss_pct=stop_loss_pct,
                    mom_weight=params.get("mom_weight", 0.4),
                    rsi_weight=params.get("rsi_weight", 0.3),
                    sharpe_weight=params.get("sharpe_weight", 0.3),
                )
                train_sharpe = result["metrics"]["sharpe"]
                if train_sharpe > best_sharpe:
                    best_sharpe = train_sharpe
                    best_params = params.copy()
                    best_train_metrics = result["metrics"].copy()
            except Exception:
                continue

        test_start_with_warmup = prices.index[
            max(0, prices.index.get_loc(fold["test_start"]) - BACKTEST_WARMUP_DAYS)
        ]
        test_prices = prices.loc[test_start_with_warmup:fold["test_end"]]

        try:
            test_result = run_backtest(
                test_prices,
                initial_capital=initial_capital,
                max_positions=best_params.get("max_positions", MAX_POSITIONS),
                tx_cost_bps=tx_cost_bps,
                rebal_freq=rebal_freq,
                max_dd_limit=max_dd_limit,
                stop_loss_pct=stop_loss_pct,
                mom_weight=best_params.get("mom_weight", 0.4),
                rsi_weight=best_params.get("rsi_weight", 0.3),
                sharpe_weight=best_params.get("sharpe_weight", 0.3),
            )
            test_metrics = test_result["metrics"]
            oos_pv = test_result["portfolio_values"]
            oos_returns = oos_pv.pct_change().dropna()
            oos_returns_list.append(oos_returns)
        except Exception:
            test_metrics = {"sharpe": 0, "total_return": 0, "max_drawdown": 0}

        fold_results.append({
            **fold,
            "best_params": best_params,
            "train_sharpe": best_sharpe,
            "train_return": best_train_metrics.get("total_return", 0),
            "test_sharpe": test_metrics.get("sharpe", 0),
            "test_return": test_metrics.get("total_return", 0),
            "test_max_dd": test_metrics.get("max_drawdown", 0),
            "test_alpha": test_metrics.get("alpha", 0),
        })

    if oos_returns_list:
        all_oos_returns = pd.concat(oos_returns_list)
        all_oos_returns = all_oos_returns[~all_oos_returns.index.duplicated(keep="first")]
        all_oos_returns = all_oos_returns.sort_index()
        oos_equity = initial_capital * (1 + all_oos_returns).cumprod()

        bench_prices = prices[BENCHMARK].reindex(oos_equity.index).dropna()
        if len(bench_prices) > 1:
            bench_equity = initial_capital * (bench_prices / bench_prices.iloc[0])
        else:
            bench_equity = pd.Series(dtype=float)

        oos_metrics = _compute_oos_metrics(oos_equity, bench_equity)
    else:
        oos_equity = pd.Series(dtype=float)
        bench_equity = pd.Series(dtype=float)
        oos_metrics = {}

    param_stability = {}
    for param_name in param_names:
        values_chosen = [f["best_params"][param_name] for f in fold_results]
        unique_vals, counts = np.unique(values_chosen, return_counts=True)
        param_stability[param_name] = {
            "values": dict(zip([str(v) for v in unique_vals], counts.tolist())),
            "most_common": str(unique_vals[counts.argmax()]),
            "stability": float(counts.max() / len(values_chosen)),
        }

    return {
        "folds": fold_results,
        "oos_equity": oos_equity,
        "benchmark_equity": bench_equity,
        "oos_metrics": oos_metrics,
        "param_stability": param_stability,
        "n_folds": len(fold_results),
        "n_param_combos": len(param_combos),
    }


def _compute_oos_metrics(equity, benchmark):
    if len(equity) < 30:
        return {"error": "Insufficient OOS data"}

    returns = equity.pct_change().dropna()
    n_years = len(equity) / 252

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    rf_daily = (1 + 0.045) ** (1 / 252) - 1
    std = returns.std()
    sharpe = ((returns.mean() - rf_daily) / std) * np.sqrt(252) if std > 0 else 0

    cummax = equity.cummax()
    max_dd = ((equity - cummax) / cummax).min()

    bench_return = 0
    bench_sharpe = 0
    if len(benchmark) > 30:
        bench_ret = benchmark.pct_change().dropna()
        bench_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
        bench_std = bench_ret.std()
        bench_sharpe = ((bench_ret.mean() - rf_daily) / bench_std) * np.sqrt(252) if bench_std > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "benchmark_return": bench_return,
        "benchmark_sharpe": bench_sharpe,
        "n_years": n_years,
    }
