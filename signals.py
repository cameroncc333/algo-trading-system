"""
Signals module — generates trade signals from RSI, momentum, rolling Sharpe,
and moving average filters. Produces a composite score used to rank sectors.
"""

import numpy as np
import pandas as pd
from config import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MOMENTUM_LOOKBACK_DAYS, MOMENTUM_WEIGHT, RSI_WEIGHT, SHARPE_WEIGHT,
    SHARPE_LOOKBACK_DAYS, RISK_FREE_RATE,
    MA_LONG, MA_SHORT, SECTOR_ETFS
)


def compute_rsi(prices: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_momentum(prices: pd.DataFrame, lookback: int = MOMENTUM_LOOKBACK_DAYS) -> pd.DataFrame:
    return prices.pct_change(lookback)


def compute_rolling_sharpe(
    prices: pd.DataFrame,
    lookback: int = SHARPE_LOOKBACK_DAYS,
    risk_free: float = RISK_FREE_RATE
) -> pd.DataFrame:
    daily_returns = prices.pct_change()
    rolling_mean = daily_returns.rolling(lookback).mean() * 252
    rolling_std = daily_returns.rolling(lookback).std() * np.sqrt(252)
    sharpe = (rolling_mean - risk_free) / rolling_std
    return sharpe


def compute_moving_averages(prices: pd.DataFrame) -> tuple:
    ma_short = prices.rolling(MA_SHORT).mean()
    ma_long = prices.rolling(MA_LONG).mean()
    return ma_short, ma_long


def ma_filter(prices: pd.DataFrame) -> pd.DataFrame:
    ma_short, ma_long = compute_moving_averages(prices)
    above_trend = prices > ma_long
    golden_cross = ma_short > ma_long
    return above_trend & golden_cross


def normalize_cross_sectional(df: pd.DataFrame, sector_cols: list) -> pd.DataFrame:
    subset = df[sector_cols]
    row_mean = subset.mean(axis=1)
    row_std = subset.std(axis=1)
    normalized = subset.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)
    return normalized.fillna(0)


def generate_composite_scores(
    prices: pd.DataFrame,
    mom_weight: float = MOMENTUM_WEIGHT,
    rsi_weight: float = RSI_WEIGHT,
    sharpe_weight: float = SHARPE_WEIGHT,
) -> pd.DataFrame:
    sector_tickers = list(SECTOR_ETFS.keys())

    total_weight = mom_weight + rsi_weight + sharpe_weight
    if total_weight > 0:
        mom_weight /= total_weight
        rsi_weight /= total_weight
        sharpe_weight /= total_weight

    rsi = compute_rsi(prices[sector_tickers])
    momentum = compute_momentum(prices[sector_tickers])
    sharpe = compute_rolling_sharpe(prices[sector_tickers])
    trend_ok = ma_filter(prices[sector_tickers])

    rsi_score = 50 - rsi

    rsi_z = normalize_cross_sectional(rsi_score, sector_tickers)
    mom_z = normalize_cross_sectional(momentum, sector_tickers)
    sharpe_z = normalize_cross_sectional(sharpe, sector_tickers)

    composite = (
        mom_weight * mom_z
        + rsi_weight * rsi_z
        + sharpe_weight * sharpe_z
    )

    composite = composite.where(trend_ok, -999)
    return composite


def get_signal_components(prices: pd.DataFrame) -> dict:
    sector_tickers = list(SECTOR_ETFS.keys())
    return {
        "rsi": compute_rsi(prices[sector_tickers]),
        "momentum": compute_momentum(prices[sector_tickers]),
        "rolling_sharpe": compute_rolling_sharpe(prices[sector_tickers]),
        "ma_filter": ma_filter(prices[sector_tickers]),
        "composite": generate_composite_scores(prices),
    }
