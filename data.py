"""
Data module — fetches and preprocesses sector ETF and benchmark price data.
Uses Yahoo Finance via yfinance for adjusted close prices.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from config import SECTOR_ETFS, BENCHMARK, BACKTEST_START, BACKTEST_END


def fetch_prices(start: str = BACKTEST_START, end: str = BACKTEST_END) -> pd.DataFrame:
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna(how="all")
    data = data.ffill().bfill()
    return data


def compute_returns(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return prices.pct_change(period).dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


if __name__ == "__main__":
    prices = fetch_prices()
    print(f"Fetched {len(prices)} trading days for {prices.shape[1]} tickers")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(prices.tail())
