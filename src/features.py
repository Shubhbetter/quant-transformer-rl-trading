import numpy as np
import pandas as pd


def _series_1d(values):
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _macd(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _ema(close, period=20):
    return close.ewm(span=period, adjust=False).mean()


def create_features(df):
    close = _series_1d(df["Close"])

    df["Close"] = close
    df["rsi"] = _rsi(close)
    df["macd"] = _macd(close)
    df["ema"] = _ema(close)
    df["volatility"] = close.rolling(10).std()

    df.dropna(inplace=True)

    return df
