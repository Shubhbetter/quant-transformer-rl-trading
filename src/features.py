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


def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    return tr.rolling(period).mean()


def create_features(df):
    close = _series_1d(df["Close"])
    high = _series_1d(df["High"])
    low = _series_1d(df["Low"])
    volume = _series_1d(df["Volume"])

    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Volume"] = volume

    # Existing indicators
    df["rsi"] = _rsi(close)
    df["macd"] = _macd(close)
    df["ema"] = _ema(close)
    df["volatility"] = close.rolling(10).std()

    # Expanded features
    df["ema_50"] = _ema(close, period=50)
    df["ema_200"] = _ema(close, period=200)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    vol_mean = volume.rolling(20).mean()
    df["volume_spike"] = (volume > (1.5 * vol_mean)).astype(float)

    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    df["vwap"] = (typical_price * volume).cumsum() / cum_vol

    df["atr"] = _atr(high, low, close)
    df["momentum_20"] = close.pct_change(20)

    df.dropna(inplace=True)
    return df
