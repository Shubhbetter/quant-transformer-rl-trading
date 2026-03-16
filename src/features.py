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


def _adx(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr = pd.Series(tr, index=close.index).rolling(period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(period).sum() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(period).sum() / atr.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.rolling(period).mean()
    return adx


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
    df["ema_20"] = _ema(close, period=20)
    df["ema_50"] = _ema(close, period=50)
    df["ema_200"] = _ema(close, period=200)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    vol_mean = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_spike"] = (volume > (1.5 * vol_mean)).astype(float)
    df["volume_spike_ratio"] = (volume / vol_mean).replace([np.inf, -np.inf], np.nan)

    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    df["vwap"] = (typical_price * volume).cumsum() / cum_vol

    df["atr"] = _atr(high, low, close)
    df["adx"] = _adx(high, low, close)
    df["momentum_20"] = close.pct_change(20)
    df["log_return"] = np.log(close / close.shift(1))
    df["rolling_volatility_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)

    df.dropna(inplace=True)
    return df
