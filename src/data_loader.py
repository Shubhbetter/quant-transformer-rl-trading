import numpy as np
import pandas as pd
import yfinance as yf


PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _fallback_ohlcv(periods=300, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0005, scale=0.015, size=periods)
    close = 100 * np.exp(np.cumsum(returns))

    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.01, size=periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.01, size=periods))
    volume = rng.integers(100_000, 1_000_000, size=periods)

    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def _to_series(column):
    if isinstance(column, pd.DataFrame):
        # Happens with yfinance multi-index outputs or duplicated labels.
        column = column.iloc[:, 0]
    return pd.to_numeric(column, errors="coerce")


def _normalize_price_frame(df):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    normalized = pd.DataFrame(index=df.index)
    for col in PRICE_COLS:
        if col in df.columns:
            normalized[col] = _to_series(df[col])

    return normalized


def load_data(tickers, start="2014-01-01", allow_fallback=True):
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        except Exception:
            df = pd.DataFrame()

        df = _normalize_price_frame(df)

        if df.empty and allow_fallback:
            df = _fallback_ohlcv()

        data[ticker] = df

    return data
