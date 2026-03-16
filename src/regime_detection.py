import numpy as np


def detect_regime(df):
    """
    Deterministic market regime model using requested feature set:
    - EMA 50 / EMA 200 relationship
    - momentum
    - volatility context

    Numeric mapping kept for downstream compatibility:
      0 -> Bullish
      1 -> Sideways
      2 -> Bearish
    """
    price = df["Close"]
    ema50 = df["ema_50"]
    ema200 = df["ema_200"]
    momentum = df["momentum_20"]
    vol = df["volatility"]
    vol_mean = vol.rolling(60).mean()

    bullish = (price > ema200) & (ema50 > ema200) & (momentum > -0.01)
    bearish = (price < ema200) & (ema50 < ema200) & (momentum < 0.01)

    # If volatility is very elevated, avoid overconfident bull/bear assignment.
    high_vol = vol > vol_mean

    regime = np.where(
        bullish & (~high_vol),
        0,
        np.where(bearish & (~high_vol), 2, 1),
    )

    df["regime"] = regime.astype(int)
    return df
