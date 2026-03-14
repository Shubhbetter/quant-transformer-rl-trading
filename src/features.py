import ta

def create_features(df):

    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    df["macd"] = ta.trend.MACD(df["Close"]).macd()

    df["ema"] = ta.trend.EMAIndicator(df["Close"]).ema_indicator()

    df["volatility"] = df["Close"].rolling(10).std()

    df.dropna(inplace=True)

    return df
