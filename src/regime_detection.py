from sklearn.cluster import KMeans

def detect_regime(df):

    features = df[["rsi","macd","volatility"]]

    model = KMeans(n_clusters=3)

    regimes = model.fit_predict(features)

    df["regime"] = regimes

    return df
