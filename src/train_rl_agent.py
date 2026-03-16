import os

from stable_baselines3 import PPO

from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regime
from src.rl_environment import TradingEnv
from src.transformer_model import add_transformer_predicted_return, set_global_seed


def build_training_frame(ticker="AAPL"):
    data = load_data([ticker, "SPY", "QQQ", "^VIX"], allow_fallback=True)
    df = data[ticker]

    if df.empty:
        raise ValueError(f"No training data available for ticker: {ticker}")

    df = create_features(df)
    spy = create_features(data["SPY"])
    qqq = create_features(data["QQQ"])
    vix = create_features(data["^VIX"])

    df["spy_trend_20"] = spy["Close"].pct_change(20).reindex(df.index).fillna(0.0)
    df["qqq_trend_20"] = qqq["Close"].pct_change(20).reindex(df.index).fillna(0.0)
    df["market_index_trend"] = ((df["spy_trend_20"] + df["qqq_trend_20"]) / 2.0).fillna(0.0)
    df["vix_trend_20"] = vix["Close"].pct_change(20).reindex(df.index).fillna(0.0)

    df = detect_regime(df)

    # Placeholder sentiment in training set; replace with real aligned news feed when available.
    df["sentiment"] = 0.0

    # Volatility regime
    atr_mean = df["atr"].rolling(60).mean()
    df["vol_regime"] = (df["atr"] > atr_mean).astype(float)

    # Transformer predicted return feature for RL state integration
    df = add_transformer_predicted_return(df, price_col="Close", window=64, seed=42)

    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("Feature pipeline produced an empty dataframe.")

    return df


def train_agent(data, total_timesteps=500_000, model_path="models/ppo_trading_agent", seed=42):
    set_global_seed(seed)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


if __name__ == "__main__":
    training_df = build_training_frame("AAPL")
    train_agent(training_df)
