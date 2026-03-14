from stable_baselines3 import PPO

from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regime
from src.rl_environment import TradingEnv


def build_training_frame(ticker="AAPL"):
    df = load_data([ticker], allow_fallback=True)[ticker]
    if df.empty:
        raise ValueError(f"No training data available for ticker: {ticker}")

    df = create_features(df)
    df = detect_regime(df)
    if df.empty:
        raise ValueError("Feature pipeline produced an empty dataframe.")

    return df


def train_agent(data, total_timesteps=500_000, model_path="models/ppo_trading_agent"):
    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    return model


if __name__ == "__main__":
    training_df = build_training_frame("AAPL")
    train_agent(training_df)
