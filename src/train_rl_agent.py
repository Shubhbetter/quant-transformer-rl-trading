from stable_baselines3 import PPO
from rl_environment import TradingEnv

env = TradingEnv(data)

model = PPO(
"MlpPolicy",
env,
verbose=1
)

model.learn(
total_timesteps=500000
)

model.save("models/ppo_trading_agent")
