import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):

    def __init__(self, df):

        self.df = df
        self.step_index = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

    def reset(self):

        self.step_index = 0
        return self._get_obs()

    def _get_obs(self):

        row = self.df.iloc[self.step_index]

        return np.array([
            row["Close"],
            row["rsi"],
            row["macd"],
            row["volatility"],
            row["regime"]
        ])

    def step(self, action):

        self.step_index += 1

        done = self.step_index >= len(self.df)-1

        price_now = self.df["Close"].iloc[self.step_index]
        price_prev = self.df["Close"].iloc[self.step_index-1]

        change = price_now - price_prev

        reward = 0

        if action == 1:
            reward = change

        elif action == 2:
            reward = -change

        return self._get_obs(), reward, done, {}
