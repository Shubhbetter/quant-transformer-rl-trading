import gymnasium as gym
import numpy as np
from gymnasium import spaces


STATE_COLS = [
    "Close",
    "rsi",
    "macd",
    "volatility",
    "atr",
    "momentum_20",
    "ema_50",
    "ema_200",
    "bb_width",
    "volume_spike",
    "vwap",
    "regime",
    "sentiment",
    "predicted_return",
    "vol_regime",
]


class TradingEnv(gym.Env):
    """
    Actions:
      0: HOLD
      1: BUY small
      2: BUY large
      3: SELL small
      4: SELL large
    """

    def __init__(self, df, initial_capital=10000.0, transaction_cost=0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)

        self.step_index = 0
        self.capital = self.initial_capital
        self.position_qty = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(STATE_COLS),),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        self.capital = self.initial_capital
        self.position_qty = 0.0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.step_index]
        obs = [row[c] if c in row else 0.0 for c in STATE_COLS]
        return np.array(obs, dtype=np.float32)

    def _apply_trade(self, action, price):
        size_factor = 0.0
        side = 0
        if action == 1:
            side, size_factor = 1, 0.25
        elif action == 2:
            side, size_factor = 1, 0.50
        elif action == 3:
            side, size_factor = -1, 0.25
        elif action == 4:
            side, size_factor = -1, 0.50

        txn_cost = 0.0

        if side == 1 and self.capital > 0:
            trade_capital = self.capital * size_factor
            qty = trade_capital / price
            self.position_qty += qty
            self.capital -= trade_capital
            txn_cost = trade_capital * self.transaction_cost
            self.capital -= txn_cost

        if side == -1 and self.position_qty > 0:
            qty = self.position_qty * size_factor
            proceeds = qty * price
            self.position_qty -= qty
            txn_cost = proceeds * self.transaction_cost
            self.capital += (proceeds - txn_cost)

        return txn_cost

    def step(self, action):
        current_price = float(self.df["Close"].iloc[self.step_index])

        txn_cost = self._apply_trade(action, current_price)

        self.step_index += 1
        terminated = self.step_index >= len(self.df) - 1
        truncated = False

        next_price = float(self.df["Close"].iloc[self.step_index])
        prev_value = self.portfolio_value
        self.portfolio_value = self.capital + self.position_qty * next_price

        pnl = self.portfolio_value - prev_value
        ret = pnl / max(prev_value, 1e-9)

        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-9)
        drawdown_penalty = drawdown * 0.1

        risk_adjusted = ret / max(float(self.df["atr"].iloc[self.step_index] / next_price), 1e-6)
        reward = pnl - txn_cost - drawdown_penalty + risk_adjusted

        return self._get_obs(), float(reward), terminated, truncated, {}
