import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, prices, initial_cash=1000):
        super(TradingEnv, self).__init__()
        self.prices = prices
        self.initial_cash = initial_cash
        self.current_step = 0

        self.cash = initial_cash
        self.holdings = 0
        self.done = False

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.cash = self.initial_cash
        self.holdings = 0
        self.current_step = 0
        self.done = False

        return self._get_obs(), {}

    def step(self, action):
        price = self.prices[self.current_step]

        if action == 1 and self.cash >= price:  # Buy
            self.cash -= price
            self.holdings += 1
        elif action == 2 and self.holdings > 0:  # Sell
            self.cash += price
            self.holdings -= 1
        # else: Hold or invalid move → do nothing

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            self.done = True

        next_obs = self._get_obs()
        reward = self.cash + self.holdings * price  # total equity as reward basis
        terminated = self.done
        truncated = False  # not used here
        info = {}

        return next_obs, reward, terminated, truncated, info

    def _get_obs(self):
        price = float(self.prices[self.current_step])  # explizit casten
        return np.array([price, self.cash, self.holdings], dtype=np.float32)

