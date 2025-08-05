import pandas as pd
import os

class MovingAverageAgent:
    def __init__(self, short_window=3, long_window=10, initial_cash=1000):
        self.short_window = short_window
        self.long_window = long_window
        self.cash = initial_cash
        self.holdings = 0
        self.history = []

    def act(self, price_series, current_step):
        if current_step < self.long_window:
            self.history.append((0, price_series[current_step], self.cash, self.holdings))  # 0 = Hold
            return 0

        data = pd.Series(price_series[:current_step + 1])
        short_ma = data.rolling(window=self.short_window).mean()
        long_ma = data.rolling(window=self.long_window).mean()

        current_price = price_series[current_step]
        action = 0  # Default: Hold

        # Kauf-Signal
        if short_ma.iloc[-2] < long_ma.iloc[-2] and short_ma.iloc[-1] > long_ma.iloc[-1]:
            if self.cash >= current_price:
                self.cash -= current_price
                self.holdings += 1
                action = 1  # Buy

        # Verkauf-Signal
        elif short_ma.iloc[-2] > long_ma.iloc[-2] and short_ma.iloc[-1] < long_ma.iloc[-1]:
            if self.holdings > 0:
                self.cash += current_price
                self.holdings -= 1
                action = 2  # Sell

        self.history.append((action, current_price, self.cash, self.holdings))
        return action

    def save_history(self, path="data/ma_log.csv"):
        df = pd.DataFrame(self.history, columns=["Action", "Price", "Cash", "Holdings"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index_label="Step")
