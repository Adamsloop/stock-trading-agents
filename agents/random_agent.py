import random
import pandas as pd
import os

class RandomAgent:
    def __init__(self, initial_cash=1000):
        self.cash = initial_cash
        self.holdings = 0
        self.history = []

    def decide(self, price):
        """
        Entscheidungen (numerisch):
        - 0 = Halten
        - 1 = Kaufen
        - 2 = Verkaufen
        """
        return random.choice([0, 1, 2])

    def act(self, price):
        action = self.decide(price)

        if action == 1 and self.cash >= price:
            # Buy
            self.cash -= price
            self.holdings += 1
        elif action == 2 and self.holdings > 0:
            # Sell
            self.cash += price
            self.holdings -= 1
        else:
            # Hold (keine Aktion möglich oder entschieden)
            action = 0

        self.history.append((action, price, self.cash, self.holdings))
        return action

    def save_history(self, path="data/random_log.csv"):
        """
        Speichert die Handels-Historie als CSV.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.history, columns=["Action", "Price", "Cash", "Holdings"])
        df.to_csv(path, index_label="Step")