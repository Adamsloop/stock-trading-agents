import numpy as np

class MarketEnvironment:
    def __init__(self, initial_price=100.0, volatility=0.02, length=100):
        self.initial_price = initial_price
        self.volatility = volatility
        self.length = length
        self.prices = self._generate_price_series()

    def _generate_price_series(self):
        prices = [self.initial_price]
        for _ in range(1, self.length):
            change = np.random.normal(loc=0, scale=self.volatility)
            prices.append(prices[-1] * (1 + change))
        return prices

    def get_prices(self):
        return self.prices
