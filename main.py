from market.environment import MarketEnvironment
from agents.moving_average_agent import MovingAverageAgent
import matplotlib.pyplot as plt

def main():
    market = MarketEnvironment(initial_price=100, volatility=0.02, length=100)
    agent = MovingAverageAgent(short_window=5, long_window=20, initial_cash=1000)

    prices = market.get_prices()
    actions = []

    for step in range(len(prices)):
        action = agent.act(prices, step)
        actions.append(action)

    # Plot
    plt.plot(prices, label="Price")

    buy_x = [i for i, a in enumerate(actions) if a == "Buy"]
    buy_y = [prices[i] for i in buy_x]
    sell_x = [i for i, a in enumerate(actions) if a == "Sell"]
    sell_y = [prices[i] for i in sell_x]

    plt.scatter(buy_x, buy_y, color="green", marker="^", label="Buy")
    plt.scatter(sell_x, sell_y, color="red", marker="v", label="Sell")

    plt.title("Moving Average Agent – Buy/Sell Aktionen")
    plt.xlabel("Zeit")
    plt.ylabel("Preis")
    plt.legend()
    plt.grid()
    plt.show()

    # Speichere Trading-Verlauf in CSV
    agent.save_history("data/trading_log.csv")


if __name__ == "__main__":
    main()
