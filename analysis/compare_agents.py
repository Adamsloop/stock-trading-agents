from market.environment import MarketEnvironment
from agents.random_agent import RandomAgent
from agents.moving_average_agent import MovingAverageAgent
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_agent(agent, prices, name, output_path):
    actions = []
    for step in range(len(prices)):
        if agent.__class__.__name__ == "RandomAgent":
            action = agent.act(prices[step])
        else:
            action = agent.act(prices, step)
        actions.append(action)
    agent.save_history(output_path)
    return pd.read_csv(output_path), actions

def compare_agents():
    # 1. Erzeuge den selben Markt für beide
    market = MarketEnvironment(initial_price=100, volatility=0.02, length=100)
    prices = market.get_prices()

    os.makedirs("data", exist_ok=True)

    # 2. Starte beide Agenten
    random_agent = RandomAgent()
    ma_agent = MovingAverageAgent()

    df_random, _ = run_agent(random_agent, prices, "Random", "data/random_log.csv")
    df_ma, _ = run_agent(ma_agent, prices, "MA", "data/ma_log.csv")

    # 3. Berechne Equity
    df_random["Equity"] = df_random["Cash"] + df_random["Holdings"] * df_random["Price"]
    df_ma["Equity"] = df_ma["Cash"] + df_ma["Holdings"] * df_ma["Price"]

    # 4. Plotten
    plt.figure(figsize=(10, 5))
    plt.plot(df_random["Equity"], label="RandomAgent")
    plt.plot(df_ma["Equity"], label="MovingAverageAgent")
    plt.plot(prices, label="Market Price", alpha=0.3, linestyle="--")
    plt.title("Agentenvergleich: Vermögensentwicklung")
    plt.xlabel("Schritt")
    plt.ylabel("Equity (€)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 5. Metriken
    def print_stats(df, name):
        profit = df["Equity"].iloc[-1] - df["Equity"].iloc[0]
        trades = df[df["Action"].isin(["Buy", "Sell"])].shape[0]
        print(f"\n{name}:")
        print(f"  📊 Gewinn: {profit:.2f} €")
        print(f"  🔁 Trades: {trades}")

    print_stats(df_random, "RandomAgent")
    print_stats(df_ma, "MovingAverageAgent")

if __name__ == "__main__":
    compare_agents()
