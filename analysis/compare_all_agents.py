from market.environment import MarketEnvironment
from market.trading_env import TradingEnv
from agents.random_agent import RandomAgent
from agents.moving_average_agent import MovingAverageAgent
from stable_baselines3 import PPO
from utils.data_loader import load_real_data
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys

def run_random_agent(prices):
    agent = RandomAgent()
    history = []

    for price in prices:
        action = agent.act(price)
        history.append((action, price, agent.cash, agent.holdings))

    df = pd.DataFrame(history, columns=["Action", "Price", "Cash", "Holdings"])
    df["Equity"] = df["Cash"] + df["Holdings"] * df["Price"]
    return df

def run_ma_agent(prices):
    agent = MovingAverageAgent()
    _ = [agent.act(prices, i) for i in range(len(prices))]
    df = pd.DataFrame(agent.history, columns=["Action", "Price", "Cash", "Holdings"])
    df["Equity"] = df["Cash"] + df["Holdings"] * df["Price"]
    return df

def run_rl_agent(prices, model_path="agents/ppo_trading_model"):
    model = PPO.load(model_path)
    env = TradingEnv(prices)
    obs, _ = env.reset()
    done = False
    history = []

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        history.append([
            int(action),
            obs[0],  # price
            obs[1],  # cash
            obs[2]   # holdings
        ])

    df = pd.DataFrame(history, columns=["Action", "Price", "Cash", "Holdings"])
    df["Equity"] = df["Cash"] + df["Holdings"] * df["Price"]
    return df

def print_stats(df, name):
    profit = df["Equity"].iloc[-1] - df["Equity"].iloc[0]
    trades = df[df["Action"].isin([1, 2])].shape[0]
    print(f"{name}: 📊 Gewinn: {profit:.2f} €, 🔁 Trades: {trades}")

def compare_all(ticker="AAPL"):
    today = datetime.today().strftime("%Y-%m-%d")
    prices_scaled, timestamps, scaler = load_real_data(ticker=ticker, start="2020-01-01", end=today)
    timestamps = timestamps[:len(prices_scaled)]

    if not prices_scaled:
        print("❌ No price data loaded. Exiting...")
        return

    # Agenten ausführen
    df_random = run_random_agent(prices_scaled)
    df_ma = run_ma_agent(prices_scaled)
    df_rl = run_rl_agent(prices_scaled)

    # Längen angleichen vor dem Plotten
    min_len = min(len(timestamps), len(df_random), len(df_ma), len(df_rl))
    timestamps = timestamps[:min_len]
    df_random = df_random.iloc[:min_len]
    df_ma = df_ma.iloc[:min_len]
    df_rl = df_rl.iloc[:min_len]

    # Marktpreis rückskalieren
    prices_original = scaler.inverse_transform(np.array(prices_scaled).reshape(-1, 1)).flatten()[:min_len]

    # Plot mit Zeitachse
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(timestamps, df_random["Equity"], label="RandomAgent", zorder=2)
    ax1.plot(timestamps, df_ma["Equity"], label="MovingAverageAgent", zorder=3)
    ax1.plot(timestamps, df_rl["Equity"], label="RL-Agent", zorder=4)
    ax1.set_xlabel("Datum")
    ax1.set_ylabel("Equity (€)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # Zeitformat verbessern
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    # Marktpreis auf rechter Achse
    ax2 = ax1.twinx()
    ax2.plot(timestamps, prices_original, label="Marktpreis", linestyle="--", color="magenta", alpha=0.9, linewidth=2.0, zorder=1)
    ax2.set_ylabel("Preis (€)")
    ax2.legend(loc="upper right")

    plt.title(f"⚔️ Vergleich: RL vs MA vs Random — Ticker: {ticker}")

    print_stats(df_random, "RandomAgent")
    print_stats(df_ma, "MovingAverageAgent")
    print_stats(df_rl, "RL-Agent")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    compare_all(ticker)
