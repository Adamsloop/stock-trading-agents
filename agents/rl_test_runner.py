from stable_baselines3 import PPO
from market.trading_env import TradingEnv
from market.environment import MarketEnvironment
import pandas as pd
import numpy as np
import os

def test_rl_agent(model_path="agents/ppo_trading_model", log_path="data/rl_log.csv"):
    # 1. Erzeuge neue Preisdaten
    market = MarketEnvironment(initial_price=100, volatility=0.02, length=100)
    prices = market.get_prices()

    # 2. Umgebung initialisieren
    env = TradingEnv(prices)
    obs, _ = env.reset()

    # 3. Modell laden
    model = PPO.load(model_path)

    # 4. Simulation ausführen
    history = []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        history.append([
            int(action),
            obs[0],    # price
            obs[1],    # cash
            obs[2]     # holdings
        ])

    # 5. Speichern
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df = pd.DataFrame(history, columns=["Action", "Price", "Cash", "Holdings"])
    df.to_csv(log_path, index_label="Step")
    print(f"✅ RL-Trading-Log gespeichert unter: {log_path}")
