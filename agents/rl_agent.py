from stable_baselines3 import PPO
from market.trading_env import TradingEnv
from utils.data_loader import load_real_data
from datetime import datetime
import numpy as np

def train_rl_agent(ticker="AAPL", total_timesteps=50_000, model_path="agents/ppo_trading_model"):
    # 📅 Aktuelles Datum
    today = datetime.today().strftime("%Y-%m-%d")

    # 📥 Echte Daten laden (skaliert)
    prices, _, _ = load_real_data(ticker=ticker, start="2020-01-01", end=today)

    if not prices:
        print(f"❌ Training aborted: No data for {ticker}")
        return

    # 🧠 Gym-Umgebung erstellen
    env = TradingEnv(prices)

    # 📦 RL-Modell initialisieren und trainieren
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    # 💾 Modell speichern
    model.save(model_path)
    print(f"✅ Modell für {ticker} gespeichert unter: {model_path}")
