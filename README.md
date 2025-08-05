# Stock Trading Agents

A Python project that compares different trading agents including **Random**, **Moving Average (MA)**, and **Reinforcement Learning (PPO)** agents using real stock market data.  
This project includes training, evaluation, and visualization of agent performance on historical price data from Yahoo Finance.

---

## Features

- **Random Agent**: Trades randomly (buy, sell, hold).  
- **Moving Average Agent**: Uses short and long moving averages crossover strategy for trading decisions.  
- **Reinforcement Learning Agent**: Trained with PPO algorithm on historical price data.  
- Supports real market data fetching via Yahoo Finance (`yfinance` package).  
- Data scaling and inverse transformation for better training and plotting.  
- Interactive performance visualization with equity curves and market price overlay.  
- CLI support for different stock tickers (e.g., AAPL, TSLA, BTC-USD).

---

# Installation
## 1. Clone this repository
```
git clone https://github.com/YourUsername/stock-trading-agents.git
cd stock-trading-agents
```

## 2. Create and activate a virtual environment (optional but recommended):
```
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

## 3. Install the required packages:
```
pip install -r requirements.txt
```
---
# Usage
## Train the Reinforcement Learning Agent
```
python train_rl.py [TICKER]

```
TICKER is optional, defaults to AAPL.


## Example:

```
python train_rl.py TSLA
```

## Compare Agents Performance

```
python -m analysis.compare_all_agents [TICKER]
```
Visualizes the equity curves of Random, Moving Average, and RL agents using historical data.
## Example:

```
python -m analysis.compare_all_agents BTC-USD
```
---
## Project Structure
- agents/ – Contains trading agent implementations (Random, MA, RL).

- market/ – Market environment and trading simulation environment.

- utils/ – Utility functions such as data loading from Yahoo Finance.

- analysis/ – Scripts for comparing and plotting agent performance.

- train_rl.py – Script for training the PPO RL agent on historical data.
---
## Dependencies
- Python 3.8+
- stable-baselines3
- gym
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn

(Install via pip install -r requirements.txt)

---
## Notes

- Data is automatically downloaded and scaled before training and evaluation.

- You can specify any valid Yahoo Finance ticker symbol to evaluate different stocks or cryptocurrencies.

- The RL agent training can take some time depending on the total_timesteps parameter.

---

## License
This project is licensed under the MIT License.

---

## Author
Adam Kaczmarek (GitHub: @Adamsloop)