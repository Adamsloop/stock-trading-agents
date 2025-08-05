from agents.rl_agent import train_rl_agent
import sys

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    train_rl_agent(ticker=ticker)
