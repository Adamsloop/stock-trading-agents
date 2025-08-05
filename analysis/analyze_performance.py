import pandas as pd
import matplotlib.pyplot as plt

def analyze_trading_log(path="data/trading_log.csv"):
    df = pd.read_csv(path)

    # Berechne Gesamtequity (Cash + Holdings * Preis)
    df["Equity"] = df["Cash"] + df["Holdings"] * df["Price"]

    # Plot Equity-Kurve
    plt.figure(figsize=(10, 5))
    plt.plot(df["Equity"], label="Equity (Vermögen)")
    plt.plot(df["Price"], label="Preis", alpha=0.5)
    plt.title("Performance des Agenten")
    plt.xlabel("Schritt")
    plt.ylabel("Wert (€)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Metriken berechnen
    total_profit = df["Equity"].iloc[-1] - df["Equity"].iloc[0]
    trades = df[df["Action"].isin(["Buy", "Sell"])].shape[0]
    buys = df[df["Action"] == "Buy"].shape[0]
    sells = df[df["Action"] == "Sell"].shape[0]

    print(f"\n📈 Gesamtgewinn: {total_profit:.2f} €")
    print(f"🔁 Anzahl Trades: {trades} (Buys: {buys}, Sells: {sells})")
    print(f"📊 Letzter Stand: {df['Equity'].iloc[-1]:.2f} €")

if __name__ == "__main__":
    analyze_trading_log()
