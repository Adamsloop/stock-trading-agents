import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def load_real_data(ticker="AAPL", start="2020-01-01", end=None, scale=True, feature_range=(80, 120)):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end)
        if "Close" not in df.columns or df.empty:
            raise ValueError("No closing prices found in data.")

        prices = df["Close"].dropna().values.reshape(-1, 1)
        timestamps = df["Close"].dropna().index.to_list()
        scaler = None

        if scale:
            scaler = MinMaxScaler(feature_range=feature_range)
            prices = scaler.fit_transform(prices)

        return prices.flatten().tolist(), timestamps, scaler

    except Exception as e:
        print(f"❌ Failed to load data for {ticker}: {e}")
        return [], [], None
