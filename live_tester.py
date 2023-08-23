import pandas as pd
import numpy as np
import json
from datetime import datetime
import ccxt
import os

# Initialize Kraken client
kraken = ccxt.kraken(
    {
        "apiKey": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET",
    }
)
script_dir = os.path.dirname(os.path.abspath(__file__))


def calculate_supertrend(df, period=7, multiplier=3):
    hl_avg = (df["High"] + df["Low"]) / 2
    atr = pd.DataFrame.ewm(hl_avg - hl_avg.shift(), span=period).mean()

    df["upper_band"] = hl_avg + (multiplier * atr)
    df["lower_band"] = hl_avg - (multiplier * atr)
    df["in_uptrend"] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df["Close"][current] > df["upper_band"][previous]:
            df.at[current, "in_uptrend"] = True
        elif df["Close"][current] < df["lower_band"][previous]:
            df.at[current, "in_uptrend"] = False
        else:
            df.at[current, "in_uptrend"] = df["in_uptrend"][previous]
            if (
                df["in_uptrend"][current]
                and df["lower_band"][current] < df["lower_band"][previous]
            ):
                df.at[current, "lower_band"] = df["lower_band"][previous]
            if (
                not df["in_uptrend"][current]
                and df["upper_band"][current] > df["upper_band"][previous]
            ):
                df.at[current, "upper_band"] = df["upper_band"][previous]

    return df


def calculate_slope(series, period=5):
    # Initialize a list with np.nan for the first period-1 elements
    slopes = [np.nan for _ in range(period - 1)]

    for i in range(period, len(series) + 1):  # change this line
        y = series[i - period : i]
        x = np.array(range(period))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        if np.isnan(y_scaled).any():  # skip if any value is NaN
            slopes.append(np.nan)
            continue
        results = np.polyfit(x_scaled, y_scaled, 1)
        slopes.append(results[0])
    slope_angle = np.rad2deg(np.arctan(np.array(slopes)))
    return np.array(slope_angle)


def fetch_data():
    # Define the symbol and timeframe
    symbol = "SOL/USD"  # replace with your desired trading pair
    timeframe = "5m"  # 5 minute OHLCV data

    # Fetch the OHLCV data
    ohlcv = kraken.fetch_ohlcv(symbol, timeframe)

    # Convert to a DataFrame
    df = pd.DataFrame(
        ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")

    # Calculate the Supertrend and slope
    df = calculate_supertrend(df)
    df["slope"] = calculate_slope(df["Close"])

    return df


def place_order(order_type, price, amount):
    symbol = "SOL/USD"  # replace with your desired trading pair

    # Define the order
    order = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "type": order_type,
        "side": "buy" if order_type == "limit" else "sell",
        "price": price,
        "amount": amount,
    }

    file_path = os.path.join(script_dir, "orders.log")

    # Write the order to a file
    with open(file_path, "a") as f:
        f.write(json.dumps(order) + "\n")


def load_open_position():
    file_path = os.path.join(script_dir, "open_position.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return None


def save_open_position(open_position):
    file_path = os.path.join(script_dir, "open_position.json")
    with open(file_path, "w") as f:
        json.dump(open_position, f)


def main():
    # Fetch the data
    df = fetch_data()

    # Load the open position, if it exists
    opened_at = load_open_position()

    # Check the latest Supertrend and slope
    latest = df.iloc[-1]
    if latest["in_uptrend"] and latest["slope"] > 1 and opened_at is None:
        # Simulate a buy order
        opened_at = {"time": str(latest["Timestamp"]), "price": latest["Close"]}
        save_open_position(opened_at)
        place_order("limit", latest["Close"], 50)
    elif not latest["in_uptrend"] and latest["slope"] < 0 and opened_at is not None:
        # Simulate a sell order
        place_order("market", latest["Close"], 50)
        file_path = os.path.join(script_dir, "open_position.json")
        os.remove(file_path)


if __name__ == "__main__":
    main()
