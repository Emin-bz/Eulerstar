import ccxt
import pandas as pd
import numpy as np

# Initialize Kraken client
kraken = ccxt.kraken(
    {
        "apiKey": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET",
    }
)


def calculate_supertrend(df, period=7, multiplier=3):
    """Calculate Supertrend given a pandas dataframe of OHLC data."""
    hl_avg = (df["High"] + df["Low"]) / 2
    atr = pd.DataFrame.ewm(hl_avg - hl_avg.shift(), span=period).mean()

    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)

    df["upper_band"] = upper_band
    df["lower_band"] = lower_band
    df["in_uptrend"] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df["Close"][current] > df["upper_band"][previous]:
            df["in_uptrend"][current] = True
        elif df["Close"][current] < df["lower_band"][previous]:
            df["in_uptrend"][current] = False
        else:
            df["in_uptrend"][current] = df["in_uptrend"][previous]
            if (
                df["in_uptrend"][current]
                and df["lower_band"][current] < df["lower_band"][previous]
            ):
                df["lower_band"][current] = df["lower_band"][previous]
            if (
                not df["in_uptrend"][current]
                and df["upper_band"][current] > df["upper_band"][previous]
            ):
                df["upper_band"][current] = df["upper_band"][previous]

    return df


def calculate_slope(series, period=5):
    """Calculate the slope of a pandas series."""
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
    """Fetch the latest data from Kraken."""
    # Define the symbol and timeframe
    symbol = "BTC/USD"  # replace with your desired trading pair
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
    """Place an order on Kraken."""
    symbol = "BTC/USD"  # replace with your desired trading pair

    # Define the order
    order = {
        "symbol": symbol,
        "type": order_type,
        "side": "buy" if order_type == "limit" else "sell",
        "price": price,
        "amount": amount,
    }

    # Place the order
    result = kraken.create_order(**order)

    return result


def main():
    """Main trading bot function."""
    # Fetch the data
    df = fetch_data()

    # Check the latest Supertrend and slope
    latest = df.iloc[-1]
    if latest["in_uptrend"] and latest["slope"] > 1:
        # Place a buy order
        place_order("limit", latest["Close"], 0.01)  # replace with your desired amount
    elif not latest["in_uptrend"] and latest["slope"] < 0:
        # Place a sell order
        place_order("market", latest["Close"], 0.01)  # replace with your desired amount


if __name__ == "__main__":
    main()
