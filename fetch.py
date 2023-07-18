import auth
import pandas as pd
import yfinance as yf

polygon_api_key = "IiYvpxn8CCkNSR5gYFj3NgkRLZvL4YyF"

PAIRNAME = "BTCUSD"
TIMEFRAME = "5"
UNIT = "minute"


def load(start, end, mode, limit):
    raw_bars = []
    bars = []

    if mode == "yfinance":
        df = yf.download("TSLA", start=start, end=end, interval="1m")

    else:
        if mode == "polygon":
            ENDPOINT_URL = (
                f"https://api.polygon.io/v2/aggs/ticker/X:{PAIRNAME}/range/{TIMEFRAME}/{UNIT}/{start}/{end}?adjusted=false&sort=asc&limit={str(limit)}&apiKey="
                + polygon_api_key
            )
            raw_bars = auth.authenticate("GET", ENDPOINT_URL)["results"]

            for entry in raw_bars:
                bars.append(
                    [entry["t"], entry["o"], entry["h"], entry["l"], entry["c"]]
                )
                # with open("chart.txt", "a") as w:
                #   w.write(f"{entry['t']}, {entry['o']}, {entry['h']}, {entry['l']}, {entry['c']}\n")

        elif mode == "local":
            with open("chart.txt", "r") as readfile:
                for line in readfile:
                    raw_bars.append(line.rstrip().split(", "))

            for entry in raw_bars:
                ts = int(entry[0])
                o = float(entry[1])
                high = float(entry[2])
                low = float(entry[3])
                close = float(entry[4])
                bars.append([ts, o, high, low, close])

        elif mode == "live":
            ENDPOINT_URL = f"https://api.kraken.com/0/public/OHLC?pair={PAIRNAME}"
            raw_bars = auth.authenticate("GET", ENDPOINT_URL)["result"][PAIRNAME]

            for entry in raw_bars:
                entry[0] = str(entry[0]) + "000"
                bars.append(
                    [
                        int(entry[0]),
                        float(entry[1]),
                        float(entry[2]),
                        float(entry[3]),
                        float(entry[4]),
                    ]
                )

        df = pd.DataFrame(bars[:], columns=["Datetime", "Open", "High", "Low", "Close"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")

    df["type"] = mode

    return df
