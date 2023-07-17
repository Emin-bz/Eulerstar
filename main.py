import talib
import fetch
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")

_start = "2023-01-01"
_end = "2023-07-18"

df = fetch.load(_start, _end, "polygon", 50000)

morning_star = talib.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])
engulfing = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])

df["Morning Star"] = morning_star
df["Engulfing"] = engulfing

opened_at = None
res = ""
capital = 1000
profit = -0


def calc_profit(start, end):
    global profit

    diff = 1 - (start / end)
    profit += capital * diff
    fee = ((capital / 100) * 0.1) * 2
    profit -= fee


def calc_loss(start, end):
    global profit

    diff = 1 - (end / start)
    profit -= capital * diff
    fee = ((capital / 100) * 0.1) * 2
    profit -= fee


import numpy as np
import pandas as pd


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


df = calculate_supertrend(df, period=7, multiplier=3)
df["slope"] = calculate_slope(
    df["Close"], period=5
)  # or df['upper_band'] or df['lower_band']

trace = 0

for idx in range(20, len(df["Open"])):
    if opened_at is None:
        if df["in_uptrend"][idx] and df["slope"][idx] > 1:
            opened_at = (
                df["Datetime"][idx] if df["type"][idx] == "polygon" else df.index[idx],
                df["Open"][idx],
            )

    elif opened_at is not None:
        if not df["in_uptrend"][idx] and df["slope"][idx] < 0:
            res = f"Opened: {opened_at[0]}, {opened_at[1]}, Closed: {df['Datetime'][idx] if df['type'][idx] == 'polygon' else df.index[idx]} {df['Open'][idx]}"
            print(res)
            calc_profit(opened_at[1], df["Open"][idx])
            opened_at = None

    trace += 1

if opened_at != None:
    res = f"Opened: {opened_at[0]}, {opened_at[1]}"
    print(res)

print("\n")
print(f"Total profit ({_start} - {_end}):", profit, "$ USD")
