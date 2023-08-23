from datetime import datetime
import numpy as np
import pandas as pd

import crypto_fetcher
import warnings

warnings.filterwarnings("ignore")

def calc_profit(start, end):
    global profit

    diff = 1 - (start / end)
    profit += capital * diff
    fee = ((capital / 100) * entry_exit_fee) * 2
    profit -= fee

def calculate_supertrend(df, period=7, multiplier=3):
    """Calculate Supertrend given a pandas dataframe of OHLC data."""
    hl_avg = (df["High"] + df["Low"]) / 2
    atr = pd.DataFrame.ewm(hl_avg - hl_avg.shift(), span=period).mean()

    df["upper_band"] = hl_avg + (multiplier * atr)
    df["lower_band"] = hl_avg - (multiplier * atr)
    df["in_uptrend"] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df["Open"][current] > df["upper_band"][previous]:
            df.at[current, "in_uptrend"] = True
        elif df["Open"][current] < df["lower_band"][previous]:
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

def update_general_meta_data(closing_type: str) -> None:
    global trade_meta_infos
    global opened_at

    trade_meta_infos[opened_at[0]]["closing_type"] = closing_type

def get_trade_duration(start: str, end: str) -> float:
    # Convert the input strings to datetime objects
    start_datetime = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')

    # Calculate the time difference
    time_difference = end_datetime - start_datetime

    # Calculate the difference in hours
    return time_difference.total_seconds() / 3600

def update_trade_duration(curr_date) -> None:
    global trade_meta_infos
    global opened_at

    hours_difference = get_trade_duration(opened_at[0], str(curr_date))

    trade_meta_infos[opened_at[0]]["duration"] = hours_difference

def update_drawback(curr_price) -> None:
    global trade_meta_infos
    global opened_at

    curr_drawback = ((curr_price - opened_at[1]) / opened_at[1]) * 100

    if curr_drawback > trade_meta_infos[opened_at[0]]["drawback"]:
        return
    
    trade_meta_infos[opened_at[0]]["drawback"] = curr_drawback

_start = "2023-05-01"
_end = "2023-08-23"

df = crypto_fetcher.load(_start, _end, "polygon", 50000)

opened_at = None
trade_meta_infos = {}

res = ""
capital = 10000
profit = 0
entry_exit_fee = 0.05
MAX_TRADE_DURATION_IN_HOURS = 3
expected_profit_rate = 1.003
stop_loss_rate = 0.998

df = calculate_supertrend(df, period=7, multiplier=3)
df["slope"] = calculate_slope(
    df["Open"], period=5
)  # or df['upper_band'] or df['lower_band']

trace = 0

for idx in range(20, len(df["Open"])):
    if opened_at is None:
        if df["in_uptrend"][idx] and df["slope"][idx] > 1:
            opened_at = (
                str(df["Datetime"][idx]) if df["type"][idx] == "polygon" else df.index[idx],
                df["Close"][idx],
            )
            trade_meta_infos[str(df["Datetime"][idx])] = {}
            trade_meta_infos[str(df["Datetime"][idx])]["drawback"] = 0
            trade_meta_infos[str(df["Datetime"][idx])]["duration"] = 0
            trade_meta_infos[str(df["Datetime"][idx])]["closing_type"] = "not_closed"


    elif opened_at is not None:
        update_drawback(df["Close"][idx])

        is_stop_loss_hit = (
            (df["Close"][idx] <= opened_at[1] * stop_loss_rate or df["Low"][idx] <= opened_at[1] * stop_loss_rate)
            and get_trade_duration(opened_at[0], str(df["Datetime"][idx])) >= MAX_TRADE_DURATION_IN_HOURS
        )

        if df["Close"][idx] >= opened_at[1] * expected_profit_rate or is_stop_loss_hit:
            closing_type = "normal"
            if is_stop_loss_hit:
                closing_type = "emergency"

            res = f"Opened: {opened_at[0]}, {opened_at[1]}, Closed: {df['Datetime'][idx] if df['type'][idx] == 'polygon' else df.index[idx]} {df['Open'][idx]}, {'(' + closing_type + ')' if closing_type =='emergency' else ''}"
            print(res)
            calc_profit(opened_at[1],
                        (
                            df["Close"][idx] if closing_type == "normal" else
                            opened_at[1] * stop_loss_rate
                            if opened_at[1] * stop_loss_rate <= df["Close"][idx] else df["Close"][idx]
                        )
                    )
            update_trade_duration(df["Datetime"][idx])
            update_general_meta_data(closing_type)
            opened_at = None

    trace += 1

if opened_at != None:
    res = f"Opened: {opened_at[0]}, {opened_at[1]}"
    print(res)

print(f"\nTotal profit ({_start} - {_end}):", profit, "$ USD")
print("Winning Avg. drawback:", sum([trade_meta_infos[m]["drawback"] for m in trade_meta_infos if trade_meta_infos[m]["closing_type"] == "normal"]) / len(trade_meta_infos))
print("Winning Avg. duration:", sum([trade_meta_infos[m]["duration"] for m in trade_meta_infos if trade_meta_infos[m]["closing_type"] == "normal"]) / len(trade_meta_infos))
print("Normal : Emergency Ratio:", sum([1 for m in trade_meta_infos if trade_meta_infos[m]["closing_type"] == "normal"]), ":", sum([1 for m in trade_meta_infos if trade_meta_infos[m]["closing_type"] == "emergency"]))
