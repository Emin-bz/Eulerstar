from datetime import datetime as dt
import numpy as np
import fetch
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

_start = "2023-01-01"
_end = "2023-05-19"

data = fetch.load(_start, _end, 'polygon', 50000)
# morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
# engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])

# data['Morning Star'] = morning_star
# data['Engulfing'] = engulfing

Opened_at = None
res = ""
capital = 1000
profit =- 0

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

def rsi(df, n=14):
  if len(df['Close']) == 0:
      return
  
  deltas = np.diff(df['Close'])
  seed = deltas[:n+1]
  up = seed[seed >= 0].sum()/n
  down = -seed[seed < 0].sum()/n
  rs = up/down
  rsi = np.zeros_like(df['Close'])
  rsi[:n] = 100. - 100./(1.+rs)

  for i in range(n, len(df['Close'])):
      delta = deltas[i-1]  # The diff is 1 shorter

      if delta > 0:
          upval = delta
          downval = 0.
      else:
          upval = 0.
          downval = -delta

      up = (up*(n-1) + upval)/n
      down = (down*(n-1) + downval)/n

      rs = up/down
      rsi[i] = 100. - 100./(1.+rs)

  df["rsi"] = rsi
  return df

def tr(data):
  data['previous_Close'] = data['Close'].shift(1)
  data['High-Low'] = abs(data['High'] - data['Low'])
  data['High-pc'] = abs(data['High'] - data['previous_Close'])
  data['Low-pc'] = abs(data['Low'] - data['previous_Close'])

  tr = data[['High-Low', 'High-pc', 'Low-pc']].max(axis=1)

  return tr

def atr(data, period):
  data['tr'] = tr(data)
  atr = data['tr'].rolling(period).mean()

  return atr

def supertrend(df, period=10, atr_multiplier=13):
    hl2 = (df['High'] + df['Low']) / 2
    df['atr'] = atr(df, period)
    df['upperband'] = hl2 + (atr_multiplier * df['atr'])
    df['Lowerband'] = hl2 - (atr_multiplier * df['atr'])
    df['in_uptrend'] = True

    for current in range(1, len(df.index)):
        previous = current - 1

        if df['Close'][current] > df['upperband'][previous]:
            df['in_uptrend'][current] = True
        elif df['Close'][current] < df['Lowerband'][previous]:
            df['in_uptrend'][current] = False
        else:
            df['in_uptrend'][current] = df['in_uptrend'][previous]

            if df['in_uptrend'][current] and df['Lowerband'][current] < df['Lowerband'][previous]:
                df['Lowerband'][current] = df['Lowerband'][previous]

            if not df['in_uptrend'][current] and df['upperband'][current] > df['upperband'][previous]:
                df['upperband'][current] = df['upperband'][previous]
        
    return df

#linReg = LogisticRegression(random_state=0)

trail = 3
lookup_aftertime = 20000
training_range = 1
testing_range = 1 - training_range
expected_profit = 0.2
d_count = 0

def determine_side_from(idx, momentum_type):
  if momentum_type == "upcoming":
    for idy in range(idx, lookup_aftertime):
      if data['Close'][idy] > (1+expected_profit) * data['Close'][idx]:
        return "long"

  elif momentum_type == "downgoing":
    for idy in range(idx, lookup_aftertime):
      if data['Close'][idy] < (1-expected_profit) * data['Close'][idx]:
        return "short"

  return "neither"

def is_long_curve(derivs):
  for i in range(1, len(derivs)):
    if derivs[i] < derivs[i-1] or derivs[i] == derivs[i-1]:
      return False
  return True

def is_short_curve(derivs):
  for i in range(1, len(derivs)):
    if derivs[i] > derivs[i-1] or derivs[i] == derivs[i-1]:
      return False
  return True

def get_trail(idx) -> list:
  return list(data["rsi"][idx-trail:idx+1])

def get_derivations(t: list):
  derivs = []
  
  for idy in range(1, len(t)):
    #print(idy)
    deriv = t[idy] - t[idy-1]
    derivs.append(deriv)
  
  return derivs


def get_rsi_conclusions():
  res = {}

  for idx in range(trail, len(data['Close'])):
    if idx+lookup_aftertime == len(data['Close']):
      break

    if data['rsi'][idx] > 30 and data['rsi'][idx] < 70:
      continue
    
    rsi_trail = get_trail(idx)
    res[idx] = {}
    res[idx]["derivs"] = get_derivations(rsi_trail)
    res[idx]["side"] = "neither"
    
    momentum_type = None
    global d_count
    d_count += 1

    if is_long_curve(res[idx]["derivs"]):
      momentum_type = "upcoming"
    elif is_short_curve(res[idx]["derivs"]):
      momentum_type = "downgoing"

    if momentum_type is not None:
      res[idx]["side"] = determine_side_from(
        idx=idx,
        momentum_type=momentum_type
      )
    #print("Ongoing: ", str(idx))
  
  # Elimination
  del_keys = []
  
  for k, v in res.items():
    if v["side"] == "neither":
      del_keys.append(k)
  
  for dk in del_keys:
    del res[dk]
  
  return res

def split_training_data():
  rsi_derivatives = get_rsi_conclusions()

  training_sample_size = round(len(rsi_derivatives) * training_range)

  x_train = []
  y_train = []

  x_test = []
  y_test = []

  count = 1
  for k, v in rsi_derivatives.items():
    if count <= training_sample_size:
      x_train.append(np.array(v["derivs"]))
      y_train.append(np.array(v["side"]))
    else:
      x_test.append(np.array(v["derivs"]))
      y_test.append(np.array(v["side"]))
    count+=1
  
  return { 'x_train': np.array(x_train), 'y_train': np.array(y_train), 'x_test': np.array(x_test), 'y_test': np.array(y_test) }

def train():
  x_train = split_training_data()['x_train']
  y_train = split_training_data()['y_train']
  linReg.fit(x_train, y_train)

  x_test = split_training_data()['x_test']
  # y_pred = linReg.predict(x_test)
  # print(y_pred)

  # y_test = split_training_data()['y_test']
  # accuracy = accuracy_score(y_test, y_pred)
  # print(accuracy)

#train()

def get_2d_derivs(conc):
  return np.array([conc["derivs"]])

def moving_average_crossover_strategy(data, short_period, long_period):
  # Calculate the short-term and long-term moving averages
  data['short_ma'] = data['Close'].rolling(window=short_period).mean()
  data['long_ma'] = data['Close'].rolling(window=long_period).mean()
  return data

data = moving_average_crossover_strategy(data, 10, 12)
data = rsi(data, n=8)
data = supertrend(data, period=10, atr_multiplier=1)

OPEN_TIME = 13.0
OPEN_THRESHOLD = 5
CLOSE_THRESHOLD = 10.0

def get_float_time_from(date: str) -> str:
  return float(str(date).split(" ")[1][0] + str(date).split(" ")[1][1] + "." + str(date).split(" ")[1][3] + str(date).split(" ")[1][4])

def positive_deriv_history(derivs):
  for i in range(1, len(derivs)):
    if list(derivs)[i] - list(derivs)[i-1] < 0:
      return False
  return True

def negative_deriv_history(derivs):
  for i in range(1, len(derivs)):
    if list(derivs)[i] - list(derivs)[i-1] > 0:
      return False
  return True

for idx in range(10, len(data['Close'])):
  idx_date = data['Datetime'][idx]
  idx_uptrend = data['in_uptrend'][idx]
  closes = data['Close']
  deriv_range = 5

  if Opened_at == None:
    #curr_derivs = [get_derivations(get_trail(idx))]
    if get_float_time_from(idx_date) >= OPEN_TIME and get_float_time_from(idx_date) <= OPEN_TIME + OPEN_THRESHOLD:
      if positive_deriv_history(closes[idx-deriv_range:idx+1]) and idx_uptrend:
        Opened_at = (data['Datetime'][idx], data['Close'][idx], "long")
      elif negative_deriv_history(closes[idx-deriv_range:idx+1]) and not idx_uptrend:
        Opened_at = (data['Datetime'][idx], data['Close'][idx], "short")

  elif Opened_at != None:
    if Opened_at[2] == "long":
      if data['Close'][idx] >= Opened_at[1] * 1.01:
        res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Long"
        print(res)
        calc_profit(Opened_at[1], data['Close'][idx])
        Opened_at = None
      # TODO fix elif cuz time is repeating so it doesn't exit properly, make it depend on subtraction on full dates
      elif get_float_time_from(idx_date) > OPEN_TIME + CLOSE_THRESHOLD:
        res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Long"
        print(res)
        calc_profit(Opened_at[1], data['Close'][idx])
        Opened_at = None
      # elif data['rsi'][idx] >= 70 and not data['in_uptrend'][idx]:
      #   res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Long (Emergency)"
      #   print(res)
      #   calc_profit(Opened_at[1], data['Close'][idx])
      #   Opened_at = None
    elif Opened_at[2] == "short":
      if data['Close'][idx] <= Opened_at[1] * 0.99:
        res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Short"
        print(res)
        calc_profit(data['Close'][idx], Opened_at[1])
        Opened_at = None
      # TODO fix elif cuz time is repeating so it doesn't exit properly, make it depend on subtraction on full dates
      elif get_float_time_from(idx_date) > OPEN_TIME + CLOSE_THRESHOLD:
        res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Short"
        print(res)
        calc_profit(data['Close'][idx], Opened_at[1])
        Opened_at = None
      # elif data['rsi'][idx] <= 30 and data['in_uptrend'][idx]:
      #   res = f"Opened: {Opened_at[0]}, {Opened_at[1]}, Closed: {data['Datetime'][idx]} {data['Close'][idx]}, Short (Emergency)"
      #   print(res)
      #   calc_profit(data['Close'][idx], Opened_at[1])
      #   Opened_at = None

if Opened_at != None:
  res = f"Opened: {Opened_at[0]}, {Opened_at[1]}"
  print(res)

print('\n')
print(f'Total profit ({_start}-{_end}): {profit}$ USD ({float((profit / capital) * 100)}%)')
