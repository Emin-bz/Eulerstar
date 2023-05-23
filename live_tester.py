import json
import os
import numpy as np
import fetch
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

_start = "2021-03-10"
_end = "2023-02-13"

data = fetch.load(_start, _end, 'live', 50000)

opened_at = [None, None]
dirname, fname = os.path.split(__file__)
opened_at_path = os.path.join(dirname, 'opened_at.json')

with open(opened_at_path, 'r') as r:
  opened_at = json.load(r)

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

res = ""

linReg = LogisticRegression(random_state=0)
data = supertrend(data, period=10, atr_multiplier=3)
data = rsi(data, n=8)

n = len(data['Close']) - 1
trace = n - 20

trail = 4
lookup_aftertime = 40000
training_range = 1
testing_range = 1 - training_range
expected_profit = 0.03
d_count = 0

def determine_side_from(idx, momentum_type):
  if momentum_type == "upcoming":
    for idy in range(idx, lookup_aftertime):
      #print(d_count)
      if data['Close'][idy] > (1+expected_profit) * data['Close'][idx]:
        print(d_count)
        return "long"

  elif momentum_type == "downgoing":
    for idy in range(idx, lookup_aftertime):
      if data['Close'][idy] < (1-expected_profit) * data['Close'][idx]:
        print(d_count)
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

def get_rsi_conclusions():
  res = {}
  print("Doing...")
  for idx in range(trail, len(data['Close'])):
    if idx+lookup_aftertime == len(data['Close']):
      print("Broke at: ", str(idx))
      break

    if data['rsi'][idx] > 30 and data['rsi'][idx] < 70:
      continue
    
    rsi_trail = list(data["rsi"][idx-trail:idx+1])
    res[idx] = {}
    res[idx]["derivs"] = []
    res[idx]["side"] = "neither" # Either "long", "short" or "neither"

    for idy in range(1, len(rsi_trail)):
      deriv = rsi_trail[idy] - rsi_trail[idy-1]
      res[idx]["derivs"].append(deriv)
    
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

train()

if opened_at[0] == None and opened_at[1] == None:
  if detect_w_pattern_second_bottom(data['Close'][trace:n + 1]):
    opened_at = [str(data['Datetime'][n]), data['Close'][n]]
    with open(opened_at_path, 'w') as f:
      json.dump(opened_at, f)
    print(f"Opened at {opened_at[0]}, price {opened_at[1]}.")

elif opened_at[0] != None and opened_at[1] != None:
  if data['Close'][n] >= opened_at[1] * 1.004:
    res = f"Opened: {opened_at[0]}, {opened_at[1]}, Closed: {data['Datetime'][n]} {data['Close'][n]}"
    opened_at = [None, None]
    with open(opened_at_path, 'w') as f:
      json.dump(opened_at, f)
    print(res)
