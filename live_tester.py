import json
import os
import fetch
import warnings
warnings.filterwarnings('ignore')

_start = "2023-02-10"
_end = "2023-02-13"

data = fetch.load(_start, _end, 'live', 100)

opened_at = [None, None]
dirname, fname = os.path.split(__file__)
opened_at_path = os.path.join(dirname, 'opened_at.json')

with open(opened_at_path, 'r') as r:
  opened_at = json.load(r)

res = ""

def detect_w_pattern_second_bottom(p):
  prices = []

  for price in p:
    prices.append(price)
  
  n = len(prices)
  
  for i in range(n - 4):
      # Check if the first valley is found
      if prices[i] > prices[i + 1] and prices[i + 1] < prices[i + 2]:
          # Check if the first peak is found
          if prices[i + 2] > prices[i + 3] and prices[i + 3] > prices[i + 4]:
              # Check if the second valley is found
              for j in range(i + 4, n - 2):
                  if prices[j] > prices[j + 1] and prices[j + 1] < prices[j + 2]:
                    return True
  return False

n = len(data['Close']) - 1
trace = n - 20

if opened_at[0] == None and opened_at[1] == None:
  if detect_w_pattern_second_bottom(data['Close'][trace:n + 1]):
    opened_at = [data['Datetime'][n], data['Close'][n]]
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
