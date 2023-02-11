import talib
import fetch
import warnings
warnings.filterwarnings('ignore')

_start = "2022-04-01"
_end = "2023-02-11"

data = fetch.load(_start, _end, 'polygon', 50000)

morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])

data['Morning Star'] = morning_star
data['Engulfing'] = engulfing

opened_at = None
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
  
trace = 0

for idx in range(20, len(data['Open'])):
  if detect_w_pattern_second_bottom(data['Open'][trace:idx]):
    opened_at = (data['Datetime'][idx] if data['type'][idx] == 'polygon' else data.index[idx], data['Open'][idx])

  elif opened_at != None:
    if data['Open'][idx] >= opened_at[1] * 1.004 or (data['rsi'][idx] > 80 and opened_at[1] <= data['Open'][idx]):
      res = f"Opened: {opened_at[0]}, {opened_at[1]}, Closed: {data['Datetime'][idx] if data['type'][idx] == 'polygon' else data.index[idx]} {data['Open'][idx]}"
      print(res)
      calc_profit(opened_at[1], data['Open'][idx])
      opened_at = None
  
  trace += 1

if opened_at != None:
  res = f"Opened: {opened_at[0]}, {opened_at[1]}"
  print(res)

print('\n')
print(f'Total profit ({_start} - {_end}):', profit, '$ USD')