import fetch
import warnings
warnings.filterwarnings('ignore')

_start = "2023-02-10"
_end = "2023-02-13"

data = fetch.load(_start, _end, 'live', 100)

opened_at = None
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

if detect_w_pattern_second_bottom(data['Close'][trace:n + 1]):
  opened_at = (data['Datetime'][n], data['Close'][n])
  print(f"Opened at {opened_at[0]}, price {opened_at[1]}.")

elif opened_at != None:
  if data['Close'][n] >= opened_at[1] * 1.004:
    res = f"Opened: {opened_at[0]}, {opened_at[1]}, Closed: {data['Datetime'][n]} {data['Close'][n]}"
    opened_at = None
    print(res)
