import numpy as np

def calculate_rsi(df, n=8):
  prices = []

  for c in df['Close']:
      prices.append(c)

  if len(prices) == 0:
      return
  
  deltas = np.diff(prices)
  seed = deltas[:n+1]
  up = seed[seed >= 0].sum()/n
  down = -seed[seed < 0].sum()/n
  rs = up/down
  rsi = np.zeros_like(prices)
  rsi[:n] = 100. - 100./(1.+rs)

  for i in range(n, len(prices)):
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

  df['rsi'] = rsi

  return df