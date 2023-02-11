import auth
import pandas as pd
import yfinance as yf
import supertrend
import rsi

polygon_api_key = "IiYvpxn8CCkNSR5gYFj3NgkRLZvL4YyF"

PAIRNAME = 'SOLUSD'
TIMEFRAME = '5'
UNIT = 'minute'

def load(start, end, mode):
  raw_bars = []
  bars = []

  if mode == 'yfinance':
    df = yf.download("TSLA", start=start, end=end, interval="1m")
  
  else:
    if mode == 'polygon':
      ENDPOINT_URL = f'https://api.polygon.io/v2/aggs/ticker/X:{PAIRNAME}/range/{TIMEFRAME}/{UNIT}/{start}/{end}?adjusted=false&sort=asc&limit=50000&apiKey=' + polygon_api_key
      raw_bars = auth.authenticate('GET', ENDPOINT_URL)['results']

      for entry in raw_bars:
        bars.append([entry['t'], entry['o'], entry['h'], entry['l'], entry['c']])
    
    elif mode == 'local':
      with open('chart.txt', 'r') as readfile:
        for line in readfile:
            raw_bars.append(line.rstrip().split(', '))
      
      for entry in raw_bars:
        ts = int(entry[0])
        o = float(entry[1])
        high = float(entry[2])
        low = float(entry[3])
        close = float(entry[4])

        bars.append([ts, o, high, low, close])

    df = pd.DataFrame(bars[:], columns=['Datetime', 'Open', 'High', 'Low', 'Close'])
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')

  df = supertrend.calculate_supertrend(df, 10, 3)
  df = rsi.calculate_rsi(df, 14)
  
  df['type'] = mode

  return df