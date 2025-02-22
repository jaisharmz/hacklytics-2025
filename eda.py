# Starting out with stock data from Yahoo finance. 
# This should eventually change to the price over time of 
# products like the iPhone or GPUs as well as other derivatives
# like futures. 

import yfinance as yf
import matplotlib.pyplot as plt
from constants import IMPORTANT_TICKERS_36, SPY_500, RUSSELL_1000

# interesting_tickers = ["MSFT", "AAPL", "GOOG", "NVDA", "AMD", "TSLA"]
interesting_tickers = SPY_500
df = yf.download(interesting_tickers, period='5y')
df = df / df.iloc[0]
print(df.shape)

open_columns = [("Open", ticker) for ticker in interesting_tickers][:10]
df.loc[:,open_columns].plot()
plt.show()
df.loc[:,open_columns].rolling(10).mean().plot()
plt.show()