# Starting out with stock data from Yahoo finance. 
# This should eventually change to the price over time of 
# products like the iPhone or GPUs as well as other derivatives
# like futures. 

import yfinance as yf
import matplotlib.pyplot as plt

interesting_tickers = ["MSFT", "AAPL", "GOOG", "NVDA", "AMD", "TSLA"]
df = yf.download(interesting_tickers, period='5y')
print(df.shape)

open_columns = [("Open", ticker) for ticker in interesting_tickers]
df.loc[:,open_columns].plot()
plt.show()