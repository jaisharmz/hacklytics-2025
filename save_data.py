import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from constants import IMPORTANT_TICKERS_36, SPY_500, RUSSELL_1000

df = yf.download(SPY_500, period='5y')
open_columns = [("Open", ticker) for ticker in SPY_500]
df = df.loc[:,open_columns]
df.to_csv("SPY_500_5y.csv")

df = yf.download(RUSSELL_1000, period='5y')
open_columns = [("Open", ticker) for ticker in RUSSELL_1000]
df = df.loc[:,open_columns]
df.to_csv("RUSSELL_1000_5y.csv")