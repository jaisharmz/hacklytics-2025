import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from constants import IMPORTANT_TICKERS_36, SYP_500, RUSSELL_1000

df = yf.download(SYP_500, period='5y')
open_columns = [("Open", ticker) for ticker in SYP_500]
df = df.loc[:,open_columns]
df.to_csv("SYP_500_5y.csv")

df = yf.download(RUSSELL_1000, period='5y')
open_columns = [("Open", ticker) for ticker in RUSSELL_1000]
df = df.loc[:,open_columns]
df.to_csv("RUSSELL_1000_5y.csv")