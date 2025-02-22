import pandas as pd

IMPORTANT_TICKERS_36 = [
    # Major Indices
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones Industrial Average
    "^IXIC",  # Nasdaq Composite
    "^RUT",   # Russell 2000
    "^VIX",   # Volatility Index

    # Tech Giants
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet (Google)
    "AMZN",  # Amazon
    "NVDA",  # NVIDIA
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "AMD",   # Advanced Micro Devices

    # Financials
    "JPM",   # JPMorgan Chase
    "BAC",   # Bank of America
    "GS",    # Goldman Sachs
    "BRK.B", # Berkshire Hathaway

    # Energy
    "XOM",   # ExxonMobil
    "CVX",   # Chevron

    # Consumer Goods
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo

    # Healthcare
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer
    "UNH",   # UnitedHealth Group

    # Industrials
    "BA",    # Boeing
    "CAT",   # Caterpillar

    # Retail & E-commerce
    "WMT",   # Walmart
    "COST",  # Costco
    "TGT",   # Target

    # Semiconductors
    "TSM",   # Taiwan Semiconductor
    "INTC",  # Intel

    # Entertainment & Streaming
    "NFLX",  # Netflix
    "DIS",   # Disney

    # EV & Automotive
    "F",     # Ford
    "GM",    # General Motors
]

def get_tickers(url, filename):
    table = pd.read_html(url)
    df = table[0]
    tickers = df["Symbol"]
    tickers.to_csv(filename, index=False)

def get_syp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    filename = "sp500_tickers.csv"
    get_tickers(url, filename)

def get_russell1000():
    url = "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies"
    filename = "russell1000_tickers.csv"
    get_tickers(url, filename)

get_syp500()
get_russell1000()
SYP_500 = pd.read_csv("sp500_tickers.csv")["Symbol"].tolist()
RUSSELL_1000 = pd.read_csv("sp500_tickers.csv")["Symbol"].tolist()