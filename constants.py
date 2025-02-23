import pandas as pd
import os

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

def get_tickers(url, filename, index=0):
    if filename in os.listdir("."):
        return
    table = pd.read_html(url)
    df = table[index]
    tickers = df["Symbol"]
    tickers.to_csv(filename, index=False)

def get_spy500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    filename = "sp500_tickers.csv"
    get_tickers(url, filename)

def get_russell1000():
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    filename = "russell1000_tickers.csv"
    get_tickers(url, filename, index=3)

get_spy500()
get_russell1000()
SPY_500 = pd.read_csv("sp500_tickers.csv")["Symbol"].tolist()
RUSSELL_1000 = pd.read_csv("russell1000_tickers.csv")["Symbol"].tolist()

STOCK_CLUSTER_NAMES_DEPTH_1 = {
    0: "Consumer and Financial Services",
    1: "Utilities and Consumer Essentials",
    2: "Technology and Industrial Innovation",
    3: "Energy and Advanced Manufacturing",
    4: "Consumer Goods and Services",
    5: "Entertainment and Real Estate"
}

STOCK_CLUSTER_NAMES_DEPTH_2 = {
    (0, 0): "Consumer Discretionary and Financial Giants",
    (0, 1): "Technology, Retail, and Industrial Mix",
    (0, 2): "Semiconductors and Healthcare Services",
    (0, 3): "Financials and Industrial Equipment",
    (0, 4): "Consumer Goods and Financial Services Diversity",

    (1, 0): "Utilities and Essential Consumer Products",
    (1, 1): "Utilities and Real Estate Investment Trusts (REITs)",
    (1, 2): "Retail, Healthcare, and Consumer Products",

    (2, 0): "Tech and Industrial Innovators",
    (2, 1): "Energy and Natural Resources",
    (2, 2): "Technology Solutions and Energy Sector",
    (2, 3): "Emerging Tech and Automotive Leaders",
    (2, 4): "Biotechnology Pioneer",

    (3, 0): "Energy and Manufacturing Leaders",
    (3, 1): "Financial Services and Technological Advancements",
    (3, 2): "Industrial Leaders and High-Tech Innovators",
    (3, 3): "Diverse Manufacturing and Tech Conglomerates",

    (4, 0): "Consumer Goods and Retail Services",
    (4, 1): "Financial Services and Utilities",
    (4, 2): "Consumer and Industrial Goods",
    (4, 3): "Comprehensive Financial and Consumer Services",
    (4, 4): "Specialized Technology and Healthcare",

    (5, 0): "Entertainment and Real Estate Holdings",
    (5, 1): "Diverse Real Estate and Consumer Services",
    (5, 2): "Cruise Line Operators",
    (5, 3): "Healthcare and Travel Services"
}