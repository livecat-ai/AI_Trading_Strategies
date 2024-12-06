import pandas as pd
import yfinance as yf

def create_historical_prices(start_date='2000-01-01', end_date='2024-11-01'):
    MIN_REQUIRED_OBS_PER_TICKER = 100

    # Get S&P 500 tickers from Wikipedia
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0].Symbol.to_list()

    # Filter out the B shares
    sp500_tickers = [ticker for ticker in sp500_tickers if '.B' not in ticker]

    # Download historical prices for Yahoo Finance
    prices = yf.download(sp500_tickers, start=start_date, end=end_date)

    # Filter out the Adj Close column
    prices = prices.loc[:, prices.columns.get_level_values(0) == 'Adj Close']

    # Remove the multi index
    prices.columns = prices.columns.droplevel(0)

    # Get the names of all the tickers with more than the min required observations
    ticker_counts = prices.count()
    valid_ticker_mask = ticker_counts[ticker_counts >= MIN_REQUIRED_OBS_PER_TICKER].index

    # Filter the prices using the mask
    prices = prices[valid_ticker_mask]

    return prices

    