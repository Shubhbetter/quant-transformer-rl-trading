import yfinance as yf

def load_data(tickers):

    data = {}

    for ticker in tickers:

        df = yf.download(ticker, start="2014-01-01")

        data[ticker] = df

    return data
