import yfinance as yf


def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)


# use the function
print(get_stock_price('AAPL'))
print(get_stock_price('GOOG'))
