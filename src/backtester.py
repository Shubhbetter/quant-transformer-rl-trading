def backtest(prices, signals):

    capital = 10000
    position = 0

    for i in range(len(signals)):

        if signals[i] == "BUY":

            position = capital / prices[i]

        if signals[i] == "SELL":

            capital = position * prices[i]
            position = 0

    return capital
