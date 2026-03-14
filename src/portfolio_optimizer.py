import numpy as np

def equal_weight_portfolio(stocks):

    weights = np.ones(len(stocks))

    weights = weights / len(stocks)

    return weights
