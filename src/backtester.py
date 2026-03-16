import numpy as np


def backtest(prices, signals, initial_capital=10000.0, transaction_cost=0.0005, slippage=0.0005):
    capital = float(initial_capital)
    position = 0.0

    for price, signal in zip(prices, signals):
        exec_price = price * (1 + slippage) if signal == "BUY" else price * (1 - slippage)

        if signal == "BUY" and position == 0:
            units = capital / exec_price
            cost = capital * transaction_cost
            capital = 0.0 - cost
            position = units

        elif signal == "SELL" and position > 0:
            proceeds = position * exec_price
            cost = proceeds * transaction_cost
            capital = proceeds - cost
            position = 0.0

    if position > 0 and len(prices) > 0:
        final_price = prices[-1] * (1 - slippage)
        proceeds = position * final_price
        capital = proceeds - proceeds * transaction_cost

    return float(capital)


def _sharpe_ratio(returns):
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def _sortino_ratio(returns):
    returns = np.asarray(returns, dtype=float)
    downside = returns[returns < 0]
    if len(returns) < 2 or len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / downside.std())


def backtest_with_stats(prices, signals, initial_capital=10000.0, transaction_cost=0.0005, slippage=0.0005):
    capital = float(initial_capital)
    position_qty = 0.0
    entry_price = None
    trades = 0
    wins = 0

    equity_curve = []
    returns = []
    prev_equity = capital
    peak = capital
    max_drawdown = 0.0

    for price, signal in zip(prices, signals):
        buy_price = price * (1 + slippage)
        sell_price = price * (1 - slippage)

        if signal == "BUY" and position_qty == 0:
            position_qty = capital / buy_price
            entry_price = buy_price
            cost = capital * transaction_cost
            capital = -cost

        elif signal == "SELL" and position_qty > 0:
            exit_value = position_qty * sell_price
            cost = exit_value * transaction_cost
            if entry_price is not None and sell_price > entry_price:
                wins += 1
            capital += exit_value - cost
            position_qty = 0.0
            entry_price = None
            trades += 1

        equity = capital + position_qty * sell_price
        equity_curve.append(equity)
        r = (equity - prev_equity) / max(prev_equity, 1e-9)
        returns.append(r)
        prev_equity = equity

        peak = max(peak, equity)
        dd = (peak - equity) / max(peak, 1e-9)
        max_drawdown = max(max_drawdown, dd)

    if position_qty > 0 and len(prices) > 0:
        final_price = prices[-1] * (1 - slippage)
        if entry_price is not None and final_price > entry_price:
            wins += 1
        capital = capital + position_qty * final_price
        trades += 1

    final_capital = float(capital)
    win_rate = (wins / trades * 100.0) if trades else 0.0

    return {
        "final_capital": final_capital,
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown),
        "sharpe": _sharpe_ratio(returns),
        "sortino": _sortino_ratio(returns),
    }


def walk_forward_backtest(prices, signal_fn, train_window=252, test_window=63, **kwargs):
    """
    signal_fn(train_prices, test_prices) -> list[str] for the test slice
    """
    prices = list(prices)
    all_signals = ["HOLD"] * len(prices)

    start = train_window
    while start < len(prices):
        end = min(start + test_window, len(prices))
        train_slice = prices[start - train_window : start]
        test_slice = prices[start:end]
        test_signals = signal_fn(train_slice, test_slice)

        for i, s in enumerate(test_signals):
            if start + i < len(all_signals):
                all_signals[start + i] = s

        start = end

    return backtest_with_stats(prices, all_signals, **kwargs)
