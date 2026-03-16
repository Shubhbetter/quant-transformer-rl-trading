import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import torch
from stable_baselines3 import PPO

from src.backtester import backtest_with_stats, walk_forward_backtest
from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regime
from src.sentiment import sentiment_score
from src.transformer_model import add_transformer_predicted_return


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS"]
REGIME_MAP = {0: "Bullish", 1: "Sideways", 2: "Bearish"}


def build_features_for_ticker(raw_df, spy_df, qqq_df, vix_df, sentiment):
    df = create_features(raw_df.copy())
    spy = create_features(spy_df.copy())
    qqq = create_features(qqq_df.copy())
    vix = create_features(vix_df.copy())

    df["spy_trend_20"] = spy["Close"].pct_change(20).reindex(df.index).fillna(0.0)
    df["qqq_trend_20"] = qqq["Close"].pct_change(20).reindex(df.index).fillna(0.0)
    df["market_index_trend"] = ((df["spy_trend_20"] + df["qqq_trend_20"]) / 2.0).fillna(0.0)
    df["vix_trend_20"] = vix["Close"].pct_change(20).reindex(df.index).fillna(0.0)

    df = detect_regime(df)
    df["sentiment"] = sentiment
    df = add_transformer_predicted_return(df, price_col="Close", window=64, seed=SEED)

    # prediction pipeline safety + realism cap
    predicted_price = df["Close"] * (1.0 + df["predicted_return"])
    predicted_price = np.where(predicted_price > 0, predicted_price, df["Close"])
    cap = df["Close"] * 0.20
    delta = np.clip(predicted_price - df["Close"], -cap, cap)
    df["predicted_price"] = df["Close"] + delta

    # Volatility regime + volume confirmation
    atr_mean = df["atr"].rolling(60).mean()
    df["volatility_regime"] = np.where(df["atr"] > atr_mean, "high_volatility", "normal")
    df["vol_regime_flag"] = (df["volatility_regime"] == "high_volatility").astype(float)
    df["vol_regime"] = df["vol_regime_flag"]

    avg_vol_20 = df["Volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_spike_ratio"] = (df["Volume"] / avg_vol_20).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df.dropna(inplace=True)
    return df


def load_rl_model(path="models/ppo_trading_agent.zip"):
    if Path(path).exists():
        try:
            return PPO.load(path)
        except Exception:
            return None
    return None


def rl_action_vote(rl_model, latest_row):
    if rl_model is None:
        return 0.0

    state_cols = [
        "Close", "rsi", "macd", "volatility", "atr", "momentum_20", "ema_20", "ema_50", "ema_200",
        "bb_width", "adx", "volume_spike", "vwap", "regime", "sentiment", "predicted_return", "vol_regime",
        "market_index_trend", "vix_trend_20",
    ]
    obs = np.array([float(latest_row.get(c, 0.0)) for c in state_cols], dtype=np.float32)
    try:
        action, _ = rl_model.predict(obs, deterministic=True)
    except Exception:
        return 0.0

    mapping = {0: 0.0, 1: 0.5, 2: 1.0, 3: -0.5, 4: -1.0}
    return float(mapping.get(int(action), 0.0))


def market_index_filter(latest_row):
    spy_bull = latest_row["spy_trend_20"] > 0
    qqq_bull = latest_row["qqq_trend_20"] > 0
    spy_bear = latest_row["spy_trend_20"] < 0
    qqq_bear = latest_row["qqq_trend_20"] < 0

    allow_long = bool(spy_bull and qqq_bull)
    allow_short = bool(spy_bear and qqq_bear)
    return allow_long, allow_short


def component_score(latest_row, rl_vote):
    regime_label = REGIME_MAP.get(int(latest_row["regime"]), "Sideways")
    score = {
        "macd": 1.0 if latest_row["macd"] > 0 else -1.0,
        "rsi": 1.0 if latest_row["rsi"] < 35 else -1.0 if latest_row["rsi"] > 65 else 0.0,
        "momentum": 1.0 if latest_row["momentum_20"] > 0 else -1.0,
        "predicted_return": 1.0 if latest_row["predicted_return"] > 0 else -1.0,
        "market_index_trend": 1.0 if latest_row["market_index_trend"] > 0 else -1.0,
        "sentiment": 1.0 if latest_row["sentiment"] >= 0.1 else -1.0 if latest_row["sentiment"] <= -0.1 else 0.0,
        "ema_trend": 1.0 if latest_row["ema_50"] > latest_row["ema_200"] else -1.0,
        "adx_trend": 0.5 if latest_row["adx"] > 20 else 0.0,
        "vix": -0.5 if latest_row["vix_trend_20"] > 0 else 0.25,
        "regime": 0.75 if regime_label == "Bullish" else -0.75 if regime_label == "Bearish" else 0.0,
        "rl_policy": rl_vote,
        "volume_confirm": 0.5 if latest_row["volume_spike_ratio"] > 1.5 else 0.0,
    }
    return score


def probabilities_from_score(score_dict):
    total = float(sum(score_dict.values()))
    logits = np.array([total, 2.5 - min(abs(total), 2.5), -total], dtype=float)
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return {"BUY": float(p[0]), "HOLD": float(p[1]), "SELL": float(p[2]), "score": total}


def trade_plan(current_price, atr_value, signal, rr=2.0, atr_mult=1.5):
    stop_distance = max(float(atr_value) * atr_mult, current_price * 0.005)

    if signal == "BUY":
        stop_loss = current_price - stop_distance
        take_profit = current_price + rr * stop_distance
    elif signal == "SELL":
        stop_loss = current_price + stop_distance
        take_profit = current_price - rr * stop_distance
    else:
        stop_loss = current_price - stop_distance
        take_profit = current_price + stop_distance

    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    rr_val = reward / max(risk, 1e-9)
    return stop_loss, take_profit, risk, rr_val


def position_size(capital, risk_per_trade, stop_distance, price, volatility_regime):
    qty = (capital * risk_per_trade) / max(stop_distance, 1e-9)
    if volatility_regime == "high_volatility":
        qty *= 0.5
    cap_limited_qty = (capital * 0.20) / max(price, 1e-9)
    return float(min(qty, cap_limited_qty))


def apply_filters(raw_signal, confidence, risk_reward, allow_long, allow_short):
    if confidence <= 0.60:
        return "HOLD", "Confidence <= 60%"
    if risk_reward <= 1.5:
        return "HOLD", "Risk/Reward <= 1.5"
    if raw_signal == "BUY" and not allow_long:
        return "HOLD", "Index filter blocked longs"
    if raw_signal == "SELL" and not allow_short:
        return "HOLD", "Index filter blocked shorts"
    return raw_signal, "Passed filters"


def historical_signals(frame):
    sig = []
    for _, r in frame.iterrows():
        s = 0
        s += 1 if r["macd"] > 0 else -1
        s += 1 if r["rsi"] < 35 else -1 if r["rsi"] > 65 else 0
        s += 1 if r["momentum_20"] > 0 else -1
        s += 1 if r["predicted_return"] > 0 else -1
        s += 1 if r["market_index_trend"] > 0 else -1
        s += 1 if r["volume_spike_ratio"] > 1.5 else 0

        if s >= 3:
            sig.append("BUY")
        elif s <= -3:
            sig.append("SELL")
        else:
            sig.append("HOLD")
    return sig


def wf_signal_fn(train_slice, test_slice):
    mean_train = np.mean(np.diff(train_slice)) if len(train_slice) > 2 else 0.0
    return ["BUY" if mean_train > 0 else "SELL" if mean_train < 0 else "HOLD" for _ in test_slice]


def evaluate_ticker(ticker, raw_df, spy_df, qqq_df, vix_df, sentiment, rl_model, capital, risk_per_trade):
    frame = build_features_for_ticker(raw_df, spy_df, qqq_df, vix_df, sentiment)
    latest = frame.iloc[-1]

    rl_vote = rl_action_vote(rl_model, latest)
    scores = component_score(latest, rl_vote)
    probs = probabilities_from_score(scores)

    raw_signal = max(["BUY", "HOLD", "SELL"], key=lambda x: probs[x])
    confidence = probs[raw_signal]

    stop_loss, take_profit, risk_abs, rr_val = trade_plan(float(latest["Close"]), float(latest["atr"]), raw_signal)

    allow_long, allow_short = market_index_filter(latest)
    signal, filter_reason = apply_filters(raw_signal, confidence, rr_val, allow_long, allow_short)

    qty = position_size(
        capital=capital,
        risk_per_trade=risk_per_trade,
        stop_distance=risk_abs,
        price=float(latest["Close"]),
        volatility_regime=str(latest["volatility_regime"]),
    )

    hist = historical_signals(frame)
    bt = backtest_with_stats(frame["Close"].to_list(), hist, initial_capital=capital, transaction_cost=0.001, slippage=0.0005)

    wf = walk_forward_backtest(
        frame["Close"].to_list(),
        wf_signal_fn,
        train_window=504,
        test_window=126,
        initial_capital=capital,
        transaction_cost=0.001,
        slippage=0.0005,
    )

    market_regime = REGIME_MAP.get(int(latest["regime"]), "Sideways")
    expected_return = (float(latest["predicted_price"]) - float(latest["Close"])) / max(float(latest["Close"]), 1e-9)

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": float(confidence),
        "predicted_price": float(latest["predicted_price"]),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_reward": float(rr_val),
        "position_size": float(qty),
        "max_position_size": float(qty),
        "volatility_regime": str(latest["volatility_regime"]),
        "market_regime": market_regime,
        "raw_signal": raw_signal,
        "filter_reason": filter_reason,
        "buy_prob": float(probs["BUY"]),
        "hold_prob": float(probs["HOLD"]),
        "sell_prob": float(probs["SELL"]),
        "score": float(probs["score"]),
        "current_price": float(latest["Close"]),
        "expected_return": float(expected_return),
        "volume_spike_ratio": float(latest["volume_spike_ratio"]),
        "momentum_20": float(latest["momentum_20"]),
        "sentiment": float(latest["sentiment"]),
        "backtest": bt,
        "walk_forward": wf,
        "frame": frame,
    }


def top_opportunities(results):
    def rank_key(x):
        return x["confidence"] + x["expected_return"] + x["momentum_20"] + x["sentiment"]

    buys = sorted([r for r in results if r["signal"] == "BUY"], key=rank_key, reverse=True)
    sells = sorted([r for r in results if r["signal"] == "SELL"], key=rank_key, reverse=True)

    combined = sorted([r for r in results if r["signal"] in {"BUY", "SELL"}], key=rank_key, reverse=True)
    return buys, sells, combined[:3]


def risk_guard(results, initial_capital):
    total_final = sum(r["backtest"]["final_capital"] for r in results)
    baseline = initial_capital * max(len(results), 1)
    daily_loss = max((baseline - total_final) / max(baseline, 1e-9), 0.0)
    trading_halted = daily_loss > 0.03
    return daily_loss, trading_halted


st.set_page_config(page_title="Transformer RL Trading AI", layout="wide")
st.title("Transformer RL Trading AI")

selected = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
sentiment_text = st.text_input("News headline", "Company posts strong earnings growth")

initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000.0, value=10000.0, step=500.0)
risk_per_trade = st.sidebar.slider("Risk per trade", min_value=0.005, max_value=0.03, value=0.01, step=0.005)

if selected:
    sentiment = sentiment_score(sentiment_text)
    with st.spinner("Loading market data and evaluating opportunities..."):
        data = load_data(list(set(selected + ["SPY", "QQQ", "^VIX"])), allow_fallback=True)

    rl_model = load_rl_model()

    results = []
    errors = []
    skipped = []
    for t in selected:
        if data[t].empty:
            skipped.append(f"{t}: empty market data")
            continue
        try:
            results.append(
                evaluate_ticker(
                    ticker=t,
                    raw_df=data[t],
                    spy_df=data["SPY"],
                    qqq_df=data["QQQ"],
                    vix_df=data["^VIX"],
                    sentiment=sentiment,
                    rl_model=rl_model,
                    capital=initial_capital,
                    risk_per_trade=risk_per_trade,
                )
            )
        except Exception as exc:
            errors.append(f"{t}: {type(exc).__name__}: {exc}")

    if errors:
        with st.expander("Ticker evaluation errors", expanded=False):
            for e in errors:
                st.write("-", e)
    if skipped:
        with st.expander("Skipped tickers", expanded=False):
            for msg in skipped:
                st.write("-", msg)

    if not results:
        st.error("No valid ticker data to evaluate. Check the error panels above.")
        st.stop()

    daily_loss, trading_halted = risk_guard(results, initial_capital)

    buys, sells, top3 = top_opportunities(results)

    st.subheader("Top Opportunities")
    st.write("Best BUY opportunities:", [x["ticker"] for x in buys[:3]])
    st.write("Best SELL opportunities:", [x["ticker"] for x in sells[:3]])
    st.write("Top 3 stocks today:", [x["ticker"] for x in top3])

    table = pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "signal": r["signal"],
                "confidence": round(r["confidence"], 4),
                "expected_return": round(r["expected_return"], 4),
                "predicted_price": round(r["predicted_price"], 2),
                "risk_reward": round(r["risk_reward"], 2),
                "volume_spike": round(r["volume_spike_ratio"], 2),
                "market_regime": r["market_regime"],
                "volatility_regime": r["volatility_regime"],
                "position_size": round(r["position_size"], 2),
                "buy_prob": round(r["buy_prob"], 3),
                "hold_prob": round(r["hold_prob"], 3),
                "sell_prob": round(r["sell_prob"], 3),
            }
            for r in results
        ]
    )
    st.dataframe(table, width="stretch")

    best = top3[0] if top3 else results[0]
    if trading_halted:
        best["signal"] = "HOLD"
        best["filter_reason"] = "Risk manager halt: daily_loss > 3%"

    st.subheader(f"Detailed View: {best['ticker']}")

    f = best["frame"]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(x=f.index, open=f["Open"], high=f["High"], low=f["Low"], close=f["Close"], name=best["ticker"])
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=420)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", best["market_regime"])
    c2.metric("Volatility", best["volatility_regime"])
    c3.metric("Signal", best["signal"])
    c4.metric("Confidence", f"{best['confidence']*100:.1f}%")

    r1, r2, r3 = st.columns(3)
    r1.metric("BUY Probability", f"{best['buy_prob']*100:.1f}%")
    r2.metric("HOLD Probability", f"{best['hold_prob']*100:.1f}%")
    r3.metric("SELL Probability", f"{best['sell_prob']*100:.1f}%")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Current Price", f"${best['current_price']:.2f}")
    p2.metric("Predicted Price", f"${best['predicted_price']:.2f}")
    p3.metric("Stop Loss", f"${best['stop_loss']:.2f}")
    p4.metric("Take Profit", f"${best['take_profit']:.2f}")

    b = best["backtest"]
    w = best["walk_forward"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Backtest Trades", b["trades"])
    m2.metric("Win Rate", f"{b['win_rate']:.1f}%")
    m3.metric("Sharpe", f"{b['sharpe']:.2f}")
    m4.metric("Sortino", f"{b['sortino']:.2f}")
    m5.metric("Profit Factor", f"{b['profit_factor']:.2f}")

    st.caption(
        f"Backtest capital ${b['final_capital']:,.2f} | Max DD {b['max_drawdown']*100:.1f}% | "
        f"Walk-forward capital ${w['final_capital']:,.2f}, trades={w['trades']}"
    )

    st.caption(f"Risk manager daily_loss estimate: {daily_loss*100:.2f}%")

    windows = pd.DataFrame(w.get("windows", []))
    if not windows.empty:
        cfig = go.Figure()
        cfig.add_trace(go.Scatter(x=windows["end_idx"], y=windows["final_capital"], mode="lines+markers", name="WF Capital"))
        cfig.update_layout(height=280, title="Walk-Forward Window Capital")
        st.plotly_chart(cfig, width="stretch")

    total_trades = sum(r["backtest"]["trades"] for r in results)
    st.caption(f"Total simulated trades across selected tickers: {total_trades} (target >= 500)")

    output = {
        "signal": best["signal"],
        "confidence": round(best["confidence"], 4),
        "predicted_price": round(best["predicted_price"], 4),
        "stop_loss": round(best["stop_loss"], 4),
        "take_profit": round(best["take_profit"], 4),
        "risk_reward": round(best["risk_reward"], 4),
        "position_size": round(best["position_size"], 4),
        "market_regime": best["market_regime"],
        "volatility_regime": best["volatility_regime"],
    }
    st.subheader("Structured Output")
    st.json(output)
