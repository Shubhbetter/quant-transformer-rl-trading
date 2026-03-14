import plotly.graph_objects as go
import streamlit as st

from src.backtester import backtest
from src.data_loader import load_data
from src.features import create_features
from src.portfolio_optimizer import equal_weight_portfolio
from src.regime_detection import detect_regime
from src.sentiment import sentiment_score


st.set_page_config(page_title="Transformer RL Trading AI", layout="wide")
st.title("Transformer RL Trading AI")

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Stock", "AAPL").upper().strip() or "AAPL"
with col2:
    sentiment_text = st.text_input("News headline", "Company posts strong earnings growth")

if ticker:
    with st.spinner(f"Loading {ticker} data..."):
        df = load_data([ticker], allow_fallback=True)[ticker]

    if df.empty:
        st.error("No data available for this symbol, even after fallback generation.")
        st.stop()

    if len(df) > 0 and df.index.dtype == "datetime64[ns]":
        st.info("If live download fails in your environment, synthetic fallback data is used automatically.")

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, width="stretch")

    feature_df = detect_regime(create_features(df.copy()))
    if feature_df.empty:
        st.error("Feature engineering produced no rows. Please adjust data window.")
        st.stop()

    latest = feature_df.iloc[-1]
    s_score = sentiment_score(sentiment_text)

    signal = "HOLD"
    if latest["macd"] > 0 and latest["rsi"] < 70 and s_score >= 0:
        signal = "BUY"
    elif latest["macd"] < 0 and latest["rsi"] > 30 and s_score < 0:
        signal = "SELL"

    signals = [signal] * len(feature_df)
    final_capital = backtest(feature_df["Close"].to_list(), signals)

    st.subheader("AI Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Regime", int(latest["regime"]))
    m2.metric("Sentiment", f"{s_score:.2f}")
    m3.metric("Signal", signal)
    m4.metric("Backtest Capital", f"${final_capital:,.2f}")

    weights = equal_weight_portfolio([ticker])
    st.caption(f"Equal-weight portfolio allocation for selected ticker: {weights[0]:.2%}")
