import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("Transformer RL Trading AI")

ticker = st.text_input("Stock","AAPL")

df = yf.download(ticker)

fig = go.Figure()

fig.add_trace(go.Candlestick(
x=df.index,
open=df["Open"],
high=df["High"],
low=df["Low"],
close=df["Close"]
))

st.plotly_chart(fig)

if st.button("Run AI Prediction"):

    signal = "BUY"

    st.success(f"AI Signal: {signal}")
