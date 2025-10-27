# =========================================================
# 📊 Stock Sentiment & Price Prediction Dashboard
# =========================================================

import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# =========================================================
# 🔑 SET YOUR API KEY HERE
# =========================================================
ALPHA_VANTAGE_API_KEY = "9EJ41V9XS6Q5ZN1Y"  # 👈 Replace with your key

# =========================================================
# 🏗️ Streamlit App Setup
# =========================================================
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")
st.write("Predict stock trends using sentiment and deep learning models.")

# =========================================================
# 🧠 Sentiment Analysis Utilities
# =========================================================
@st.cache_data
def get_textblob_sentiment(text):
    """Compute sentiment using TextBlob (simple polarity)."""
    return TextBlob(text).sentiment.polarity

@st.cache_data
def get_hf_sentiment():
    """Load Hugging Face sentiment pipeline (cached)."""
    return pipeline("sentiment-analysis")

hf_pipeline = get_hf_sentiment()

# =========================================================
# 📰 Fetch News
# =========================================================
@st.cache_data
def fetch_news(symbol):
    google_news = GNews(language="en", max_results=10)
    return google_news.get_news(symbol)

# =========================================================
# 💹 Fetch Stock Data
# =========================================================
@st.cache_data
def fetch_stock_data(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data.index = pd.to_datetime(data.index)
    return data.sort_index()

# =========================================================
# 🧮 Simple PyTorch Regression Model (Fixed)
# =========================================================
def train_model(X, y):
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(150):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
    return model

# =========================================================
# 🔍 Analyze Stock
# =========================================================
def analyze_stock(symbol):
    # 1️⃣ Get stock data
    df = fetch_stock_data(symbol)

    # 2️⃣ Get news + sentiment
    news = fetch_news(symbol)
    sentiments = []
    for n in news:
        text = n["title"] + " " + n.get("description", "")
        tb = get_textblob_sentiment(text)
        hf = hf_pipeline(text[:512])[0]["label"]
        val = 1 if hf == "POSITIVE" else -1 if hf == "NEGATIVE" else 0
        sentiments.append((text, tb, val))
    sent_df = pd.DataFrame(sentiments, columns=["Text", "TextBlob", "HF_Sentiment"])
    avg_sentiment = sent_df[["TextBlob", "HF_Sentiment"]].mean().mean()

    # 3️⃣ Prepare data for model
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    X = torch.tensor(df[["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    y = torch.tensor(df["Close"].values.reshape(-1, 1), dtype=torch.float32)

    # 4️⃣ Train model
    model = train_model(X, y)

    # 5️⃣ Predict next-day price
    last_row = torch.tensor(df.iloc[-1][["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    next_pred = model(last_row.unsqueeze(0)).item()

    # 6️⃣ Final suggestion
    suggestion = "📈 Buy" if next_pred > df["Close"].iloc[-1] and avg_sentiment > 0 else "📉 Sell"
    return df, sent_df, next_pred, suggestion

# =========================================================
# 🧾 Streamlit UI
# =========================================================
symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")

if st.button("Analyze"):
    with st.spinner("Fetching data and running analysis..."):
        df, sent_df, next_pred, suggestion = analyze_stock(symbol)

    st.success(f"✅ Analysis Complete for {symbol}")
    st.metric("Predicted Next-Day Close", f"${next_pred:.2f}")
    st.metric("Trading Suggestion", suggestion)

    st.subheader("Recent Stock Prices")
    st.line_chart(df["Close"])

    st.subheader("Sentiment Summary")
    st.dataframe(sent_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(sent_df["TextBlob"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

# =========================================================
# 🧾 Footer
# =========================================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and HuggingFace.")
