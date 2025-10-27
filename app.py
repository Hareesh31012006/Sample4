# =========================================================
# 📊 Stock Sentiment & Price Prediction Dashboard (Stable)
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
ALPHA_VANTAGE_API_KEY = "9EJ41V9XS6Q5ZN1Y"  # Replace with your valid key

# =========================================================
# 🏗️ Streamlit App Setup
# =========================================================
st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")
st.title("📈 Stock Sentiment + Price Prediction Dashboard")
st.write("Predict stock trends using sentiment and deep learning models.")

# =========================================================
# ⚙️ Helper Settings
# =========================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =========================================================
# 🧠 Sentiment Analysis Utilities
# =========================================================
@st.cache_data
def get_textblob_sentiment(text):
    """Compute sentiment using TextBlob."""
    if not text:
        return 0
    return TextBlob(text).sentiment.polarity

@st.cache_resource
def get_hf_sentiment():
    """Load Hugging Face sentiment pipeline (cached)."""
    return pipeline("sentiment-analysis")

hf_pipeline = get_hf_sentiment()

# =========================================================
# 📰 Fetch News
# =========================================================
@st.cache_data(ttl=3600)
def fetch_news(symbol):
    google_news = GNews(language="en", max_results=10, country="US", period="7d")
    return google_news.get_news(f"{symbol} stock")

# =========================================================
# 💹 Fetch Stock Data
# =========================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="compact")
    if data.empty:
        return pd.DataFrame()
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
# 🧮 PyTorch Linear Model (Deterministic)
# =========================================================
def train_model(X, y, seed=123):
    torch.manual_seed(seed)
    model = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        opt.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
    return model

# =========================================================
# 🔍 Analyze Stock (Safe & Stable)
# =========================================================
def analyze_stock(symbol):
    # 1️⃣ Fetch stock data
    df = fetch_stock_data(symbol)
    if df.empty or len(df) < 5:
        st.warning(f"No valid stock data found for '{symbol}'. Try another ticker like AAPL or MSFT.")
        return None, None, None, None

    # 2️⃣ Fetch news and sentiment
    news = fetch_news(symbol)
    if not news:
        st.warning(f"No recent news found for {symbol}. Using neutral sentiment.")
        sentiments = [("No News", 0, 0)]
    else:
        sentiments = []
        for n in news:
            text = (n.get("title", "") or "") + " " + (n.get("description", "") or "")
            tb = get_textblob_sentiment(text)
            try:
                hf_label = hf_pipeline(text[:512])[0]["label"]
                hf_score = 1 if hf_label == "POSITIVE" else -1 if hf_label == "NEGATIVE" else 0
            except Exception:
                hf_score = 0
            sentiments.append((text, tb, hf_score))

    sent_df = pd.DataFrame(sentiments, columns=["Text", "TextBlob", "HF_Sentiment"])
    avg_sentiment = sent_df[["TextBlob", "HF_Sentiment"]].mean().mean()

    # 3️⃣ Prepare model data
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    if df.empty:
        st.warning("Not enough valid stock data to train model.")
        return None, sent_df, None, None

    X = torch.tensor(df[["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    y = torch.tensor(df["Close"].values.reshape(-1, 1), dtype=torch.float32)

    if X.numel() == 0 or y.numel() == 0:
        st.warning("Insufficient data for prediction.")
        return None, sent_df, None, None

    # 4️⃣ Train model
    model = train_model(X, y)

    # 5️⃣ Predict next-day close
    last_row = torch.tensor(df.iloc[-1][["Open", "High", "Low", "Volume"]].values, dtype=torch.float32)
    next_pred = model(last_row.unsqueeze(0)).item()

    # 6️⃣ Decision logic
    last_close = df["Close"].iloc[-1]
    suggestion = "📈 BUY" if next_pred > last_close and avg_sentiment > 0 else "📉 SELL"

    return df, sent_df, next_pred, suggestion

# =========================================================
# 🧾 Streamlit UI
# =========================================================
symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT):", "AAPL")

if st.button("Analyze"):
    with st.spinner("Fetching data and running analysis..."):
        df, sent_df, next_pred, suggestion = analyze_stock(symbol)

    if df is None:
        st.stop()

    st.success(f"✅ Analysis Complete for {symbol}")
    st.metric("Predicted Next-Day Close", f"${next_pred:.2f}")
    st.metric("Trading Suggestion", suggestion)

    # Stock Chart
    st.subheader("Recent Stock Prices")
    st.line_chart(df["Close"])

    # Sentiment Summary
    st.subheader("Sentiment Summary")
    st.dataframe(sent_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(sent_df["TextBlob"], bins=10, kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

# =========================================================
# 🧾 Footer
# =========================================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, TextBlob, and HuggingFace.")
