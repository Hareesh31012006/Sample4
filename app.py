# =========================================================
# 📈 ENHANCED STOCK MARKET SENTIMENT & PRICE PREDICTOR
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
from datetime import datetime

# ---------------------------------------------------------
# 🔧 Basic Config
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📊 AI-Powered Stock Market Predictor")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------
# 🔑 API Key — Paste yours here
# ---------------------------------------------------------
ALPHA_VANTAGE_API_KEY = "9EJ41V9XS6Q5ZN1Y"  # <-- Replace if you have your own

if not ALPHA_VANTAGE_API_KEY:
    st.error("❌ Alpha Vantage API key not found. Please paste it in app.py.")
    st.stop()

# ---------------------------------------------------------
# ⚙️ Helper Functions
# ---------------------------------------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    """Compute sentiment using TextBlob"""
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    polarity = TextBlob(text).sentiment.polarity
    return (polarity + 1) / 2  # Convert -1..1 → 0..1

@st.cache_resource
def get_hf_sentiment():
    """Load HuggingFace sentiment model"""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

hf_sentiment_pipe = get_hf_sentiment()

def get_hf_score(text: str) -> float:
    """Compute sentiment with HuggingFace model"""
    try:
        res = hf_sentiment_pipe(text[:512])[0]
        score = res["score"] if res["label"] == "POSITIVE" else 1 - res["score"]
        return score
    except Exception:
        return 0.5

def combined_sentiment(text: str) -> float:
    """Combine TextBlob + HuggingFace"""
    tb = get_textblob_sentiment(text)
    hf = get_hf_score(text)
    return (tb + hf) / 2

# ---------------------------------------------------------
# 🧮 Simple PyTorch Model (Regression)
# ---------------------------------------------------------
@st.cache_resource
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

# ---------------------------------------------------------
# 📊 Fetch Stock + News Sentiment
# ---------------------------------------------------------
def analyze_stock(symbol: str):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=symbol, outputsize="compact")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index().reset_index()
    data.rename(columns={"index": "date"}, inplace=True)
    data["MA_50"] = data["4. close"].rolling(50).mean()
    data["MA_200"] = data["4. close"].rolling(200).mean()
    data["Daily_Return"] = data["4. close"].pct_change().fillna(0)

    # Fetch recent news
    gn = GNews(language="en", country="US", period="7d")
    articles = gn.get_news(f"{symbol} stock")
    news_df = pd.DataFrame(articles)
    if not news_df.empty:
        date_col = [c for c in news_df.columns if "date" in c.lower()][0]
        text_col = "title"
        news_df[date_col] = pd.to_datetime(news_df[date_col], errors="coerce")
        news_df.dropna(subset=[date_col], inplace=True)
        news_df["sentiment_score"] = news_df[text_col].astype(str).apply(combined_sentiment)
        news_df["date"] = news_df[date_col].dt.date
        daily_sentiment = news_df.groupby("date")["sentiment_score"].mean().reset_index()
        daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    else:
        daily_sentiment = pd.DataFrame(columns=["date", "sentiment_score"])

    # Merge with stock data
    merged = pd.merge(data, daily_sentiment, on="date", how="left")
    merged["sentiment_score"] = merged["sentiment_score"].interpolate().fillna(0.5)

    # Train PyTorch model
    features = ["4. close", "5. volume", "MA_50", "MA_200", "Daily_Return", "sentiment_score"]
    merged.fillna(method="bfill", inplace=True)
    X = torch.tensor(merged[features].values, dtype=torch.float32)
    y = torch.tensor(merged["4. close"].values, dtype=torch.float32).reshape(-1, 1)
    model = train_model(X, y)
    pred = model(X).detach().numpy().flatten()
    merged["Predicted_Close"] = pred

    # Suggestion
    last = merged.iloc[-1]
    diff = (last["Predicted_Close"] - last["4. close"]) / last["4. close"]
    if diff > 0.01 and last["sentiment_score"] > 0.55:
        suggestion = "BUY"
    elif diff < -0.01 and last["sentiment_score"] < 0.45:
        suggestion = "SELL"
    else:
        suggestion = "HOLD"

    return merged, suggestion

# ---------------------------------------------------------
# 🖥️ Streamlit UI
# ---------------------------------------------------------
st.sidebar.header("Stock Input")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT)", "AAPL").upper()

if st.sidebar.button("Analyze"):
    with st.spinner("Analyzing... please wait."):
        df, suggestion = analyze_stock(symbol)

    st.subheader(f"📈 Investment Suggestion: **{suggestion}**")
    st.caption(f"Based on predicted trend and sentiment analysis for {symbol}")

    # Price Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="date", y="4. close", data=df, label="Actual Close", ax=ax)
    sns.lineplot(x="date", y="Predicted_Close", data=df, label="Predicted Close", ax=ax)
    ax.set_title(f"{symbol} - Actual vs Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)

    # Sentiment Chart
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.lineplot(x="date", y="sentiment_score", data=df, color="orange", ax=ax2)
    ax2.axhline(0.5, color="gray", linestyle="--")
    ax2.set_title("Daily Average Sentiment Score")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment (0-1)")
    st.pyplot(fig2)

    # Backtest Performance
    df["Position"] = np.where(df["MA_50"] > df["MA_200"], 1, 0)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Daily_Return"]
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()
    df["Cumulative_Market"] = (1 + df["Daily_Return"]).cumprod()

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    sns.lineplot(x="date", y="Cumulative_Strategy", data=df, label="Strategy", ax=ax3)
    sns.lineplot(x="date", y="Cumulative_Market", data=df, label="Market", ax=ax3)
    ax3.set_title("Backtest Performance (MA Crossover)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative Return")
    st.pyplot(fig3)

    st.subheader("📋 Latest Data")
    st.dataframe(df.tail(10)[["date", "4. close", "Predicted_Close", "sentiment_score"]])
