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
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="Advanced Stock Predictor", layout="wide")

# -----------------------------
# API Key
# -----------------------------
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    st.error("Alpha Vantage API key not found. Add it to Streamlit secrets.")
    st.stop()

# -----------------------------
# Sentiment Functions
# -----------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    tb = TextBlob(text)
    return (tb.sentiment.polarity + 1) / 2

@st.cache_resource
def get_hf_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

hf_pipe = get_hf_sentiment()

def get_hf_score(text: str) -> float:
    try:
        result = hf_pipe(text[:512])[0]
        score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
        return score
    except:
        return 0.5

def combined_sentiment(text: str) -> float:
    return (get_textblob_sentiment(text) + get_hf_score(text)) / 2

# -----------------------------
# Model (2-layer LSTM)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource
def train_lstm_model(X_train, y_train, input_dim):
    model = LSTMModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    return model

# -----------------------------
# Sentiment Fetch
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_sentiment(ticker):
    gn = GNews(language='en', country='US', period='14d')
    articles = gn.get_news(f"{ticker} stock")
    news_df = pd.DataFrame(articles)
    if news_df.empty:
        return pd.DataFrame(columns=['date', 'average_sentiment_score'])

    text_col = "title" if "title" in news_df.columns else news_df.columns[0]
    date_col = next((c for c in news_df.columns if "publish" in c.lower() or "date" in c.lower()), None)

    news_df['text'] = news_df[text_col].astype(str)
    if 'description' in news_df.columns:
        news_df['text'] += " " + news_df['description'].astype(str)

    news_df[date_col] = pd.to_datetime(news_df[date_col], errors='coerce')
    news_df.dropna(subset=[date_col], inplace=True)

    news_df['sentiment_score'] = news_df['text'].apply(combined_sentiment)
    daily = news_df.groupby(news_df[date_col].dt.date)['sentiment_score'].mean().reset_index()
    daily.rename(columns={date_col: 'date', 'sentiment_score': 'average_sentiment_score'}, inplace=True)
    daily['date'] = pd.to_datetime(daily['date'])

    # Fill missing dates for smoother graph
    full_range = pd.date_range(daily['date'].min(), datetime.today())
    daily = daily.set_index('date').reindex(full_range).fillna(method='ffill').reset_index()
    daily.columns = ['date', 'average_sentiment_score']

    # Smooth sentiment curve
    daily['average_sentiment_score'] = daily['average_sentiment_score'].rolling(3, min_periods=1).mean()
    return daily

# -----------------------------
# Stock Analysis
# -----------------------------
def analyze_stock(ticker_symbol):
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        stock_data, _ = ts.get_daily(symbol=ticker_symbol, outputsize='full')
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data.sort_index().tail(365)
        stock_data.rename(columns={'4. close': 'Close', '5. volume': 'Volume'}, inplace=True)

        # Moving averages
        stock_data['MA_50'] = stock_data['Close'].rolling(50).mean()
        stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
        stock_data['Daily_Return'] = stock_data['Close'].pct_change().fillna(0)

        # Sentiment merge
        daily_sentiment = fetch_sentiment(ticker_symbol)
        merged = pd.merge(stock_data.reset_index(), daily_sentiment, left_on='index', right_on='date', how='left')
        merged.drop(columns=['date'], inplace=True)
        merged.rename(columns={'index': 'date'}, inplace=True)
        merged['average_sentiment_score'].fillna(method='ffill', inplace=True)
        merged['average_sentiment_score'].fillna(0.5, inplace=True)

        # Prepare model input
        features = ['Close', 'Volume', 'MA_50', 'MA_200', 'Daily_Return', 'average_sentiment_score']
        merged = merged.dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(merged[features])
        seq_len = 10

        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i+seq_len])
            y.append(scaled[i+seq_len, 0])
        X, y = np.array(X), np.array(y)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        model = train_lstm_model(X_t, y_t, X_t.shape[2])
        model.eval()
        pred = model(X_t).detach().numpy()
        pred_rescaled = scaler.inverse_transform(
            np.concatenate([pred, np.zeros((len(pred), scaled.shape[1]-1))], axis=1)
        )[:, 0]

        merged = merged.iloc[seq_len:].copy()
        merged['Predicted_Close'] = pred_rescaled

        # Buy/Sell signal
        diff = (merged['Predicted_Close'].iloc[-1] - merged['Close'].iloc[-1]) / merged['Close'].iloc[-1]
        sentiment = merged['average_sentiment_score'].iloc[-1]
        suggestion = "BUY" if diff > 0.01 and sentiment > 0.55 else "SELL" if diff < -0.01 or sentiment < 0.45 else "HOLD"

        return merged, suggestion

    except Exception as e:
        st.error(f"Error: {e}")
        return None, "HOLD"

# -----------------------------
# UI
# -----------------------------
st.title("📈 Advanced Stock Market Predictor (with Sentiment & LSTM)")
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()

if st.sidebar.button("Analyze"):
    data, suggestion = analyze_stock(ticker)
    if data is not None and not data.empty:
        st.subheader(f"Investment Suggestion: {suggestion}")

        # Price prediction
        st.subheader("Predicted vs Actual Close Price")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(x='date', y='Close', data=data, label='Actual')
        sns.lineplot(x='date', y='Predicted_Close', data=data, label='Predicted', color='red')
        ax.set_xlabel("Date"); ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Sentiment
        st.subheader("Smoothed Daily Sentiment")
        fig2, ax2 = plt.subplots(figsize=(12,4))
        sns.lineplot(x='date', y='average_sentiment_score', data=data, label='Avg Sentiment')
        ax2.axhline(0.5, color='gray', linestyle='--')
        st.pyplot(fig2)

        # Volume
        st.subheader("Volume")
        fig3, ax3 = plt.subplots(figsize=(12,4))
        sns.barplot(x='date', y='Volume', data=data, ax=ax3)
        ax3.set_xlabel("Date"); ax3.set_ylabel("Volume")
        st.pyplot(fig3)

        st.subheader("Recent Data Snapshot")
        st.dataframe(data.tail(10))
    else:
        st.warning("No data available or API error.")
