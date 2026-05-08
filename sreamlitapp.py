import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import requests
import google.generativeai as genai
import yfinance as yf

st.set_page_config(page_title="Quant & AI Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE CONNECTIONS ---
@st.cache_resource
def init_connections():
    # 1. Cosmos DB (For Live Ticks)
    cosmos_client = CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])
    database = cosmos_client.get_database_client("FinancialData")
    container = database.get_container_client("StockTicks")
    
    # 2. Google Gemini AI (Upgraded to Pro for deeper reasoning)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    ai_model = genai.GenerativeModel('gemini-1.5-pro') 
    
    return container, ai_model

container, ai_model = init_connections()

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "AMD": "Advanced Micro Devices", "TSM": "Taiwan Semiconductor",
    "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "LLY": "Eli Lilly and Co.", 
    "WMT": "Walmart Inc.", "XOM": "Exxon Mobil Corp.", "CAT": "Caterpillar Inc."
    # Add the rest of your 40 stocks here...
}

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Quant Controls")
st.sidebar.markdown("---")

selected_tickers = st.sidebar.multiselect("Select Assets", sorted(COMPANY_NAMES.keys()), default=["NVDA", "TSLA"])
timeframe = st.sidebar.radio("Timeframe (Candle Size)", ("Live Ticks (Cosmos DB)", "1-Hour Candles (yfinance)", "1-Day Candles (yfinance)"))
analysis_mode = st.sidebar.selectbox("Statistical Overlay", ("None", "Trend (Moving Averages)", "Anomaly Detection (Bollinger Bands)"))

if st.sidebar.button("Refresh Terminal Data"):
    st.rerun()

st.title("Multi-Source Quant & AI Sentiment Terminal")
st.markdown("---")

# --- DATA FETCHING & AI FUNCTIONS ---

# Router: Chooses between Live Cosmos DB Data or Historical Yahoo Data
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf): 
    if tf == "Live Ticks (Cosmos DB)":
        # Pull the live stream from your Azure Pipeline
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'price']].rename(columns={'price': 'close'})
    
    else:
        # Pull deep historical data from Yahoo Finance for statistical math
        interval = "1h" if "1-Hour" in tf else "1d"
        period = "1mo" if "1-Hour" in tf else "1y"
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        # Rename columns to match our standard format
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        # yfinance returns timezone-aware datetimes, convert to naive UTC for plotting
        df.index = df.index.tz_convert('UTC').tz_localize(None) 
        return df[['open', 'high', 'low', 'close']]

@st.cache_data(ttl=3600) # Caches AI for 1 hour to save free tier limits
def get_ai_sentiment(symbol):
    try:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        yesterday = (datetime.datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={yesterday}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
        news_data = requests.get(url).json()
        
        if not news_data:
            return "NEUTRAL", "Not enough recent news to determine sentiment."
            
        headlines = [article['headline'] for article in news_data[:8]] # Feed 8 headlines to the Pro model
        headlines_text = "\n".join(headlines)
        
        prompt = f"""
        You are an expert Wall Street quantitative analyst. Analyze these recent news headlines for {symbol}:
        {headlines_text}
        
        Based ONLY on these headlines, reply with exactly two lines:
        Line 1: The overall market sentiment (Respond with exactly one word: BULLISH, BEARISH, or NEUTRAL).
        Line 2: A strict 1-sentence summary of WHY the market feels this way.
        """
        response = ai_model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        
        sentiment_label = lines[0].replace("Line 1:", "").strip().upper()
        summary_text = lines[-1].replace("Line 2:", "").strip()
        return sentiment_label, summary_text
        
    except Exception as e:
        return "ERROR", f"Failed to fetch AI analysis: {str(e)}"

# --- BUILD THE DASHBOARD GRID ---
if selected_tickers:
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            st.subheader(f"{ticker} | {COMPANY_NAMES.get(ticker, ticker)}")
            
            # Fetch routed data
            df = fetch_market_data(ticker, timeframe)
            
            if not df.empty:
                # Apply Quant Statistics (Pandas is insanely fast at this)
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['STD_20'] = df['close'].rolling(window=20).std()
                df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
                df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)

                # Draw the Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ))

                if analysis_mode == "Trend (Moving Averages)":
                    df['SMA_50'] = df['close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', line=dict(color='orange', width=2), name="20 SMA"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', line=dict(color='dodgerblue', width=2), name="50 SMA"))
                elif analysis_mode == "Anomaly Detection (Bollinger Bands)":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name="Upper Band"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="Lower Band"))

                fig.update_layout(
                    template="plotly_dark", height=380, margin=dict(l=0, r=0, t=10, b=0), 
                    xaxis_rangeslider_visible=False, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- AI SENTIMENT MODULE ---
                sentiment, summary = get_ai_sentiment(ticker)
                
                if "BULLISH" in sentiment:
                    st.success(f"**🤖 AI: {sentiment}** — {summary}")
                elif "BEARISH" in sentiment:
                    st.error(f"**🤖 AI: {sentiment}** — {summary}")
                else:
                    st.info(f"**🤖 AI: {sentiment}** — {summary}")
                    
                st.markdown("---")
            else:
                st.info(f"Awaiting telemetry for {ticker}...")
