import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import requests
import google.generativeai as genai
from groq import Groq
import yfinance as yf
import json

# 1. Page Configuration
st.set_page_config(page_title="Quant & AI Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE CONNECTIONS ---
@st.cache_resource
def init_connections():
    # 1. Cosmos DB
    cosmos_client = CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])
    database = cosmos_client.get_database_client("FinancialData")
    container = database.get_container_client("StockTicks")
    
    # 2. Google Gemini (Primary AI)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
    
    # 3. Groq / Llama 3 (Failover AI)
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    return container, gemini_model, groq_client

container, gemini_model, groq_client = init_connections()

# The Watchlist
COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.", "CRWD": "CrowdStrike Holdings", "ORCL": "Oracle Corp.",
    "CRM": "Salesforce Inc.", "AMD": "Advanced Micro Devices", "TSM": "Taiwan Semiconductor",
    "ASML": "ASML Holding NV", "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.",
    "MA": "Mastercard Inc.", "BAC": "Bank of America Corp.", "GS": "Goldman Sachs",
    "AXP": "American Express", "LLY": "Eli Lilly and Co.", "UNH": "UnitedHealth Group",
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer Inc.", "ABBV": "AbbVie Inc.",
    "MRK": "Merck & Co.", "WMT": "Walmart Inc.", "COST": "Costco Wholesale",
    "PG": "Procter & Gamble", "KO": "Coca-Cola Co.", "PEP": "PepsiCo Inc.",
    "SBUX": "Starbucks Corp.", "MCD": "McDonald's Corp.", "NKE": "NIKE Inc.",
    "NFLX": "Netflix Inc.", "XOM": "Exxon Mobil Corp.", "CVX": "Chevron Corp.",
    "CAT": "Caterpillar Inc.", "GE": "General Electric Co.", "T": "AT&T Inc.",
    "VZ": "Verizon Communications"
}

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Quant Controls")
st.sidebar.markdown("---")

selected_tickers = st.sidebar.multiselect("Select Assets", sorted(COMPANY_NAMES.keys()), default=["NVDA", "TSLA"])
timeframe = st.sidebar.radio("Timeframe (Candle Size)", ("Live Ticks (Cosmos DB)", "1-Hour Candles (yfinance)", "1-Day Candles (yfinance)"))
analysis_mode = st.sidebar.selectbox("Statistical Overlay", ("None", "Trend (Moving Averages)", "Anomaly Detection (Bollinger Bands)"))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Graph Parameters")

with st.sidebar.expander("Trend Settings", expanded=(analysis_mode == "Trend (Moving Averages)")):
    fast_ma = st.slider("Fast Moving Average", min_value=5, max_value=50, value=20, step=1)
    slow_ma = st.slider("Slow Moving Average", min_value=20, max_value=200, value=50, step=1)

with st.sidebar.expander("Anomaly Settings", expanded=(analysis_mode == "Anomaly Detection (Bollinger Bands)")):
    bb_window = st.number_input("Bollinger Window (Periods)", min_value=5, max_value=100, value=20)
    bb_std = st.slider("Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

with st.sidebar.expander("Display Settings"):
    chart_height = st.slider("Graph Height (Pixels)", min_value=300, max_value=800, value=380, step=10)

if st.sidebar.button("Refresh Terminal Data"):
    st.rerun()

st.title("Multi-Source Quant & AI Sentiment Terminal")
st.markdown("---")

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf): 
    if tf == "Live Ticks (Cosmos DB)":
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'price']].rename(columns={'price': 'close'})
    else:
        interval = "1h" if "1-Hour" in tf else "1d"
        period = "1mo" if "1-Hour" in tf else "1y"
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        df.index = df.index.tz_convert('UTC').tz_localize(None) 
        return df[['open', 'high', 'low', 'close']]

# --- HYBRID AI FUNCTION (GOOGLE -> GROQ FAILOVER) ---
@st.cache_data(ttl=3600)
def get_batched_ai_sentiment(symbols_tuple):
    if not symbols_tuple:
        return {}
        
    try:
        # 1. Gather all news
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        past = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        master_news_dict = {}
        for symbol in symbols_tuple:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={past}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
            news_data = requests.get(url).json()
            if news_data:
                headlines = [article['headline'] for article in news_data[:5]]
                master_news_dict[symbol] = headlines
            else:
                master_news_dict[symbol] = ["No recent news found."]

        # 2. Build the Master Prompt
        prompt = f"""
        Analyze the sentiment for EACH stock based on these headlines:
        {master_news_dict}
        
        Format your response as a JSON dictionary where the keys are the stock tickers, and the values are objects with "sentiment" (BULLISH, BEARISH, or NEUTRAL) and "summary" (1 strict sentence explanation).
        """
        
        # 3. ATTEMPT 1: Google Gemini API
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            
            # Tag the output so we know who answered!
            for sym in data:
                data[sym]["summary"] = f"[Via Google] {data[sym].get('summary', '')}"
            return data
            
        except Exception as google_error:
            # 4. FAILOVER: If Google throws 429 Quota Exceeded, seamlessly route to Groq
            print(f"Google API Failed ({google_error}). Falling back to Groq Llama 3...")
            
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a quantitative analyst. You output strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192", 
                response_format={"type": "json_object"},
            )
            
            data = json.loads(chat_completion.choices[0].message.content)
            
            # Tag the output so we know who answered!
            for sym in data:
                data[sym]["summary"] = f"[Via Groq] {data[sym].get('summary', '')}"
            return data

    except Exception as e:
        # Total System Failure
        error_dict = {}
        for sym in symbols_tuple:
            error_dict[sym] = {"sentiment": "ERROR", "summary": f"Total AI Failure: {str(e)}"}
        return error_dict

# Pre-compute AI Sentiment
batched_sentiments = get_batched_ai_sentiment(tuple(selected_tickers))

# --- BUILD THE DASHBOARD GRID ---
if selected_tickers:
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            st.subheader(f"{ticker} | {COMPANY_NAMES.get(ticker, ticker)}")
            
            df = fetch_market_data(ticker, timeframe)
            
            if not df.empty:
                # --- RESTORED: TOP KPI METRICS ---
                latest = df.iloc[-1]
                delta_val = None
                if len(df) > 1:
                    prev = df.iloc[-2]
                    delta_val = latest['close'] - prev['close']
                    
                m1, m2, m3 = st.columns(3)
                m1.metric("Current Price", f"${latest['close']:.2f}", f"{delta_val:.2f}" if delta_val else None)
                m2.metric("High", f"${latest['high']:.2f}")
                m3.metric("Low", f"${latest['low']:.2f}")
                
                st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room

                # --- STATISTICAL MATH ---
                df['SMA_BB'] = df['close'].rolling(window=bb_window).mean()
                df['STD_BB'] = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                # --- DRAW THE CHART ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ))

                if analysis_mode == "Trend (Moving Averages)":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], mode='lines', line=dict(color='orange', width=2), name=f"{fast_ma} SMA"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], mode='lines', line=dict(color='dodgerblue', width=2), name=f"{slow_ma} SMA"))
                elif analysis_mode == "Anomaly Detection (Bollinger Bands)":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name="Upper Band"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="Lower Band"))

                fig.update_layout(
                    template="plotly_dark", height=chart_height, margin=dict(l=0, r=0, t=10, b=0), 
                    xaxis_rangeslider_visible=False, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- AI SENTIMENT ---
                stock_data = batched_sentiments.get(ticker, {"sentiment": "NEUTRAL", "summary": "AI data currently unavailable."})
                sentiment = stock_data.get("sentiment", "NEUTRAL")
                summary = stock_data.get("summary", "")
                
                if "BULLISH" in sentiment.upper():
                    st.success(f"**🤖 AI: {sentiment}** — {summary}")
                elif "BEARISH" in sentiment.upper():
                    st.error(f"**🤖 AI: {sentiment}** — {summary}")
                else:
                    st.info(f"**🤖 AI: {sentiment}** — {summary}")
                    
                st.markdown("---")
            else:
                st.info(f"Awaiting telemetry for {ticker}...")
