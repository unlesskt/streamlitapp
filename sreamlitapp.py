import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import requests
import google.generativeai as genai
import yfinance as yf
import json # NEW: Required for parsing the batched AI response

# 1. Page Configuration
st.set_page_config(page_title="Quant & AI Terminal", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE CONNECTIONS ---
@st.cache_resource
def init_connections():
    cosmos_client = CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])
    database = cosmos_client.get_database_client("FinancialData")
    container = database.get_container_client("StockTicks")
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # We use Flash here. Because we batch everything into 1 request, 
    # it runs instantly and never hits the 5 RPM limit!
    ai_model = genai.GenerativeModel('gemini-2.5-flash') 
    
    return container, ai_model

container, ai_model = init_connections()

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

# --- DATA FETCHING & AI FUNCTIONS ---
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

# NEW: BATCHED AI FUNCTION (With Strict JSON Enforcement)
@st.cache_data(ttl=3600)
def get_batched_ai_sentiment(symbols_tuple):
    if not symbols_tuple:
        return {}
        
    try:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        # Expanded to 5 days to ensure we always find news, even over weekends
        past = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        # 1. Gather all news
        master_news_dict = {}
        for symbol in symbols_tuple:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={past}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
            news_data = requests.get(url).json()
            if news_data:
                headlines = [article['headline'] for article in news_data[:5]]
                master_news_dict[symbol] = headlines
            else:
                master_news_dict[symbol] = ["No recent news found."]

        # 2. Build the Mega-Prompt
        prompt = f"""
        Analyze the sentiment for EACH stock based on these headlines:
        {master_news_dict}
        
        Format your response as a JSON dictionary where the keys are the stock tickers, and the values are objects with "sentiment" (BULLISH, BEARISH, or NEUTRAL) and "summary" (1 strict sentence explanation).
        """
        
        # 3. Request Strict JSON directly from the API
        response = ai_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 4. Parse effortlessly
        return json.loads(response.text)
        
    except Exception as e:
        # If it fails now, it will actually tell us WHY on the dashboard instead of hiding it!
        error_dict = {}
        for sym in symbols_tuple:
            error_dict[sym] = {"sentiment": "ERROR", "summary": f"Debug Info: {str(e)}"}
        return error_dict

# --- PRE-COMPUTE AI SENTIMENT ---
# We pass the list as a tuple so Streamlit can cache it properly
batched_sentiments = get_batched_ai_sentiment(tuple(selected_tickers))

# --- BUILD THE DASHBOARD GRID ---
if selected_tickers:
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            st.subheader(f"{ticker} | {COMPANY_NAMES.get(ticker, ticker)}")
            
            df = fetch_market_data(ticker, timeframe)
            
            if not df.empty:
                df['SMA_BB'] = df['close'].rolling(window=bb_window).mean()
                df['STD_BB'] = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

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
                
                # --- NEW: PULL PRE-COMPUTED AI SENTIMENT ---
                # Safely get the data from the dictionary we built earlier
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
