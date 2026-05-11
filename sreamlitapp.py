import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import requests
import google.generativeai as genai
from openai import OpenAI
import yfinance as yf
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Institutional Quant Terminal", layout="wide", initial_sidebar_state="expanded")

# --- 2. SYSTEM CONNECTIONS ---
@st.cache_resource
def init_connections():
    # Azure Cosmos DB
    cosmos_client = CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])
    database = cosmos_client.get_database_client("FinancialData")
    container = database.get_container_client("StockTicks")
    
    # Primary AI: Google Gemini
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
    
    # Failover AI: OpenRouter (Llama 3 Free)
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENROUTER_API_KEY"],
    )
    
    return container, gemini_model, openrouter_client

container, gemini_model, openrouter_client = init_connections()

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

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Terminal Controls")
st.sidebar.markdown("---")

selected_tickers = st.sidebar.multiselect("Select Assets", sorted(COMPANY_NAMES.keys()), default=["NVDA", "TSLA", "AAPL", "MSFT"])
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
    chart_height = st.slider("Graph Height (Pixels)", min_value=300, max_value=800, value=450, step=10)

if st.sidebar.button("Refresh Terminal Data"):
    st.rerun()

# --- 4. MACRO MARKET HEADER ---
st.title("Institutional Quant Terminal")

@st.cache_data(ttl=300)
def fetch_macro_indices():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    data = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            latest = hist.iloc[-1]['Close']
            prev = hist.iloc[-2]['Close']
            pct_change = ((latest - prev) / prev) * 100
            data[name] = {"price": latest, "change": pct_change}
        except:
            data[name] = {"price": 0.0, "change": 0.0}
    return data

macro_data = fetch_macro_indices()
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("S&P 500", f"{macro_data['S&P 500']['price']:,.2f}", f"{macro_data['S&P 500']['change']:.2f}%")
m_col2.metric("NASDAQ", f"{macro_data['NASDAQ']['price']:,.2f}", f"{macro_data['NASDAQ']['change']:.2f}%")
m_col3.metric("Dow Jones", f"{macro_data['Dow Jones']['price']:,.2f}", f"{macro_data['Dow Jones']['change']:.2f}%")
st.markdown("---")

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf): 
    if tf == "Live Ticks (Cosmos DB)":
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items: return pd.DataFrame()
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'price']].rename(columns={'price': 'close'})
    else:
        interval = "1h" if "1-Hour" in tf else "1d"
        period = "1mo" if "1-Hour" in tf else "1y"
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        df.index = df.index.tz_convert('UTC').tz_localize(None) 
        return df[['open', 'high', 'low', 'close', 'Volume']]

@st.cache_data(ttl=3600)
def get_batched_ai_sentiment(symbols_tuple):
    if not symbols_tuple: return {}
    try:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        past = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        master_news_dict = {}
        for symbol in symbols_tuple:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={past}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
            news_data = requests.get(url).json()
            master_news_dict[symbol] = [article['headline'] for article in news_data[:5]] if news_data else ["No recent news."]

        prompt = f"""
        Analyze the sentiment for EACH stock based on these headlines: {master_news_dict}
        Format your response as a strictly valid JSON dictionary where the keys are the stock tickers, and the values are objects with "sentiment" (BULLISH, BEARISH, or NEUTRAL) and "summary" (1 strict sentence explanation).
        """
        
        # Primary: Google Gemini
        try:
            response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            data = json.loads(response.text)
            for sym in data: data[sym]["summary"] = f"[Google] {data[sym].get('summary', '')}"
            return data
            
        # Failover: OpenRouter (Llama 3)
        except Exception as google_error:
            print(f"Google Rate Limit hit. Failing over to OpenRouter Llama 3...")
            chat = openrouter_client.chat.completions.create(
                messages=[{"role": "system", "content": "You output strict JSON only."}, {"role": "user", "content": prompt}],
                model="meta-llama/llama-3-8b-instruct:free", response_format={"type": "json_object"}
            )
            data = json.loads(chat.choices[0].message.content)
            for sym in data: data[sym]["summary"] = f"[OpenRouter] {data[sym].get('summary', '')}"
            return data

    except Exception as e:
        return {sym: {"sentiment": "ERROR", "summary": f"System Failure: {str(e)}"} for sym in symbols_tuple}

batched_sentiments = get_batched_ai_sentiment(tuple(selected_tickers))

# --- 6. DASHBOARD GRID & CHARTING ---
if selected_tickers:
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            st.subheader(f"{ticker} | {COMPANY_NAMES.get(ticker, ticker)}")
            
            df = fetch_market_data(ticker, timeframe)
            
            if not df.empty:
                # 1. RESTORED: TOP KPI METRICS
                latest = df.iloc[-1]
                delta_val = latest['close'] - df.iloc[-2]['close'] if len(df) > 1 else None
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"${latest['close']:.2f}", f"{delta_val:.2f}" if delta_val else None)
                c2.metric("High", f"${latest['high']:.2f}")
                c3.metric("Low", f"${latest['low']:.2f}")
                
                # 2. STATISTICAL MATH
                df['SMA_BB'] = df['close'].rolling(window=bb_window).mean()
                df['STD_BB'] = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                # 3. NEW: STACKED SUBPLOTS (Price + Volume)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

                # Top Row: Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ), row=1, col=1)

                # Bottom Row: Volume (If available from yfinance)
                if 'Volume' in df.columns:
                    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for idx, row in df.iterrows()]
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

                # Overlays
                if analysis_mode == "Trend (Moving Averages)":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], mode='lines', line=dict(color='orange', width=2), name="Fast MA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], mode='lines', line=dict(color='dodgerblue', width=2), name="Slow MA"), row=1, col=1)
                elif analysis_mode == "Anomaly Detection (Bollinger Bands)":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name="Upper"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="Lower"), row=1, col=1)

                fig.update_layout(
                    template="plotly_dark", height=chart_height, margin=dict(l=0, r=0, t=10, b=0), 
                    xaxis_rangeslider_visible=False, showlegend=False, yaxis2_showgrid=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. AI SENTIMENT
                stock_data = batched_sentiments.get(ticker, {"sentiment": "NEUTRAL", "summary": "AI data currently unavailable."})
                sentiment = stock_data.get("sentiment", "NEUTRAL")
                summary = stock_data.get("summary", "")
                
                if "BULLISH" in sentiment.upper(): st.success(f"**🤖 {sentiment}** — {summary}")
                elif "BEARISH" in sentiment.upper(): st.error(f"**🤖 {sentiment}** — {summary}")
                else: st.info(f"**🤖 {sentiment}** — {summary}")
                
                # 5. NEW: FUNDAMENTALS EXPANDER
                with st.expander("📊 View Company Fundamentals"):
                    try:
                        info = yf.Ticker(ticker).info
                        f_col1, f_col2, f_col3 = st.columns(3)
                        f_col1.metric("Market Cap", f"${info.get('marketCap', 0) / 1e9:.2f}B")
                        f_col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                        f_col3.metric("Profit Margin", f"{info.get('profitMargins', 0) * 100:.2f}%")
                    except:
                        st.caption("Fundamentals data currently unavailable.")
                    
                st.markdown("---")
            else:
                st.info(f"Awaiting telemetry for {ticker}...")
