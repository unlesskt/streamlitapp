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
st.set_page_config(page_title="Quantitative Terminal", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS to remove top padding and tighten the UI
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# --- 2. SYSTEM CONNECTIONS ---
@st.cache_resource
def init_connections():
    cosmos_client = CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])
    database = cosmos_client.get_database_client("FinancialData")
    container = database.get_container_client("StockTicks")
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') 
    
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

# --- 3. TOP-DOWN COMMAND CENTER ---
st.title("QUANTITATIVE TERMINAL")
st.markdown("---")

# Row 1: Primary Controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([3, 2, 2, 1])

with ctrl_col1:
    selected_tickers = st.multiselect("Active Symbols", sorted(COMPANY_NAMES.keys()), default=["NVDA", "TSLA"])
with ctrl_col2:
    timeframe = st.selectbox("Resolution", ["Intraday (15m)", "Daily (1D)", "Weekly (1W)", "Raw Ticks (Cosmos DB)"], index=1)
with ctrl_col3:
    analysis_mode = st.selectbox("Statistical Overlay", ["None", "Moving Averages", "Bollinger Bands"])
with ctrl_col4:
    st.write("") # Spacing alignment
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

# Row 2: Secondary Parameters (Expandable to save space)
with st.expander("Terminal Parameters & Tuning"):
    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        fast_ma = st.number_input("Fast MA Period", value=20)
        slow_ma = st.number_input("Slow MA Period", value=50)
    with param_col2:
        bb_window = st.number_input("Bollinger Window", value=20)
        bb_std = st.number_input("Bollinger Std Dev", value=2.0, step=0.5)
    with param_col3:
        chart_height = st.slider("Viewport Height", 300, 800, 450)

st.markdown("---")

# --- 4. MACRO MARKET HEADER ---
@st.cache_data(ttl=300)
def fetch_macro_indices():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "VIX (Volatility)": "^VIX"}
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
m_col1.metric("S&P 500 (Global Macro)", f"{macro_data['S&P 500']['price']:,.2f}", f"{macro_data['S&P 500']['change']:.2f}%")
m_col2.metric("NASDAQ (Tech Macro)", f"{macro_data['NASDAQ']['price']:,.2f}", f"{macro_data['NASDAQ']['change']:.2f}%")
# Note: VIX goes up when the market is crashing. We inverse the color so a green VIX = red text.
m_col3.metric("VIX (Market Fear Index)", f"{macro_data['VIX (Volatility)']['price']:,.2f}", f"{macro_data['VIX (Volatility)']['change']:.2f}%", delta_color="inverse")
st.markdown("---")

# --- 5. DATA ENGINE ---
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf): 
    if tf == "Raw Ticks (Cosmos DB)":
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items: return pd.DataFrame()
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df[['open', 'high', 'low', 'price']].rename(columns={'price': 'close'})
    else:
        # Map user-friendly labels to yfinance syntax
        mapping = {
            "Intraday (15m)": {"period": "5d", "interval": "15m"},
            "Daily (1D)": {"period": "1y", "interval": "1d"},
            "Weekly (1W)": {"period": "5y", "interval": "1wk"}
        }
        config = mapping[tf]
        df = yf.Ticker(symbol).history(period=config["period"], interval=config["interval"])
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
        
        try:
            response = gemini_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except:
            chat = openrouter_client.chat.completions.create(
                messages=[{"role": "system", "content": "You output strict JSON only."}, {"role": "user", "content": prompt}],
                model="meta-llama/llama-3-8b-instruct:free", response_format={"type": "json_object"}
            )
            return json.loads(chat.choices[0].message.content)
    except:
        return {sym: {"sentiment": "N/A", "summary": "Analysis unavailable."} for sym in symbols_tuple}

batched_sentiments = get_batched_ai_sentiment(tuple(selected_tickers))

# --- 6. DASHBOARD GRID & CHARTING ---
if selected_tickers:
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            st.markdown(f"### {ticker} | {COMPANY_NAMES.get(ticker, ticker)}")
            
            df = fetch_market_data(ticker, timeframe)
            
            if not df.empty:
                # 1. KPI METRICS
                latest = df.iloc[-1]
                delta_val = latest['close'] - df.iloc[-2]['close'] if len(df) > 1 else None
                c1, c2, c3 = st.columns(3)
                c1.metric("Last Price", f"${latest['close']:.2f}", f"{delta_val:.2f}" if delta_val else None)
                c2.metric("Period High", f"${latest['high']:.2f}")
                c3.metric("Period Low", f"${latest['low']:.2f}")
                
                # 2. STATISTICAL MATH
                df['SMA_BB'] = df['close'].rolling(window=bb_window).mean()
                df['STD_BB'] = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                # 3. STACKED SUBPLOTS (Price + Volume)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ), row=1, col=1)

                if 'Volume' in df.columns:
                    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for idx, row in df.iterrows()]
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

                if analysis_mode == "Moving Averages":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], mode='lines', line=dict(color='#FFA726', width=1.5), name="Fast MA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], mode='lines', line=dict(color='#29B6F6', width=1.5), name="Slow MA"), row=1, col=1)
                elif analysis_mode == "Bollinger Bands":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name="Upper"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.3)', dash='dash'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="Lower"), row=1, col=1)

                fig.update_layout(
                    template="plotly_dark", height=chart_height, margin=dict(l=0, r=0, t=10, b=0), 
                    xaxis_rangeslider_visible=False, showlegend=False, yaxis2_showgrid=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. QUANTITATIVE SUMMARY (Replaces chatty AI output)
                stock_data = batched_sentiments.get(ticker, {"sentiment": "NEUTRAL", "summary": "Data unavailable."})
                sentiment = stock_data.get("sentiment", "NEUTRAL")
                summary = stock_data.get("summary", "")
                
                st.caption(f"**SENTIMENT SIGNAL: {sentiment.upper()}** | {summary}")
                    
                st.markdown("<br><br>", unsafe_allow_html=True)
            else:
                st.info(f"Awaiting telemetry for {ticker}...")
