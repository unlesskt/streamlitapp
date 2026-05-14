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

# ─────────────────────────────────────────────
# 1. PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Quantitative Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. SYSTEM CONNECTIONS
# ─────────────────────────────────────────────
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
    "VZ": "Verizon Communications", "FSM": "FSM", "INDI": "INDI"
}

# ─────────────────────────────────────────────
# 3. TERMINAL HEADER
# ─────────────────────────────────────────────
now = datetime.datetime.utcnow()
session_id = f"SES-{now.strftime('%y%m%d%H%M')}"
utc_str    = now.strftime('%Y-%m-%d  %H:%M:%S')

h_col1, h_col2 = st.columns([1, 1])
with h_col1:
    st.markdown(f"### ⬛ QT TERMINAL <span style='font-size: 14px; color: gray;'>| Quantitative Analysis v2.0</span>", unsafe_allow_html=True)
with h_col2:
    st.markdown(f"<div style='text-align: right; color: gray; font-size: 14px; padding-top: 10px;'>SESSION {session_id} &nbsp;|&nbsp; UTC {utc_str} &nbsp;|&nbsp; <span style='color: #00d4aa;'>● LIVE</span></div>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# 4. COMMAND BAR & GLOBALS
# ─────────────────────────────────────────────
selected_tickers = st.multiselect(
    "Active Symbols",
    sorted(COMPANY_NAMES.keys()),
    default=["NVDA", "TSLA"]
)

row2_col1, row2_col2, row2_col3 = st.columns([3, 3, 1])
with row2_col1:
    timeframe = st.selectbox(
        "Resolution", 
        [
            "Raw Ticks (Unaggregated Stream)",
            "Resampled Ticks (Hourly)", 
            "Resampled Ticks (Daily)", 
            "Resampled Ticks (Weekly)",
            "Weekly (yfinance)"
        ],
        index=0
    )
with row2_col2:
    analysis_mode = st.selectbox("Overlay", ["None", "Moving Averages", "Bollinger Bands"], index=1)
with row2_col3:
    st.write("")
    st.write("")
    if st.button("↺ Refresh", use_container_width=True):
        st.rerun()

# Hardcoded standard parameters
fast_ma, slow_ma = 20, 50
bb_window, bb_std = 20, 2.0
chart_height = 400

st.divider()

# ─────────────────────────────────────────────
# 5. MACRO INDICES BAND
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_macro_indices():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW JONES": "^DJI", "VIX (FEAR INDEX)": "^VIX"}
    data = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            latest = hist.iloc[-1]['Close']
            prev   = hist.iloc[-2]['Close']
            pct    = ((latest - prev) / prev) * 100
            data[name] = {"price": latest, "change": pct}
        except:
            data[name] = {"price": 0.0, "change": 0.0}
    return data

macro_data = fetch_macro_indices()

m_c1, m_c2, m_c3, m_c4 = st.columns(4)
m_c1.metric("S&P 500", f"{macro_data['S&P 500']['price']:,.2f}", f"{macro_data['S&P 500']['change']:.2f}%")
m_c2.metric("NASDAQ", f"{macro_data['NASDAQ']['price']:,.2f}", f"{macro_data['NASDAQ']['change']:.2f}%")
m_c3.metric("DOW JONES", f"{macro_data['DOW JONES']['price']:,.2f}", f"{macro_data['DOW JONES']['change']:.2f}%")
m_c4.metric("VIX (FEAR INDEX)", f"{macro_data['VIX (FEAR INDEX)']['price']:,.2f}", f"{macro_data['VIX (FEAR INDEX)']['change']:.2f}%", delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# 6. DATA ENGINE
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf):
    if "Ticks" in tf:
        # Massive limit increase to capture multiple days/weeks of raw tick flow
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 25000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items: return pd.DataFrame()
        
        df = pd.DataFrame(items)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df[['open', 'high', 'low', 'price']].rename(columns={'price': 'close'})
        
        # If user explicitly wants resampled candles from the DB
        if "Resampled" in tf:
            resample_map = {"Hourly": "h", "Daily": "D", "Weekly": "W"}
            freq = next((v for k, v in resample_map.items() if k in tf), None)
            if freq:
                df = df.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        else:
            # PURE RAW TICKS (Keep as 1D array of close prices)
            df = df[['close']]
            
        return df
        
    elif tf == "Weekly (yfinance)":
        df = yf.Ticker(symbol).history(period="5y", interval="1wk")
        if df.empty: return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df[['open', 'high', 'low', 'close', 'Volume']]

@st.cache_data(ttl=3600)
def get_batched_ai_sentiment(symbols_tuple):
    if not symbols_tuple: return {}
    try:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        past  = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        master_news_dict = {}
        for symbol in symbols_tuple:
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={past}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
            news_data = requests.get(url).json()
            master_news_dict[symbol] = [a['headline'] for a in news_data[:5]] if news_data else ["No recent news."]
            
        prompt = f"""
        Analyze the sentiment for EACH stock based on these headlines: {master_news_dict}
        Format your response as a strictly valid JSON dictionary where the keys are the stock tickers,
        and the values are objects with "sentiment" (BULLISH, BEARISH, or NEUTRAL) and
        "summary" (1 strict sentence explanation).
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

# ─────────────────────────────────────────────
# 7. CHART THEME HELPER
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis_rangeslider_visible=False,
    showlegend=False,
)

# ─────────────────────────────────────────────
# 8. DASHBOARD GRID
# ─────────────────────────────────────────────
if not selected_tickers:
    st.info("No symbols active. Please select assets from the command bar above.")
else:
    n_cols = 1 if len(selected_tickers) == 1 else 2
    cols = st.columns(n_cols)

    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            company = COMPANY_NAMES.get(ticker, ticker)
            
            st.subheader(f"{ticker} | {company}")

            df = fetch_market_data(ticker, timeframe)

            if not df.empty:
                latest  = df.iloc[-1]
                prev    = df.iloc[-2] if len(df) > 1 else latest
                delta_v = latest['close'] - prev['close']
                delta_p = (delta_v / prev['close']) * 100 if prev['close'] else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Last", f"${latest['close']:,.2f}", f"{'+' if delta_v >= 0 else ''}{delta_v:.2f} ({delta_p:+.2f}%)")
                
                # Raw ticks don't have High/Low columns built in, so we dynamically calculate the period highs/lows
                period_high = df['high'].max() if 'high' in df.columns else df['close'].max()
                period_low = df['low'].min() if 'low' in df.columns else df['close'].min()
                
                c2.metric("Period High",  f"${period_high:,.2f}")
                c3.metric("Period Low",   f"${period_low:,.2f}")

                df['SMA_BB']     = df['close'].rolling(window=bb_window).mean()
                df['STD_BB']     = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                has_vol = 'Volume' in df.columns
                is_raw = timeframe == "Raw Ticks (Unaggregated Stream)"
                
                fig = make_subplots(
                    rows=2 if has_vol else 1, 
                    cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_heights=[0.75, 0.25] if has_vol else [1.0]
                )

                # High-performance plotting: Scattergl for Raw Ticks, Candlesticks for Resampled/Weekly
                if is_raw:
                    fig.add_trace(go.Scattergl(
                        x=df.index, y=df['close'], 
                        mode='lines', line=dict(color='#00d4aa', width=1.5), 
                        name="Tick Price"
                    ), row=1, col=1)
                else:
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                        name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                    ), row=1, col=1)

                if has_vol:
                    bar_colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in df.iterrows()]
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=bar_colors, opacity=0.7, name="Volume"), row=2, col=1)
                    fig.update_yaxes(showgrid=False, side="right", row=2, col=1)

                if analysis_mode == "Moving Averages":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], mode='lines', line=dict(color='orange', width=1.5), name="Fast MA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], mode='lines', line=dict(color='dodgerblue', width=1.5), name="Slow MA"), row=1, col=1)
                elif analysis_mode == "Bollinger Bands":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.4)', dash='dot'), name="Upper"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(255,255,255,0.4)', dash='dot'), fill='tonexty', fillcolor='rgba(255,255,255,0.05)', name="Lower"), row=1, col=1)

                layout = {**CHART_LAYOUT, "height": chart_height}
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

                # Sentiment block
                stock_data = batched_sentiments.get(ticker, {"sentiment": "NEUTRAL", "summary": "Data unavailable."})
                sentiment  = stock_data.get("sentiment", "NEUTRAL").upper()
                summary    = stock_data.get("summary", "")

                if "BULL" in sentiment:
                    st.success(f"**▲ {sentiment}** | {summary}")
                elif "BEAR" in sentiment:
                    st.error(f"**▼ {sentiment}** | {summary}")
                else:
                    st.info(f"**■ {sentiment}** | {summary}")

                st.markdown("<br><br>", unsafe_allow_html=True)

            else:
                st.warning(f"AWAITING TELEMETRY · {ticker}")
