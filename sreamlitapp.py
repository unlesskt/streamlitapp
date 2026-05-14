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
    page_title="QT Terminal",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safely styled CSS (Removed layout-breaking flexbox overrides)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    :root {
        --qt-bg:       #0a0c0f;
        --qt-surface:  #10141a;
        --qt-border:   #1e2530;
        --qt-border2:  #2a3340;
        --qt-text:     #c8d4e0;
        --qt-muted:    #5a6a7a;
        --qt-accent:   #00d4aa;
        --qt-accent2:  #0090ff;
        --qt-mono:     'IBM Plex Mono', monospace;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="block-container"] {
        background-color: var(--qt-bg) !important;
        color: var(--qt-text) !important;
        font-family: var(--qt-mono) !important;
    }

    /* Hide Streamlit chrome cleanly */
    footer, header { display: none !important; }

    /* Typography */
    h1, h2, h3, h4, p, span, label, div, [class*="stMarkdown"] * {
        font-family: var(--qt-mono) !important;
        color: var(--qt-text) !important;
    }

    /* Metric cards - Safely styled without breaking padding */
    [data-testid="stMetric"] {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border) !important;
        border-radius: 4px !important;
        padding: 10px !important;
    }
    
    [data-testid="stMetricLabel"] > div {
        color: var(--qt-muted) !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border) !important;
        margin-top: 1rem !important;
    }

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
    "VZ": "Verizon Communications"
}

# ─────────────────────────────────────────────
# 3. TERMINAL HEADER
# ─────────────────────────────────────────────
now = datetime.datetime.utcnow()
session_id = f"SES-{now.strftime('%y%m%d%H%M')}"
utc_str    = now.strftime('%Y-%m-%d  %H:%M:%S')

header_html = (
    '<div style="display:flex;justify-content:space-between;align-items:center;'
    'padding:0.6rem 0 0.5rem 0;border-bottom:1px solid #1e2530;margin-bottom:1rem;">'
    '<div style="display:flex;align-items:center;gap:1rem;">'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;font-weight:600;color:#00d4aa;">&#9632; QT TERMINAL</span>'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#5a6a7a;">QUANTITATIVE ANALYSIS PLATFORM v2.0</span>'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1.5rem;">'
    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#5a6a7a;">SESSION <span style="color:#00d4aa;">{session_id}</span></span>'
    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#5a6a7a;">UTC <span style="color:#c8d4e0;">{utc_str}</span></span>'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#00c896;border:1px solid #00c896;padding:2px 6px;">&#9679; LIVE</span>'
    '</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. COMMAND BAR
# ─────────────────────────────────────────────
selected_tickers = st.multiselect(
    "ACTIVE SYMBOLS  —  select up to 6",
    sorted(COMPANY_NAMES.keys()),
    default=["NVDA", "TSLA"],
    max_selections=6,
)

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([2, 2, 2, 1])
with row2_col1:
    timeframe = st.selectbox("Resolution", ["Intraday (15m)", "Daily (1D)", "Weekly (1W)", "Raw Ticks (Cosmos DB)"], index=1)
with row2_col2:
    analysis_mode = st.selectbox("Overlay", ["None", "Moving Averages", "Bollinger Bands"])
with row2_col3:
    chart_height = st.select_slider("Chart Height", options=[300, 350, 400, 450, 500, 550, 600], value=400)
with row2_col4:
    st.write("") # Spacer
    st.write("") # Spacer
    if st.button("↺ Refresh", use_container_width=True):
        st.rerun()

with st.expander("⚙  Advanced Parameters"):
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        fast_ma = st.number_input("Fast MA Period", value=20)
        slow_ma = st.number_input("Slow MA Period", value=50)
    with param_col2:
        bb_window = st.number_input("Bollinger Window", value=20)
        bb_std    = st.number_input("Bollinger Std Dev", value=2.0, step=0.5)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. MACRO INDICES BAND
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_macro_indices():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX", "DOW": "^DJI"}
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

# Using native Streamlit metrics for stability instead of raw HTML blocks
m_c1, m_c2, m_c3, m_c4 = st.columns(4)
m_c1.metric("S&P 500", f"{macro_data['S&P 500']['price']:,.2f}", f"{macro_data['S&P 500']['change']:.2f}%")
m_c2.metric("NASDAQ", f"{macro_data['NASDAQ']['price']:,.2f}", f"{macro_data['NASDAQ']['change']:.2f}%")
m_c3.metric("DOW JONES", f"{macro_data['DOW']['price']:,.2f}", f"{macro_data['DOW']['change']:.2f}%")
m_c4.metric("VIX (FEAR INDEX)", f"{macro_data['VIX']['price']:,.2f}", f"{macro_data['VIX']['change']:.2f}%", delta_color="inverse")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. DATA ENGINE
# ─────────────────────────────────────────────
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
        mapping = {
            "Intraday (15m)": {"period": "5d",  "interval": "15m"},
            "Daily (1D)":     {"period": "1y",  "interval": "1d"},
            "Weekly (1W)":    {"period": "5y",  "interval": "1wk"},
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
    paper_bgcolor="#0a0c0f",
    plot_bgcolor="#0a0c0f",
    font=dict(family="IBM Plex Mono", size=10, color="#5a6a7a"),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis_rangeslider_visible=False,
    showlegend=False,
)

# ─────────────────────────────────────────────
# 8. DASHBOARD GRID
# ─────────────────────────────────────────────
if not selected_tickers:
    st.info("NO SYMBOLS ACTIVE — SELECT FROM COMMAND BAR")
else:
    n_cols = 1 if len(selected_tickers) == 1 else 2
    cols = st.columns(n_cols)

    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            company = COMPANY_NAMES.get(ticker, ticker)
            
            # Safe Native Header
            st.markdown(f"### <span style='color:#00d4aa;'>{ticker}</span> | {company}", unsafe_allow_html=True)

            df = fetch_market_data(ticker, timeframe)

            if not df.empty:
                latest  = df.iloc[-1]
                prev    = df.iloc[-2] if len(df) > 1 else latest
                delta_v = latest['close'] - prev['close']
                delta_p = (delta_v / prev['close']) * 100 if prev['close'] else 0

                c1, c2, c3 = st.columns(3)
                c1.metric("Last", f"${latest['close']:,.2f}", f"{'+' if delta_v >= 0 else ''}{delta_v:.2f} ({delta_p:+.2f}%)")
                c2.metric("High",  f"${latest['high']:,.2f}")
                c3.metric("Low",   f"${latest['low']:,.2f}")

                df['SMA_BB']     = df['close'].rolling(window=bb_window).mean()
                df['STD_BB']     = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                    name="Price", increasing_line_color='#00c896', decreasing_line_color='#ff4d6a'
                ), row=1, col=1)

                if 'Volume' in df.columns:
                    bar_colors = ['#00c896' if row['close'] >= row['open'] else '#ff4d6a' for _, row in df.iterrows()]
                    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=bar_colors, opacity=0.5, name="Volume"), row=2, col=1)

                if analysis_mode == "Moving Averages":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Fast'], mode='lines', line=dict(color='#f5a623', width=1.5), name="Fast MA"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Slow'], mode='lines', line=dict(color='#0090ff', width=1.5), name="Slow MA"), row=1, col=1)
                elif analysis_mode == "Bollinger Bands":
                    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', line=dict(color='rgba(0,212,170,0.4)', dash='dot'), name="Upper"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', line=dict(color='rgba(0,212,170,0.4)', dash='dot'), fill='tonexty', fillcolor='rgba(0,212,170,0.05)', name="Lower"), row=1, col=1)

                layout = {**CHART_LAYOUT, "height": chart_height}
                fig.update_layout(**layout)
                fig.update_xaxes(gridcolor="#141a22", showgrid=True)
                fig.update_yaxes(gridcolor="#141a22", showgrid=True, side="right")

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
                    st.warning(f"**◆ {sentiment}** | {summary}")

                st.markdown("<br>", unsafe_allow_html=True)

            else:
                st.warning(f"AWAITING TELEMETRY · {ticker}")
