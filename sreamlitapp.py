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

st.markdown("""
<style>
    /* ── Import fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    /* ── Root token overrides ── */
    :root {
        --qt-bg:       #0a0c0f;
        --qt-surface:  #10141a;
        --qt-border:   #1e2530;
        --qt-border2:  #2a3340;
        --qt-text:     #c8d4e0;
        --qt-muted:    #5a6a7a;
        --qt-accent:   #00d4aa;
        --qt-accent2:  #0090ff;
        --qt-bull:     #00c896;
        --qt-bear:     #ff4d6a;
        --qt-warn:     #f5a623;
        --qt-mono:     'IBM Plex Mono', monospace;
        --qt-sans:     'IBM Plex Sans', sans-serif;
    }

    /* ── Global page reset ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"] {
        background-color: var(--qt-bg) !important;
        color: var(--qt-text) !important;
        font-family: var(--qt-mono) !important;
    }

    [data-testid="block-container"] {
        padding: 0 1.5rem 2rem 1.5rem !important;
        max-width: 100% !important;
    }

    /* ── Hide Streamlit chrome ── */
    footer, #MainMenu, header, [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    /* ── Dividers ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--qt-border) !important;
        margin: 0.6rem 0 !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6,
    p, span, label, div,
    [class*="stMarkdown"] * {
        font-family: var(--qt-mono) !important;
        color: var(--qt-text) !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border) !important;
        border-radius: 0 !important;
        padding: 0.65rem 0.9rem !important;
    }
    [data-testid="metric-container"] {
        background: transparent !important;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: var(--qt-muted) !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: var(--qt-text) !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricDelta"] svg { display: none !important; }
    [data-testid="stMetricDelta"] > div {
        font-size: 11px !important;
        font-family: var(--qt-mono) !important;
    }

    /* ── Select / multiselect ── */
    [data-testid="stMultiSelect"] > div > div,
    [data-testid="stSelectbox"] > div > div {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border2) !important;
        border-radius: 0 !important;
        color: var(--qt-text) !important;
        font-family: var(--qt-mono) !important;
        font-size: 12px !important;
    }
    [data-testid="stMultiSelect"] span,
    [data-baseweb="tag"] {
        background: #1a2535 !important;
        border: 1px solid var(--qt-accent) !important;
        border-radius: 0 !important;
        color: var(--qt-accent) !important;
        font-size: 11px !important;
        font-family: var(--qt-mono) !important;
    }

    /* ── Number inputs ── */
    [data-testid="stNumberInput"] input {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border2) !important;
        border-radius: 0 !important;
        color: var(--qt-text) !important;
        font-family: var(--qt-mono) !important;
        font-size: 12px !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] > div > div > div {
        background: var(--qt-accent) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--qt-surface) !important;
        border: 1px solid var(--qt-border) !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 11px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--qt-muted) !important;
        font-family: var(--qt-mono) !important;
    }

    /* ── Buttons ── */
    [data-testid="stButton"] > button {
        background: transparent !important;
        border: 1px solid var(--qt-border2) !important;
        border-radius: 0 !important;
        color: var(--qt-accent) !important;
        font-family: var(--qt-mono) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stButton"] > button:hover {
        background: rgba(0, 212, 170, 0.08) !important;
        border-color: var(--qt-accent) !important;
    }

    /* ── Info box ── */
    [data-testid="stInfo"] {
        background: #0d1520 !important;
        border-left: 3px solid var(--qt-accent2) !important;
        border-radius: 0 !important;
        color: var(--qt-muted) !important;
        font-size: 12px !important;
    }

    /* ── Column spacing ── */
    [data-testid="stHorizontalBlock"] {
        gap: 12px !important;
        align-items: flex-start !important;
    }

    /* ── Caption override ── */
    [data-testid="stCaptionContainer"] {
        background: var(--qt-surface) !important;
        border-left: 3px solid var(--qt-border2) !important;
        padding: 0.5rem 0.8rem !important;
        font-size: 11px !important;
        color: var(--qt-muted) !important;
    }

    /* ── Plotly chart container ── */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--qt-border) !important;
    }

    /* ── Label text ── */
    [data-testid="stWidgetLabel"] p {
        font-size: 10px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--qt-muted) !important;
    }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--qt-bg); }
    ::-webkit-scrollbar-thumb { background: var(--qt-border2); }
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
utc_str    = now.strftime('%Y-%m-%d&nbsp;&nbsp;%H:%M:%S')

header_html = (
    '<div style="display:flex;justify-content:space-between;align-items:center;'
    'padding:0.6rem 0 0.5rem 0;border-bottom:1px solid #1e2530;margin-bottom:0.8rem;">'
    '<div style="display:flex;align-items:center;gap:1rem;">'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;font-weight:600;color:#00d4aa;letter-spacing:0.18em;">&#9632; QT TERMINAL</span>'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#1e2530;">|</span>'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#5a6a7a;letter-spacing:0.08em;">QUANTITATIVE ANALYSIS PLATFORM v2.0</span>'
    '</div>'
    '<div style="display:flex;align-items:center;gap:1.5rem;">'
    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#5a6a7a;">SESSION <span style="color:#00d4aa;">{session_id}</span></span>'
    f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#5a6a7a;">UTC <span style="color:#c8d4e0;">{utc_str}</span></span>'
    '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:#00c896;letter-spacing:0.1em;border:1px solid #00c896;padding:2px 8px;">&#9679; LIVE</span>'
    '</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. COMMAND BAR
# ─────────────────────────────────────────────

# CSS: constrain multiselect dropdown height and clip overflow tags
st.markdown("""
<style>
    /* Keep selected tags on one line with scroll — prevents vertical explosion */
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
        max-height: 42px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        flex-wrap: nowrap !important;
    }
    /* Limit tag count overflow gracefully */
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        max-width: 80px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    /* Ensure expander never overlaps widgets above it */
    [data-testid="stExpander"] {
        margin-top: 0.5rem !important;
        clear: both !important;
    }
</style>
""", unsafe_allow_html=True)

# Row 1: Symbol picker (full width)
selected_tickers = st.multiselect(
    "ACTIVE SYMBOLS  —  select up to 6 for best layout",
    sorted(COMPANY_NAMES.keys()),
    default=["NVDA", "TSLA"],
    max_selections=6,
)

# Row 2: Resolution | Overlay | Refresh
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([2, 2, 2, 1])
with row2_col1:
    timeframe = st.selectbox(
        "Resolution",
        ["Intraday (15m)", "Daily (1D)", "Weekly (1W)", "Raw Ticks (Cosmos DB)"],
        index=1
    )
with row2_col2:
    analysis_mode = st.selectbox(
        "Overlay",
        ["None", "Moving Averages", "Bollinger Bands"]
    )
with row2_col3:
    chart_height = st.select_slider(
        "Chart Height",
        options=[300, 350, 400, 450, 500, 550, 600],
        value=450
    )
with row2_col4:
    st.markdown("<div style='padding-top:1.6rem;'></div>", unsafe_allow_html=True)
    if st.button("↺ Refresh", use_container_width=True):
        st.rerun()

# Row 3: Collapsed advanced params
with st.expander("⚙  Advanced Parameters"):
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        fast_ma = st.number_input("Fast MA Period", value=20)
        slow_ma = st.number_input("Slow MA Period", value=50)
    with param_col2:
        bb_window = st.number_input("Bollinger Window", value=20)
        bb_std    = st.number_input("Bollinger Std Dev", value=2.0, step=0.5)

st.markdown("<div style='margin-bottom:0.8rem'></div>", unsafe_allow_html=True)


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

def _macro_row_html(name, price, change, invert=False):
    """Renders a compact macro index tile as a single-line HTML string."""
    positive = change >= 0
    if invert:
        positive = not positive
    color  = "#00c896" if positive else "#ff4d6a"
    sign   = "+" if change >= 0 else ""
    arrow  = "&#9650;" if change >= 0 else "&#9660;"  # ▲ ▼ as HTML entities
    price_fmt  = f"{price:,.2f}"
    change_fmt = f"{sign}{change:.2f}%"
    return (
        '<div style="background:#10141a;border:1px solid #1e2530;padding:0.7rem 1rem;flex:1;">'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;letter-spacing:0.12em;color:#5a6a7a;margin-bottom:4px;text-transform:uppercase;">{name}</div>'
        '<div style="display:flex;align-items:baseline;gap:0.6rem;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:18px;font-weight:600;color:#c8d4e0;letter-spacing:-0.02em;">{price_fmt}</span>'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:500;color:{color};">{arrow} {change_fmt}</span>'
        '</div></div>'
    )

sp_html   = _macro_row_html("S&P 500",    macro_data['S&P 500']['price'], macro_data['S&P 500']['change'])
nas_html  = _macro_row_html("NASDAQ",     macro_data['NASDAQ']['price'],  macro_data['NASDAQ']['change'])
vix_html  = _macro_row_html("VIX ●FEAR", macro_data['VIX']['price'],     macro_data['VIX']['change'], invert=True)
dow_html  = _macro_row_html("DOW JONES", macro_data['DOW']['price'],      macro_data['DOW']['change'])

macro_band_html = (
    '<div style="display: flex; gap: 12px; margin-bottom: 0.8rem;">'
    + sp_html + nas_html + dow_html + vix_html
    + '</div>'
    + '<div style="height: 1px; background: #1e2530; margin-bottom: 0.9rem;"></div>'
)
st.markdown(macro_band_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. DATA ENGINE
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_market_data(symbol, tf):
    if tf == "Raw Ticks (Cosmos DB)":
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        if not items:
            return pd.DataFrame()
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
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df[['open', 'high', 'low', 'close', 'Volume']]


@st.cache_data(ttl=3600)
def get_batched_ai_sentiment(symbols_tuple):
    if not symbols_tuple:
        return {}
    try:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        past  = (datetime.datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        master_news_dict = {}
        for symbol in symbols_tuple:
            url = (
                f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
                f"&from={past}&to={today}&token={st.secrets['FINNHUB_API_KEY']}"
            )
            news_data = requests.get(url).json()
            master_news_dict[symbol] = (
                [a['headline'] for a in news_data[:5]] if news_data else ["No recent news."]
            )
        prompt = f"""
Analyze the sentiment for EACH stock based on these headlines: {master_news_dict}
Format your response as a strictly valid JSON dictionary where the keys are the stock tickers,
and the values are objects with "sentiment" (BULLISH, BEARISH, or NEUTRAL) and
"summary" (1 strict sentence explanation).
        """
        try:
            response = gemini_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except:
            chat = openrouter_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user",   "content": prompt}
                ],
                model="meta-llama/llama-3-8b-instruct:free",
                response_format={"type": "json_object"}
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
    margin=dict(l=0, r=0, t=4, b=0),
    xaxis_rangeslider_visible=False,
    showlegend=False,
    xaxis=dict(
        gridcolor="#141a22",
        linecolor="#1e2530",
        tickfont=dict(size=9, color="#5a6a7a"),
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="#141a22",
        linecolor="#1e2530",
        tickfont=dict(size=9, color="#5a6a7a"),
        showgrid=True,
        side="right",
    ),
    yaxis2=dict(
        gridcolor="rgba(0,0,0,0)",
        showgrid=False,
        tickfont=dict(size=9, color="#5a6a7a"),
        side="right",
    ),
)


# ─────────────────────────────────────────────
# 8. SENTIMENT BADGE HELPER
# ─────────────────────────────────────────────
def sentiment_badge(sentiment):
    cfg = {
        "BULLISH": ("&#9650; BULLISH", "#00c896", "rgba(0,200,150,0.12)",  "1px solid rgba(0,200,150,0.4)"),
        "BEARISH": ("&#9660; BEARISH", "#ff4d6a", "rgba(255,77,106,0.12)", "1px solid rgba(255,77,106,0.4)"),
        "NEUTRAL": ("&#9670; NEUTRAL", "#f5a623", "rgba(245,166,35,0.10)", "1px solid rgba(245,166,35,0.4)"),
    }
    label, color, bg, border = cfg.get(sentiment.upper(), cfg["NEUTRAL"])
    return (
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;font-weight:600;'
        f'letter-spacing:0.12em;color:{color};background:{bg};border:{border};'
        f'padding:3px 10px;white-space:nowrap;">{label}</span>'
    )


# ─────────────────────────────────────────────
# 9. DASHBOARD GRID
# ─────────────────────────────────────────────
if not selected_tickers:
    st.markdown(
        '<div style="text-align:center;padding:4rem 0;font-family:\'IBM Plex Mono\',monospace;color:#2a3340;font-size:13px;letter-spacing:0.1em;">NO SYMBOLS ACTIVE &#8212; SELECT FROM COMMAND BAR</div>',
        unsafe_allow_html=True
    )
else:
    n_cols = 1 if len(selected_tickers) == 1 else 2
    cols = st.columns(n_cols)

    for index, ticker in enumerate(selected_tickers):
        with cols[index % 2]:
            company = COMPANY_NAMES.get(ticker, ticker)

            # ── Panel header ─────────────────────
            panel_header = (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'background:#10141a;border:1px solid #1e2530;border-bottom:2px solid #00d4aa;'
                'padding:0.55rem 0.9rem;margin-bottom:0;">'
                '<div>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:15px;font-weight:600;color:#00d4aa;letter-spacing:0.06em;">{ticker}</span>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#5a6a7a;margin-left:0.7rem;letter-spacing:0.04em;">{company.upper()}</span>'
                '</div>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:9px;color:#2a3340;letter-spacing:0.08em;">{timeframe}</span>'
                '</div>'
            )
            st.markdown(panel_header, unsafe_allow_html=True)

            df = fetch_market_data(ticker, timeframe)

            if not df.empty:
                latest  = df.iloc[-1]
                prev    = df.iloc[-2] if len(df) > 1 else latest
                delta_v = latest['close'] - prev['close']
                delta_p = (delta_v / prev['close']) * 100 if prev['close'] else 0
                period_range = latest['high'] - latest['low']

                # ── KPI strip ────────────────────
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Last",
                    f"${latest['close']:,.2f}",
                    f"{'+' if delta_v >= 0 else ''}{delta_v:.2f} ({delta_p:+.2f}%)"
                )
                c2.metric("High",  f"${latest['high']:,.2f}")
                c3.metric("Low",   f"${latest['low']:,.2f}")
                c4.metric("Range", f"${period_range:.2f}")

                # ── Statistical overlays ─────────
                df['SMA_BB']     = df['close'].rolling(window=bb_window).mean()
                df['STD_BB']     = df['close'].rolling(window=bb_window).std()
                df['Upper_Band'] = df['SMA_BB'] + (df['STD_BB'] * bb_std)
                df['Lower_Band'] = df['SMA_BB'] - (df['STD_BB'] * bb_std)

                # ── Subplots ─────────────────────
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.015,
                    row_heights=[0.78, 0.22]
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['open'], high=df['high'],
                    low=df['low'],   close=df['close'],
                    name="Price",
                    increasing_line_color='#00c896',
                    decreasing_line_color='#ff4d6a',
                    increasing_fillcolor='rgba(0,200,150,0.25)',
                    decreasing_fillcolor='rgba(255,77,106,0.25)',
                    line=dict(width=1),
                    whiskerwidth=0,
                ), row=1, col=1)

                # Volume bars
                if 'Volume' in df.columns:
                    bar_colors = [
                        'rgba(0,200,150,0.35)' if row['close'] >= row['open'] else 'rgba(255,77,106,0.35)'
                        for _, row in df.iterrows()
                    ]
                    fig.add_trace(go.Bar(
                        x=df.index, y=df['Volume'],
                        marker_color=bar_colors,
                        name="Volume"
                    ), row=2, col=1)

                # Overlays
                if analysis_mode == "Moving Averages":
                    df['SMA_Fast'] = df['close'].rolling(window=fast_ma).mean()
                    df['SMA_Slow'] = df['close'].rolling(window=slow_ma).mean()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_Fast'],
                        mode='lines',
                        line=dict(color='#f5a623', width=1.2, dash='solid'),
                        name=f"MA{fast_ma}"
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['SMA_Slow'],
                        mode='lines',
                        line=dict(color='#0090ff', width=1.2, dash='solid'),
                        name=f"MA{slow_ma}"
                    ), row=1, col=1)
                elif analysis_mode == "Bollinger Bands":
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Upper_Band'],
                        mode='lines',
                        line=dict(color='rgba(0,212,170,0.4)', width=1, dash='dot'),
                        name="Upper"
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['Lower_Band'],
                        mode='lines',
                        line=dict(color='rgba(0,212,170,0.4)', width=1, dash='dot'),
                        fill='tonexty',
                        fillcolor='rgba(0,212,170,0.04)',
                        name="Lower"
                    ), row=1, col=1)

                # Apply theme
                layout = {**CHART_LAYOUT, "height": chart_height}
                fig.update_layout(**layout)
                fig.update_xaxes(
                    gridcolor="#141a22", linecolor="#1e2530",
                    tickfont=dict(size=9, color="#5a6a7a"),
                )
                fig.update_yaxes(
                    gridcolor="#141a22", linecolor="#1e2530",
                    tickfont=dict(size=9, color="#5a6a7a"),
                    side="right"
                )

                st.plotly_chart(fig, use_container_width=True)

                # ── Sentiment footer ──────────────
                stock_data = batched_sentiments.get(ticker, {"sentiment": "NEUTRAL", "summary": "Data unavailable."})
                sentiment  = stock_data.get("sentiment", "NEUTRAL")
                summary    = stock_data.get("summary", "")

                badge_html = sentiment_badge(sentiment)
                footer_html = (
                    '<div style="display:flex;align-items:flex-start;gap:0.8rem;'
                    'background:#0d1117;border:1px solid #1e2530;border-top:none;'
                    'padding:0.55rem 0.9rem;margin-bottom:1.2rem;">'
                    + badge_html
                    + '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
                    'color:#5a6a7a;line-height:1.5;margin-top:2px;">'
                    + summary
                    + '</span></div>'
                )
                st.markdown(footer_html, unsafe_allow_html=True)

            else:
                st.markdown(
                    f'<div style="background:#10141a;border:1px solid #1e2530;border-top:none;padding:2rem;text-align:center;font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:#2a3340;letter-spacing:0.1em;margin-bottom:1.2rem;">AWAITING TELEMETRY &nbsp;·&nbsp; {ticker}</div>',
                    unsafe_allow_html=True
                )
