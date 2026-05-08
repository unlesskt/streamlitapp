import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go
import datetime
from datetime import timedelta

# 1. Strict Professional Layout
st.set_page_config(page_title="Market Monitor", layout="wide", initial_sidebar_state="expanded")

# 2. Database Connection
@st.cache_resource
def init_connection():
    return CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])

client = init_connection()
database = client.get_database_client("FinancialData")
container = database.get_container_client("StockTicks")

# --- NEW: Company Name Dictionary ---
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

# 3. Sidebar Configuration
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

selected_tickers = st.sidebar.multiselect(
    "Select Assets to Monitor", 
    sorted(COMPANY_NAMES.keys()), 
    default=["NVDA", "TSLA", "AAPL", "MSFT"]
)

# --- NEW: Time Filters and Chart Type ---
time_filter = st.sidebar.radio(
    "Time Range",
    ("Last 1 Hour", "Last 3 Hours", "Last 24 Hours", "All Data")
)

chart_type = st.sidebar.radio(
    "Chart Type",
    ("Line Chart (Clean)", "Candlestick (Detailed)")
)

if st.sidebar.button("Refresh Live Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("System: Azure Cosmos DB | Feed: Finnhub")

# 4. Main Dashboard Area
st.title("Live Market Monitoring")
st.markdown("---")

# Calculate the time cutoff based on user selection
time_deltas = {
    "Last 1 Hour": timedelta(hours=1),
    "Last 3 Hours": timedelta(hours=3),
    "Last 24 Hours": timedelta(hours=24),
    "All Data": None
}
selected_delta = time_deltas[time_filter]

# Fetch data with the time filter applied
@st.cache_data(ttl=60)
def load_data(symbol, time_range_str): 
    # We pass time_range_str just to force the cache to update when the radio button changes
    delta = time_deltas[time_range_str]
    
    if delta:
        cutoff_time = (datetime.datetime.utcnow() - delta).isoformat()
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' AND c.timestamp >= '{cutoff_time}' ORDER BY c.timestamp DESC"
    else:
        query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC LIMIT 1000"
        
    return list(container.query_items(query=query, enable_cross_partition_query=True))

if not selected_tickers:
    st.warning("System Standby: Please select at least one asset from the control panel.")
else:
    # 5. Build the Dynamic Grid
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        col = cols[index % 2]
        
        with col:
            # Display Ticker AND Company Name
            company_name = COMPANY_NAMES.get(ticker, "Unknown Company")
            st.subheader(f"{ticker} | {company_name}")
            
            items = load_data(ticker, time_filter)
            
            if items:
                df = pd.DataFrame(items)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                latest = df.iloc[-1]
                delta_val = None
                if len(df) > 1:
                    prev = df.iloc[-2]
                    delta_val = latest['price'] - prev['price']
                    
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"${latest['price']:.2f}", f"{delta_val:.2f}" if delta_val else None)
                m2.metric("High", f"${latest['high']:.2f}")
                m3.metric("Low", f"${latest['low']:.2f}")
                
                # --- NEW: Toggle between Line and Candlestick ---
                if chart_type == "Line Chart (Clean)":
                    fig = go.Figure(data=[go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        line=dict(color='#00E5FF', width=2), # Cyan professional line
                        fill='tozeroy', # Adds a subtle shading under the line
                        fillcolor='rgba(0, 229, 255, 0.1)'
                    )])
                else:
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['price'],
                        increasing_line_color='#26a69a', 
                        decreasing_line_color='#ef5350'  
                    )])
                
                fig.update_layout(
                    template="plotly_dark",
                    height=350, 
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_rangeslider_visible=False,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
            else:
                st.info(f"No data available for {ticker} in the selected time range.")
