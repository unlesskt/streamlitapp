import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go

# 1. Setup Professional Page Layout
st.set_page_config(page_title="Market Tracker Pro", layout="wide", initial_sidebar_state="expanded")

# 2. Connect to Cosmos DB
@st.cache_resource
def init_connection():
    return CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])

client = init_connection()
database = client.get_database_client("FinancialData")
container = database.get_container_client("StockTicks")

# 3. Sidebar Configuration
st.sidebar.title("📈 Control Panel")
st.sidebar.markdown("---")

# The 40 Stock Watchlist
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRWD", "ORCL", "CRM",
    "AMD", "TSM", "ASML", "JPM", "V", "MA", "BAC", "GS", "AXP", "LLY", "UNH", "JNJ", 
    "PFE", "ABBV", "MRK", "WMT", "COST", "PG", "KO", "PEP", "SBUX", "MCD", "NKE", 
    "NFLX", "XOM", "CVX", "CAT", "GE", "T", "VZ"
]

ticker = st.sidebar.selectbox("Select Asset", sorted(SYMBOLS))

if st.sidebar.button("🔄 Refresh Live Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data pulled from Azure Cosmos DB via Finnhub.")

# 4. Main Dashboard Area
st.title(f"{ticker} Live Market Data")

# Query the last 1000 ticks for the selected stock
@st.cache_data(ttl=60) # Caches for 60 seconds so it doesn't spam your database
def load_data(symbol):
    query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 1000"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

items = load_data(ticker)

if items:
    # Convert Cosmos JSON to a Pandas DataFrame
    df = pd.DataFrame(items)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp') # Sort oldest to newest for the chart

    # Get the absolute latest data point
    latest = df.iloc[-1]
    
    # Calculate price movement (Tick-to-Tick Delta)
    delta_val = None
    if len(df) > 1:
        prev = df.iloc[-2]
        delta_val = latest['price'] - prev['price']

    # --- TOP ROW: METRIC CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${latest['price']:.2f}", f"{delta_val:.2f}" if delta_val else None)
    col2.metric("Today's Open", f"${latest['open']:.2f}")
    col3.metric("Daily High", f"${latest['high']:.2f}")
    col4.metric("Daily Low", f"${latest['low']:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True) # Add some spacing

    # --- MIDDLE ROW: CANDLESTICK CHART ---
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['price'], # We use the current tick price as the 'close' for this moment
        name=ticker,
        increasing_line_color='cyan', 
        decreasing_line_color='red'
    )])

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Price (USD)",
        xaxis_title="Time (UTC)",
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False # Turns off the chunky slider at the bottom for a cleaner look
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- BOTTOM ROW: RAW DATA TABLE ---
    with st.expander("🔍 View Raw Database Ticks"):
        st.dataframe(
            df[['timestamp', 'price', 'open', 'high', 'low']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )

else:
    st.info(f"Waiting for data on {ticker}... The Azure Function hasn't fetched this one yet.")
