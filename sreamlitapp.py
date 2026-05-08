import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go

# 1. Strict Professional Layout
st.set_page_config(page_title="Market Monitor", layout="wide", initial_sidebar_state="expanded")

# 2. Database Connection
@st.cache_resource
def init_connection():
    return CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])

client = init_connection()
database = client.get_database_client("FinancialData")
container = database.get_container_client("StockTicks")

# 3. Sidebar Configuration (Clean, No Emojis)
st.sidebar.title("Control Panel")
st.sidebar.markdown("---")

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRWD", "ORCL", "CRM",
    "AMD", "TSM", "ASML", "JPM", "V", "MA", "BAC", "GS", "AXP", "LLY", "UNH", "JNJ", 
    "PFE", "ABBV", "MRK", "WMT", "COST", "PG", "KO", "PEP", "SBUX", "MCD", "NKE", 
    "NFLX", "XOM", "CVX", "CAT", "GE", "T", "VZ"
]

# Allow multiple selections for a parallel Grafana-style grid
selected_tickers = st.sidebar.multiselect(
    "Select Assets to Monitor", 
    sorted(SYMBOLS), 
    default=["NVDA", "TSLA", "AAPL", "MSFT"] # Defaults to a 2x2 grid
)

if st.sidebar.button("Refresh Live Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("System: Azure Cosmos DB | Feed: Finnhub")

# 4. Main Dashboard Area
st.title("Live Market Monitoring")
st.markdown("---")

# Query limits reduced slightly to ensure fast loading when pulling multiple stocks
@st.cache_data(ttl=60)
def load_data(symbol):
    query = f"SELECT * FROM c WHERE c.ticker = '{symbol}' ORDER BY c.timestamp DESC OFFSET 0 LIMIT 200"
    return list(container.query_items(query=query, enable_cross_partition_query=True))

if not selected_tickers:
    st.warning("System Standby: Please select at least one asset from the control panel.")
else:
    # 5. Build the Dynamic Grid (2 columns wide)
    cols = st.columns(2)
    
    for index, ticker in enumerate(selected_tickers):
        # Alternate between the left and right column
        col = cols[index % 2]
        
        with col:
            st.subheader(ticker)
            items = load_data(ticker)
            
            if items:
                df = pd.DataFrame(items)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                latest = df.iloc[-1]
                delta_val = None
                if len(df) > 1:
                    prev = df.iloc[-2]
                    delta_val = latest['price'] - prev['price']
                    
                # Clean, compact metrics above each chart
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"${latest['price']:.2f}", f"{delta_val:.2f}" if delta_val else None)
                m2.metric("High", f"${latest['high']:.2f}")
                m3.metric("Low", f"${latest['low']:.2f}")
                
                # Compact Candlestick Chart
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['price'],
                    increasing_line_color='#26a69a', # Institutional green
                    decreasing_line_color='#ef5350'  # Institutional red
                )])
                
                fig.update_layout(
                    template="plotly_dark",
                    height=350, # Shorter height to fit multiple panels on screen
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_rangeslider_visible=False,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---") # Visual separator for the next row
            else:
                st.info(f"Awaiting telemetry for {ticker}...")
