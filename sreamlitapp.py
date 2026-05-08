import streamlit as st
from azure.cosmos import CosmosClient
import pandas as pd
import plotly.graph_objects as go

# 1. Setup the Page
st.set_page_config(page_title="My Stock Dashboard", layout="wide")
st.title("Live Market Tracker")

# 2. Connect to Cosmos DB securely
@st.cache_resource
def init_connection():
    # Streamlit stores your secrets securely in their cloud settings
    return CosmosClient.from_connection_string(st.secrets["COSMOS_CONNECTION_STRING"])

client = init_connection()
database = client.get_database_client("FinancialData")
container = database.get_container_client("StockTicks")

# 3. Query the Data
# Let's say we want to look at Apple (AAPL)
ticker = st.selectbox("Select a Stock", ["AAPL", "MSFT", "NVDA", "TSLA"])

query = f"SELECT * FROM c WHERE c.ticker = '{ticker}' ORDER BY c.timestamp DESC"
items = list(container.query_items(query=query, enable_cross_partition_query=True))

if items:
    # 4. Turn the data into a Pandas DataFrame and draw a chart
    df = pd.DataFrame(items)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create a professional Plotly line chart
    fig = go.Figure(data=go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=ticker))
    fig.update_layout(title=f"{ticker} Live Price", template="plotly_dark")
    
    # Display it on the webpage
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data found for this ticker yet. Waiting for Azure Function...")
