import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px



DB_NAME = "ecommerce.db"
st.set_page_config(page_title="Autonomous COO Dashboard", page_icon="📈", layout="wide")

# @st.cache_data ensures we don't query the database every single time the user clicks a button
@st.cache_data
def load_data():
    """Loads historical and forecast data from the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    
    try:
        hist_df = pd.read_sql_query("SELECT * FROM historical_sales", conn)
        hist_df['order_date'] = pd.to_datetime(hist_df['order_date'])
        daily_hist = hist_df.groupby('order_date')['sales'].sum().reset_index()
    except Exception as e:
        st.error(f"Could not load historical data. Did you run the ETL pipeline? Error: {e}")
        daily_hist = pd.DataFrame()

    try:
        forecast_df = pd.read_sql_query("SELECT * FROM future_predictions", conn)
        forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
    except Exception as e:
        st.warning(f"Could not load forecast data. Did you run the ML pipeline? Error: {e}")
        forecast_df = pd.DataFrame()
        
    conn.close()
    return hist_df, daily_hist, forecast_df


st.title("📈 Autonomous COO Dashboard")
st.markdown("Welcome to the executive overview. Here is the current health of the business and our 30-day AI forecast.")


raw_hist, daily_hist, forecast_df = load_data()

if not daily_hist.empty:
   
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3 = st.columns(3)
    
    total_sales = raw_hist['sales'].sum()
    total_orders = len(raw_hist)
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    
    with col1:
        st.metric("Total Historical Sales", f"${total_sales:,.2f}")
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
    with col3:
        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")

    st.divider()

  
    st.subheader("Historical Sales Trend")
  
    fig_hist = px.line(
        daily_hist, 
        x='order_date', 
        y='sales', 
        title="Daily Sales Over Time",
        labels={'order_date': 'Date', 'sales': 'Total Sales ($)'},
        template="plotly_white"
    )
   
    fig_hist.update_traces(line=dict(color="blue", width=2))
    st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("No historical data available to display.")

if not forecast_df.empty:
    st.divider()
    st.subheader("AI Forecast: Next 30 Days")
    
    projected_total = forecast_df['predicted_sales'].sum()
    st.metric("Projected 30-Day Revenue", f"${projected_total:,.2f}", delta="AI Projection")
    
    fig_forecast = px.line(
        forecast_df, 
        x='forecast_date', 
        y='predicted_sales', 
        title="Machine Learning Projected Sales",
        labels={'forecast_date': 'Date', 'predicted_sales': 'Predicted Sales ($)'},
        template="plotly_white"
    )
    fig_forecast.update_traces(line=dict(color="green", width=3, dash='dot'))
    st.plotly_chart(fig_forecast, use_container_width=True)


