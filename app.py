import streamlit as st  # <-- FIX: import the real Streamlit package, not this file
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
# import plotly.express as px   # (unused; comment out to keep imports tidy)
# from plotly.subplots import make_subplots  # (unused)
import gdown
import os

# Set page configuration (keep it near the top)
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stMetric:hover { transform: translateY(-2px); transition: all 0.3s ease; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .metric-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: all 0.3s ease; }
    .welcome-header { background: linear-gradient(120deg, #1f77b4, #2ecc71); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 30px; }
    .metric-container { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .metric-label { color: #2c3e50; font-weight: bold; font-size: 1em; margin-bottom: 5px; }
    .metric-value { font-size: 1.2em; font-weight: bold; }
    .positive-change { color: #2ecc71; }
    .negative-change { color: #e74c3c; }
    [data-testid="metric-container"] { background-color: #f8f9fa; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    [data-testid="metric-container"] > div { color: black !important; }
    [data-testid="metric-container"] label { color: #2c3e50 !important; }
    [data-testid="stMetricValue"] { color: black !important; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# User credentials storage (demo only)
user_credentials = {'admin': '123', 'user1': 'user123'}

def load_data_from_gdrive(file_id: str) -> pd.DataFrame:
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'temp.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    try:
        os.remove(output)
    except Exception:
        pass
    return df

def login():
    st.markdown("""
        <div class="welcome-header">
            <h1>📊 Sales Forecasting Platform</h1>
            <p>Please login to access the dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username in user_credentials and user_credentials[username] == password:
                    st.session_state.logged_in = True
                    st.success("Login Successful!")
                    st.rerun()  # immediately rerun into the dashboard
                else:
                    st.error("Invalid Username or Password")

def process_data(past_sales: pd.DataFrame, forecast: pd.DataFrame):
    try:
        # Basic schema validation
        required_cols = {"Date", "Overall Sales"}
        missing_past = required_cols - set(past_sales.columns)
        missing_fore = required_cols - set(forecast.columns)
        if missing_past or missing_fore:
            st.error(
                f"Missing required columns.\n"
                f"Past file missing: {sorted(missing_past)}\n"
                f"Forecast file missing: {sorted(missing_fore)}"
            )
            return None, None

        # Parse and sort dates
        past_sales['Date'] = pd.to_datetime(past_sales['Date'])
        forecast['Date'] = pd.to_datetime(forecast['Date'])
        past_sales = past_sales.sort_values('Date')
        forecast = forecast.sort_values('Date')
        return past_sales, forecast
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

def create_welcome_section():
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%B %d, %Y")
    current_time = current_datetime.strftime("%I:%M %p")
    st.markdown(f"""
        <div class="welcome-header">
            <h1>Welcome to Sales Forecasting Dashboard</h1>
            <p>📅 {current_date} | ⏰ {current_time}</p>
        </div>
    """, unsafe_allow_html=True)

def create_overall_visualization(past_sales, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past_sales['Date'],
        y=past_sales['Overall Sales'],
        name='Historical Sales',
        line=dict(color='#2E86C1', width=2),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['Date'],
        y=forecast['Overall Sales'],
        name='Forecasted Sales',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        mode='lines'
    ))
    fig.update_layout(
        title={'text': 'Overall Sales: Historical vs Forecast', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

def create_product_visualization(past_sales, forecast, selected_product):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=past_sales['Date'],
        y=past_sales[selected_product],
        name=f'Historical {selected_product} Sales',
        line=dict(color='#27AE60', width=2),
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['Date'],
        y=forecast[selected_product],
        name=f'Forecasted {selected_product} Sales',
        line=dict(color='#8E44AD', width=2, dash='dash'),
        mode='lines'
    ))
    fig.update_layout(
        title={'text': f'{selected_product} Analysis: Historical vs Forecast', 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    return fig

def main_dashboard():
    create_welcome_section()
    st.markdown("### 📈 Dashboard Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa;">
                <h3 style="color: #1f77b4; text-align: center;">📊 Sales Analysis</h3>
                <p style="text-align: center; color: #2c3e50;">Track historical and forecasted sales trends</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa;">
                <h3 style="color: #2ca02c; text-align: center;">📈 Product Metrics</h3>
                <p style="text-align: center; color: #2c3e50;">Analyze individual product performance</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card" style="background-color: #f8f9fa;">
                <h3 style="color: #ff7f0e; text-align: center;">📋 Performance Stats</h3>
                <p style="text-align: center; color: #2c3e50;">View detailed performance metrics</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    sales_file_id = "1WkuFBBlEOa6tS9rhlJnhggedGIBlH0Ms"
    forecast_file_id = "1PcVHRQPGktMld_hl2wVqrU17Nv3xroZz"

    try:
        with st.spinner('📊 Loading sales data...'):
            past_sales = load_data_from_gdrive(sales_file_id)
            forecast = load_data_from_gdrive(forecast_file_id)
            past_sales, forecast = process_data(past_sales, forecast)

        if past_sales is not None and forecast is not None:
            st.markdown("### 📊 Quick Summary")
            quick_stats_col1, quick_stats_col2, quick_stats_col3 = st.columns(3)

            with quick_stats_col1:
                total_sales = past_sales['Overall Sales'].sum()
                st.metric("Total Historical Sales", f"${total_sales:,.2f}")

            with quick_stats_col2:
                avg_sales = past_sales['Overall Sales'].mean()
                st.metric("Average Daily Sales", f"${avg_sales:,.2f}")

            with quick_stats_col3:
                forecast_avg = forecast['Overall Sales'].mean()
                if avg_sales and avg_sales != 0:
                    percent_change = ((forecast_avg - avg_sales) / avg_sales) * 100
                    delta_text = f"{percent_change:,.1f}% vs hist avg"
                else:
                    percent_change = 0.0
                    delta_text = "N/A (hist avg = 0)"
                # Show the forecast average as the value; show change as delta
                st.metric("Forecast Trend", f"${forecast_avg:,.2f}", delta=delta_text)

            st.markdown("### 📈 Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""<div class="metric-card"><h4 style="color: #1f77b4;">Historical Data</h4>""", unsafe_allow_html=True)
                st.write(f"📅 Date Range: {past_sales['Date'].min().strftime('%Y-%m-%d')} - {past_sales['Date'].max().strftime('%Y-%m-%d')}")
                st.write(f"💰 Average Overall Sales: ${past_sales['Overall Sales'].mean():,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("""<div class="metric-card"><h4 style="color: #2ca02c;">Forecast Data</h4>""", unsafe_allow_html=True)
                st.write(f"📅 Forecast Period: {forecast['Date'].min().strftime('%Y-%m-%d')} - {forecast['Date'].max().strftime('%Y-%m-%d')}")
                st.write(f"💰 Average Overall Sales: ${forecast['Overall Sales'].mean():,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📊 Overall Sales Analysis")
            overall_fig = create_overall_visualization(past_sales, forecast)
            st.plotly_chart(overall_fig, use_container_width=True)

            st.markdown("### 📈 Product-wise Analysis")
            product_columns = [col for col in past_sales.columns if col.startswith('Product_')]
            if not product_columns:
                st.info("No product columns found (expected columns starting with 'Product_').")
                return
            selected_product = st.selectbox("Select Product for Detailed Analysis", product_columns)
            product_fig = create_product_visualization(past_sales, forecast, selected_product)
            st.plotly_chart(product_fig, use_container_width=True)

            st.markdown("### 📊 Product Performance Metrics")
            for product in product_columns:
                with st.container():
                    st.markdown(f"""
                        <div class="metric-card" style="background-color: #f8f9fa;">
                            <h4 style="color: #1f77b4; margin-bottom: 20px; font-size: 1.5em;">{product}</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    hist_avg = past_sales[product].mean()
                    fore_avg = forecast[product].mean()
                    change = ((fore_avg - hist_avg) / hist_avg) * 100 if hist_avg != 0 else np.nan
                    max_sales = max(past_sales[product].max(), forecast[product].max())
                    min_sales = min(past_sales[product].min(), forecast[product].min())

                    with col1:
                        st.markdown("""<div class="metric-container"><div class="metric-label">Historical Average</div><div class="metric-value" style="color: #1f77b4;">""", unsafe_allow_html=True)
                        st.write(f"${hist_avg:,.2f}")
                        st.markdown("</div></div>", unsafe_allow_html=True)

                        st.markdown("""<div class="metric-container"><div class="metric-label">Forecasted Average</div><div class="metric-value" style="color: #2ecc71;">""", unsafe_allow_html=True)
                        st.write(f"${fore_avg:,.2f}")
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("""<div class="metric-container"><div class="metric-label">Expected Change</div>""", unsafe_allow_html=True)
                        if np.isnan(change):
                            st.markdown(f'<div class="metric-value">N/A</div>', unsafe_allow_html=True)
                        elif change >= 0:
                            st.markdown(f'<div class="metric-value positive-change">+{change:.1f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="metric-value negative-change">{change:.1f}%</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        st.markdown("""<div class="metric-container"><div class="metric-label">Highest Sales</div><div class="metric-value" style="color: #3498db;">""", unsafe_allow_html=True)
                        st.write(f"${max_sales:,.2f}")
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    with col3:
                        st.markdown("""<div class="metric-container"><div class="metric-label">Lowest Sales</div><div class="metric-value" style="color: #e67e22;">""", unsafe_allow_html=True)
                        st.write(f"${min_sales:,.2f}")
                        st.markdown("</div></div>", unsafe_allow_html=True)

                    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Exception details:", str(e))

def main():
    if not st.session_state.logged_in:
        login()
    else:
        main_dashboard()
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()
