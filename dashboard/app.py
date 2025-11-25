import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="PSX AI Trading System v12",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #0d47a1;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all necessary data"""
    data = {}
    
    # Load signals
    signals_path = Path("reports/trading_signals.csv")
    if signals_path.exists():
        data['signals'] = pd.read_csv(signals_path)
    
    # Load accuracy report
    accuracy_path = Path("reports/accuracy_report.csv")
    if accuracy_path.exists():
        data['accuracy'] = pd.read_csv(accuracy_path)
        
    # Load weekly forecasts
    weekly_path = Path("reports/weekly_forecasts.csv")
    if weekly_path.exists():
        data['weekly'] = pd.read_csv(weekly_path)
        
    return data

def load_stock_history(symbol):
    """Load historical data for a specific stock"""
    csv_path = Path(f"data/raw/historical/{symbol}.csv")
    if not csv_path.exists():
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle date
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        else:
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date')
        return df
    except:
        return None

def create_candlestick(df, symbol):
    """Create professional candlestick chart"""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    ))
    
    # Add Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['SMA_20'],
        line=dict(color='orange', width=1),
        name='SMA 20'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['SMA_50'],
        line=dict(color='blue', width=1),
        name='SMA 50'
    ))
    
    fig.update_layout(
        title=f"{symbol} Price History",
        yaxis_title="Price (PKR)",
        template="plotly_white",
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def create_forecast_chart(history_df, forecast_df, symbol):
    """Create chart showing history + forecast"""
    fig = go.Figure()
    
    # Recent history (last 30 days)
    recent = history_df.tail(30)
    
    fig.add_trace(go.Scatter(
        x=recent['date'],
        y=recent['close'],
        mode='lines+markers',
        name='Actual History',
        line=dict(color='blue')
    ))
    
    # Forecast
    stock_forecast = forecast_df[forecast_df['symbol'] == symbol].copy()
    stock_forecast['date'] = pd.to_datetime(stock_forecast['date'])
    stock_forecast = stock_forecast.sort_values('date')
    
    # Connect last history point to first forecast point
    last_hist = recent.iloc[-1]
    
    # Add connection line
    fig.add_trace(go.Scatter(
        x=[last_hist['date'], stock_forecast.iloc[0]['date']],
        y=[last_hist['close'], stock_forecast.iloc[0]['price']],
        mode='lines',
        line=dict(color='green', dash='dot'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=stock_forecast['date'],
        y=stock_forecast['price'],
        mode='lines+markers',
        name='v12 Forecast (5 Days)',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"{symbol} - 5 Day Forecast",
        yaxis_title="Price (PKR)",
        template="plotly_white",
        height=500
    )
    
    return fig

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/bullish.png", width=80)
    st.sidebar.title("PSX Predictor v12")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📈 Market Analysis", "🔮 Weekly Forecasts", "🎯 Model Performance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("System Status: 🟢 Online\n\nModel: v12 (LSTM+XGBoost)")
    
    # Load data
    data = load_data()
    
    if 'signals' not in data:
        st.error("No data found. Please run the integrated system first.")
        return

    signals_df = data['signals']
    
    # Main Content
    if page == "🏠 Dashboard":
        st.title("Market Overview")
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks Tracked", len(signals_df))
            
        with col2:
            bullish = len(signals_df[signals_df['direction'] == 'UP'])
            st.metric("Bullish Signals", bullish, delta=f"{bullish/len(signals_df)*100:.1f}%")
            
        with col3:
            bearish = len(signals_df[signals_df['direction'] == 'DOWN'])
            st.metric("Bearish Signals", bearish, delta=f"-{bearish/len(signals_df)*100:.1f}%", delta_color="inverse")
            
        with col4:
            avg_conf = signals_df['confidence'].mean() * 100
            st.metric("Avg Model Confidence", f"{avg_conf:.1f}%")
            
        # Top Opportunities
        st.subheader("🔥 Top Trading Opportunities")
        
        # Filter for high confidence
        top_picks = signals_df[
            (signals_df['confidence'] > 0.7) & 
            (abs(signals_df['percent_change']) > 1.0)
        ].sort_values('confidence', ascending=False).head(10)
        
        st.dataframe(
            top_picks[['symbol', 'current_price', 'predicted_price', 'percent_change', 'direction', 'confidence', 'recommendation']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "percent_change": st.column_config.NumberColumn("Change %", format="%.2f%%"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f")
            }
        )
        
    elif page == "📈 Market Analysis":
        st.title("Detailed Stock Analysis")
        
        symbol = st.selectbox("Select Stock", sorted(signals_df['symbol'].unique()))
        
        if symbol:
            stock_data = signals_df[signals_df['symbol'] == symbol].iloc[0]
            history = load_stock_history(symbol)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"PKR {stock_data['current_price']:.2f}")
            col2.metric("Predicted (Next Day)", f"PKR {stock_data['predicted_price']:.2f}", f"{stock_data['percent_change']:.2f}%")
            col3.metric("Signal", stock_data['direction'])
            col4.metric("Confidence", f"{stock_data['confidence']*100:.1f}%")
            
            # Chart
            if history is not None:
                fig = create_candlestick(history.tail(100), symbol)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Historical data not available.")
                
    elif page == "🔮 Weekly Forecasts":
        st.title("5-Day Price Forecasts")
        
        if 'weekly' not in data:
            st.warning("No weekly forecasts available yet. Run training to generate them.")
        else:
            weekly_df = data['weekly']
            symbol = st.selectbox("Select Stock for Forecast", sorted(weekly_df['symbol'].unique()))
            
            if symbol:
                history = load_stock_history(symbol)
                if history is not None:
                    fig = create_forecast_chart(history, weekly_df, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.subheader("Forecast Data")
                    forecast_data = weekly_df[weekly_df['symbol'] == symbol][['date', 'price']].copy()
                    forecast_data['price'] = forecast_data['price'].apply(lambda x: f"PKR {x:.2f}")
                    st.table(forecast_data)
                    
    elif page == "🎯 Model Performance":
        st.title("Model Accuracy Tracking")
        
        if 'accuracy' not in data:
            st.info("Accuracy data is being collected. Check back after a few days of predictions.")
        else:
            acc_df = data['accuracy']
            
            # Overall Stats
            col1, col2 = st.columns(2)
            
            with col1:
                overall_acc = acc_df['direction_accuracy'].mean()
                st.metric("Overall Direction Accuracy", f"{overall_acc:.1f}%")
                
            with col2:
                avg_error = acc_df['avg_error_pct'].mean()
                st.metric("Average Price Error", f"{avg_error:.2f}%")
                
            # Chart
            st.subheader("Accuracy by Stock")
            fig = px.bar(
                acc_df.sort_values('direction_accuracy', ascending=False).head(20),
                x='symbol',
                y='direction_accuracy',
                title="Top 20 Stocks by Prediction Accuracy",
                labels={'direction_accuracy': 'Direction Accuracy (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(acc_df, use_container_width=True)

if __name__ == "__main__":
    main()