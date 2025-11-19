"""
PSX Stock Predictor Dashboard - Complete Working Version
100% Fixed - Shows Live Prices
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PSX AI Trading",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #rgb(14, 17, 23);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stMetric {
        background-color: #rgb(14, 17, 23);
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_predictions():
    """Load predictions from updated signals file"""
    
    # Primary source: Updated signals with live prices
    signals_file = Path("reports/trading_signals.csv")
    
    if signals_file.exists():
        try:
            df = pd.read_csv(signals_file)
            
            # Ensure numeric columns
            numeric_cols = ['current_price', 'predicted_price', 'price_change', 
                          'percent_change', 'confidence']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename percent_change to change_pct if needed
            if 'percent_change' in df.columns and 'change_pct' not in df.columns:
                df = df.rename(columns={'percent_change': 'change_pct'})
            
            # Add recommendation if missing
            if 'recommendation' not in df.columns:
                df['recommendation'] = 'HOLD'
            
            # Remove rows with critical NaN values
            df = df.dropna(subset=['symbol', 'current_price', 'predicted_price'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading signals: {e}")
            return pd.DataFrame()
    
    # Fallback: Try signals_enhanced.csv
    enhanced_file = Path("reports/signals_enhanced.csv")
    if enhanced_file.exists():
        try:
            df = pd.read_csv(enhanced_file)
            
            # Ensure numeric
            numeric_cols = ['current_price', 'predicted_price', 'change_pct', 'confidence']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['symbol', 'current_price'])
            return df
            
        except Exception as e:
            pass
    
    return pd.DataFrame()


def load_stock_data(symbol):
    """Load historical data for a stock"""
    csv_path = Path(f"data/raw/historical/{symbol}.csv")
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        # Handle date column
        if 'TIME' in df.columns:
            df['DATE'] = pd.to_datetime(df['TIME'], format='%d-%b-%y', errors='coerce')
        elif 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        
        # Remove rows with NaN dates
        df = df.dropna(subset=['DATE'])
        
        # Sort by date
        df = df.sort_values('DATE', ascending=True)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {symbol}: {e}")
        return None


def create_candlestick_chart(df, symbol):
    """Create candlestick chart"""
    
    if df is None or len(df) == 0:
        return None
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['DATE'],
        open=df['OPEN'],
        high=df['HIGH'],
        low=df['LOW'],
        close=df['CLOSE'],
        name=symbol
    )])
    
    fig.update_layout(
        title=f"{symbol} - Price History (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price (PKR)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def format_price(value):
    """Safely format price"""
    try:
        return f"PKR {float(value):.2f}"
    except:
        return "N/A"


def format_percentage(value):
    """Safely format percentage"""
    try:
        return f"{float(value):+.2f}%"
    except:
        return "N/A"


def format_confidence(value):
    """Safely format confidence"""
    try:
        return f"{float(value):.1%}"
    except:
        return "N/A"


def main():
    """Main dashboard function"""
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<p class="main-header">📊 PSX AI Trading System</p>', unsafe_allow_html=True)
    
    with col2:
        now = datetime.now()
        market_open = (now.hour >= 9 and now.hour < 15 and now.weekday() < 5)
        if market_open:
            st.success("🟢 Market OPEN")
        else:
            st.info("🔴 Market CLOSED")
    
    with col3:
        st.metric("🕐 Time", now.strftime('%H:%M:%S'))
    
    # Check data freshness
    signals_file = Path("reports/trading_signals.csv")
    if signals_file.exists():
        mod_time = datetime.fromtimestamp(signals_file.stat().st_mtime)
        minutes_ago = int((datetime.now() - mod_time).total_seconds() / 60)
        
        if minutes_ago < 60:
            st.caption(f"✓ Data updated {minutes_ago} minutes ago")
        else:
            hours_ago = minutes_ago // 60
            st.warning(f"⚠️ Data is {hours_ago} hours old. Run: python update_live_signals.py")
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🎯 Trading Signals", "🔍 Stock Details"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **PSX AI Trading System**
    
    LSTM + XGBoost Models
    
    Live price updates from Sarmaaya API
    """)
    
    # Load data
    with st.spinner("Loading predictions..."):
        predictions_df = load_predictions()
    
    if len(predictions_df) == 0:
        st.error("❌ No predictions found!")
        st.info("""
        **To generate predictions:**
        1. Update live prices: `python update_live_signals.py`
        2. Or train models: `python train_all.py --max 50`
        """)
        return
    
    # PAGE: OVERVIEW
    if page == "📊 Overview":
        st.markdown("## 🤖 Model Predictions Overview")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Total Predictions", len(predictions_df))
        
        with col2:
            buy_count = len(predictions_df[predictions_df.get('recommendation', '') == 'BUY'])
            st.metric("🟢 BUY Signals", buy_count)
        
        with col3:
            sell_count = len(predictions_df[predictions_df.get('recommendation', '') == 'SELL'])
            st.metric("🔴 SELL Signals", sell_count)
        
        with col4:
            avg_conf = predictions_df['confidence'].mean() * 100
            st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
        
        st.markdown("---")
        
        # Predictions table
        st.markdown("### 📋 All Predictions")
        
        # Create display dataframe
        display_df = predictions_df.copy()
        
        # Format columns safely
        display_df['Current Price'] = display_df['current_price'].apply(format_price)
        display_df['Predicted Price'] = display_df['predicted_price'].apply(format_price)
        display_df['Change %'] = display_df['change_pct'].apply(format_percentage)
        display_df['Confidence'] = display_df['confidence'].apply(format_confidence)
        display_df['Symbol'] = display_df['symbol']
        display_df['Direction'] = display_df.get('direction', 'N/A')
        display_df['Action'] = display_df.get('recommendation', 'HOLD')
        
        # Display table
        st.dataframe(
            display_df[['Symbol', 'Current Price', 'Predicted Price', 
                       'Change %', 'Direction', 'Confidence', 'Action']],
            height=600,
            use_container_width=True
        )
        
        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            "📥 Download All Predictions (CSV)",
            csv,
            "psx_predictions.csv",
            "text/csv",
            key='download-csv'
        )
    
    # PAGE: TRADING SIGNALS
    elif page == "🎯 Trading Signals":
        st.markdown("## 🎯 Trading Signals & Opportunities")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_change = st.slider(
                "Minimum Change %", 
                -20.0, 20.0, 2.0, 0.5,
                help="Filter stocks by minimum expected price change"
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence", 
                0.0, 1.0, 0.3, 0.05,
                format="%.0f%%",
                help="Filter by model confidence level"
            )
        
        st.markdown("---")
        
        # Filter data
        filtered_df = predictions_df[
            (predictions_df['change_pct'].abs() >= abs(min_change)) &
            (predictions_df['confidence'] >= min_confidence)
        ].copy()
        
        # BUY Signals
        st.markdown("### 🟢 BUY Opportunities")
        buy_signals = filtered_df[filtered_df['change_pct'] > 0].sort_values(
            'change_pct', ascending=False
        ).head(20)
        
        if len(buy_signals) > 0:
            buy_display = buy_signals.copy()
            buy_display['Current'] = buy_display['current_price'].apply(format_price)
            buy_display['Target'] = buy_display['predicted_price'].apply(format_price)
            buy_display['Upside'] = buy_display['change_pct'].apply(format_percentage)
            buy_display['Confidence'] = buy_display['confidence'].apply(format_confidence)
            
            st.dataframe(
                buy_display[['symbol', 'Current', 'Target', 'Upside', 'Confidence']],
                use_container_width=True
            )
        else:
            st.info("No BUY signals match current filters")
        
        st.markdown("---")
        
        # SELL Signals
        st.markdown("### 🔴 SELL Warnings")
        sell_signals = filtered_df[filtered_df['change_pct'] < 0].sort_values(
            'change_pct'
        ).head(20)
        
        if len(sell_signals) > 0:
            sell_display = sell_signals.copy()
            sell_display['Current'] = sell_display['current_price'].apply(format_price)
            sell_display['Target'] = sell_display['predicted_price'].apply(format_price)
            sell_display['Downside'] = sell_display['change_pct'].apply(format_percentage)
            sell_display['Confidence'] = sell_display['confidence'].apply(format_confidence)
            
            st.dataframe(
                sell_display[['symbol', 'Current', 'Target', 'Downside', 'Confidence']],
                use_container_width=True
            )
        else:
            st.info("No SELL signals match current filters")
    
    # PAGE: STOCK DETAILS
    elif page == "🔍 Stock Details":
        st.markdown("## 🔍 Detailed Stock Analysis")
        
        # Stock selector
        available_stocks = sorted(predictions_df['symbol'].unique())
        selected_stock = st.selectbox(
            "Select Stock Symbol",
            available_stocks,
            help="Choose a stock to view detailed analysis"
        )
        
        if selected_stock:
            # Get prediction
            stock_data = predictions_df[predictions_df['symbol'] == selected_stock].iloc[0]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    format_price(stock_data['current_price'])
                )
            
            with col2:
                change_val = stock_data['change_pct']
                st.metric(
                    "Predicted Price",
                    format_price(stock_data['predicted_price']),
                    format_percentage(change_val),
                    delta_color="normal"
                )
            
            with col3:
                direction = stock_data.get('direction', 'NEUTRAL')
                st.metric("Direction", direction)
            
            with col4:
                conf_val = stock_data['confidence']
                st.metric("Confidence", format_confidence(conf_val))
            
            # Recommendation box
            rec = stock_data.get('recommendation', 'HOLD')
            if rec == 'BUY':
                st.success(f"🟢 **Recommendation: BUY**")
            elif rec == 'SELL':
                st.error(f"🔴 **Recommendation: SELL**")
            else:
                st.warning(f"🟡 **Recommendation: HOLD**")
            
            st.markdown("---")
            
            # Load historical data
            historical_df = load_stock_data(selected_stock)
            
            if historical_df is not None and len(historical_df) > 0:
                st.markdown("### 📈 Price Chart (Last 90 Days)")
                
                # Create chart
                chart_df = historical_df.tail(90)
                fig = create_candlestick_chart(chart_df, selected_stock)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.markdown("### 📊 Key Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if len(historical_df) >= 30:
                        return_30d = (
                            (historical_df['CLOSE'].iloc[-1] - historical_df['CLOSE'].iloc[-30]) 
                            / historical_df['CLOSE'].iloc[-30] * 100
                        )
                        st.metric("30-Day Return", f"{return_30d:+.2f}%")
                
                with col2:
                    avg_vol = historical_df['VOLUME'].tail(30).mean()
                    st.metric("Avg Volume (30D)", f"{avg_vol:,.0f}")
                
                with col3:
                    volatility = historical_df['CLOSE'].tail(30).std()
                    st.metric("Volatility (30D)", f"{volatility:.2f}")
                
                with col4:
                    latest_vol = historical_df['VOLUME'].iloc[-1]
                    st.metric("Latest Volume", f"{latest_vol:,.0f}")
            else:
                st.warning(f"No historical data available for {selected_stock}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"<small>PSX AI Trading System • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()