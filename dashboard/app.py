import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="PSX AI Trading System",
    page_icon="📊",
    layout="wide"
)

def load_signals():
    """Load trading signals"""
    signals_path = Path("reports/trading_signals.csv")
    if signals_path.exists():
        return pd.read_csv(signals_path)
    return None

def load_stock_data(symbol):
    """Load historical data for a stock"""
    csv_path = Path(f"data/raw/historical/{symbol}.csv")
    
    if not csv_path.exists():
        return None
    
    try:
        # Read CSV
        raw_df = pd.read_csv(csv_path)
        
        # Remove completely empty columns
        raw_df = raw_df.dropna(axis=1, how='all')
        
        # Keep only first occurrence of duplicate column names
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated(keep='first')]
        
        # Standardize column names
        raw_df.columns = raw_df.columns.str.strip().str.upper()
        
        # Create new clean dataframe
        clean_df = pd.DataFrame()
        
        # Handle date column
        if 'TIME' in raw_df.columns:
            clean_df['DATE'] = pd.to_datetime(raw_df['TIME'], format='%d-%b-%y', errors='coerce')
        elif 'DATE' in raw_df.columns:
            clean_df['DATE'] = pd.to_datetime(raw_df['DATE'], errors='coerce')
        else:
            return None
        
        # Copy OHLCV columns
        for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
            if col in raw_df.columns:
                clean_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            else:
                return None
        
        # Remove invalid rows
        clean_df = clean_df.dropna(subset=['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE'])
        
        if len(clean_df) == 0:
            return None
        
        # Sort by date
        clean_df = clean_df.sort_values('DATE').reset_index(drop=True)
        
        return clean_df
        
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None

def create_candlestick_chart(df, symbol):
    """Create candlestick chart"""
    if df is None or len(df) == 0:
        return None
    
    try:
        # Create figure
        fig = go.Figure(data=[go.Candlestick(
            x=df['DATE'].tolist(),
            open=df['OPEN'].tolist(),
            high=df['HIGH'].tolist(),
            low=df['LOW'].tolist(),
            close=df['CLOSE'].tolist(),
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
    except Exception as e:
        print(f"Chart error for {symbol}: {e}")
        return None

def get_recommendation(percent_change, confidence, direction):
    """Get trading recommendation"""
    if direction == 'UP' and percent_change > 2 and confidence > 0.6:
        return "🟢 Recommendation: STRONG BUY"
    elif direction == 'UP' and percent_change > 0.5:
        return "🟢 Recommendation: BUY"
    elif direction == 'DOWN' and percent_change < -2 and confidence > 0.6:
        return "🔴 Recommendation: STRONG SELL"
    elif direction == 'DOWN' and percent_change < -0.5:
        return "🔴 Recommendation: SELL"
    else:
        return "🟡 Recommendation: HOLD"

def calculate_stats(df):
    """Calculate key statistics"""
    if df is None or len(df) == 0:
        return None
    
    df_30d = df.tail(30)
    
    if len(df_30d) < 2:
        return None
    
    stats = {
        'return_30d': ((df_30d['CLOSE'].iloc[-1] / df_30d['CLOSE'].iloc[0]) - 1) * 100,
        'avg_volume': df_30d['VOLUME'].mean(),
        'volatility': df_30d['CLOSE'].std(),
        'latest_volume': df_30d['VOLUME'].iloc[-1]
    }
    
    return stats

def main():
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Select Page", ["📊 Overview", "🎯 Trading Signals", "🔍 Stock Details"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### PSX AI Trading System")
    st.sidebar.markdown("LSTM + XGBoost Models")
    st.sidebar.markdown("Live price updates from Sarmaaya API")
    
    # Load signals
    signals_df = load_signals()
    
    if signals_df is None:
        st.error("⚠️ No trading signals found. Run: python update_live_signals.py")
        return
    
    # Header
    st.title("📊 PSX AI Trading System")
    
    # Market status
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    is_market_open = market_open <= now <= market_close and now.weekday() < 5
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if is_market_open:
            st.success("🟢 Market OPEN")
        else:
            st.error("🔴 Market CLOSED")
    
    with col2:
        st.metric("🕐 Time", now.strftime("%H:%M:%S"))
    
    with col3:
        if 'last_update' in signals_df.columns and len(signals_df) > 0:
            last_update = pd.to_datetime(signals_df['last_update'].iloc[0])
            time_diff = now - last_update
            if time_diff.seconds < 3600:
                st.success(f"✓ Data updated {time_diff.seconds // 60} minutes ago")
            else:
                st.warning(f"⚠️ Data is {time_diff.seconds // 3600} hours old. Run: python update_live_signals.py")
    
    # Page content
    if page == "📊 Overview":
        st.header("📈 Top Predictions")
        
        # Show top gainers
        top_gainers = signals_df.nlargest(10, 'percent_change')
        
        display_df = top_gainers[['symbol', 'current_price', 'predicted_price', 'percent_change', 'direction', 'confidence']].copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"PKR {x:.2f}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"PKR {x:.2f}")
        display_df['percent_change'] = display_df['percent_change'].apply(lambda x: f"{x:+.2f}%")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    elif page == "🎯 Trading Signals":
        st.header("💡 Trading Signals")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_filter = st.selectbox("Direction", ["All", "UP", "DOWN"])
        
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5, 0.05)
        
        with col3:
            min_change = st.slider("Min Change %", 0.0, 10.0, 0.0, 0.5)
        
        # Apply filters
        filtered_df = signals_df.copy()
        
        if direction_filter != "All":
            filtered_df = filtered_df[filtered_df['direction'] == direction_filter]
        
        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
        filtered_df = filtered_df[abs(filtered_df['percent_change']) >= min_change]
        
        st.write(f"Found {len(filtered_df)} signals")
        
        display_df = filtered_df[['symbol', 'current_price', 'predicted_price', 'percent_change', 'direction', 'confidence']].copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"PKR {x:.2f}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"PKR {x:.2f}")
        display_df['percent_change'] = display_df['percent_change'].apply(lambda x: f"{x:+.2f}%")
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    else:  # Stock Details
        st.header("🔍 Detailed Stock Analysis")
        
        # Stock selector
        stock_symbols = sorted(signals_df['symbol'].tolist())
        selected_stock = st.selectbox("Select Stock Symbol", stock_symbols)
        
        if selected_stock:
            # Get stock data
            stock_signal = signals_df[signals_df['symbol'] == selected_stock].iloc[0]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"PKR {stock_signal['current_price']:.2f}")
            
            with col2:
                st.metric("Predicted Price", f"PKR {stock_signal['predicted_price']:.2f}",
                         f"{stock_signal['percent_change']:.2f}%")
            
            with col3:
                st.metric("Direction", stock_signal['direction'])
            
            with col4:
                st.metric("Confidence", f"{stock_signal['confidence']*100:.1f}%")
            
            # Recommendation
            rec = get_recommendation(stock_signal['percent_change'], 
                                   stock_signal['confidence'], 
                                   stock_signal['direction'])
            st.info(rec)
            
            # Load historical data
            historical_df = load_stock_data(selected_stock)
            
            if historical_df is not None and len(historical_df) > 0:
                # Chart
                st.subheader("📈 Price Chart (Last 90 Days)")
                chart_df = historical_df.tail(90)
                fig = create_candlestick_chart(chart_df, selected_stock)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📊 Key Statistics")
                stats = calculate_stats(historical_df)
                
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("30-Day Return", f"{stats['return_30d']:+.2f}%")
                    
                    with col2:
                        st.metric("Avg Volume (30D)", f"{stats['avg_volume']:,.0f}")
                    
                    with col3:
                        st.metric("Volatility (30D)", f"{stats['volatility']:.2f}")
                    
                    with col4:
                        st.metric("Latest Volume", f"{stats['latest_volume']:,.0f}")
            else:
                st.warning(f"No historical data available for {selected_stock}")
    
    # Footer
    st.markdown("---")
    st.caption(f"PSX AI Trading System • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()