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
        background-color: rgb(0 0 0 / 75%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8eaf6;
        border-radius: 5px;
        padding: 10px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #c5cae9;
        color: #1a237e;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1a237e;
    }
</style>
""", unsafe_allow_html=True)

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path to ensure imports work
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

@st.cache_data
def load_data():
    """Load all necessary data"""
    data = {}
    
    # Load signals
    signals_path = PROJECT_ROOT / "reports" / "trading_signals.csv"
    if signals_path.exists():
        data['signals'] = pd.read_csv(signals_path)
    
    # Load accuracy report
    accuracy_path = PROJECT_ROOT / "reports" / "accuracy_report.csv"
    if accuracy_path.exists():
        data['accuracy'] = pd.read_csv(accuracy_path)
        
    # Load weekly forecasts
    weekly_path = PROJECT_ROOT / "reports" / "weekly_forecasts.csv"
    if weekly_path.exists():
        data['weekly'] = pd.read_csv(weekly_path)
        
    return data

def load_stock_history(symbol):
    """Load historical data for a specific stock"""
    csv_path = PROJECT_ROOT / "data" / "raw" / "historical" / f"{symbol}.csv"
    if not csv_path.exists():
        return None
        
    try:
        df = pd.read_csv(csv_path)
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Remove duplicate columns again after lowercasing
        df = df.loc[:, ~df.columns.duplicated()]
        
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
    
    # Auto-update system
    if 'data_initialized' not in st.session_state:
        with st.sidebar.status("🔄 Checking for updates...", expanded=True) as status:
            try:
                # Try importing as package first
                from dashboard.auto_update import run_essential_updates
                run_essential_updates()
                st.session_state.data_initialized = True
                status.update(label="✅ System Ready", state="complete", expanded=False)
            except ImportError:
                try:
                    # Fallback: try importing directly if in dashboard dir
                    import auto_update
                    auto_update.run_essential_updates()
                    st.session_state.data_initialized = True
                    status.update(label="✅ System Ready", state="complete", expanded=False)
                except Exception as e:
                    st.warning(f"⚠️ Auto-update skipped: {e}")
                    st.session_state.data_initialized = True
                    status.update(label="⚠️ Update Skipped", state="error", expanded=False)
            except Exception as e:
                st.warning(f"⚠️ Auto-update skipped: {e}")
                st.session_state.data_initialized = True
                status.update(label="⚠️ Update Skipped", state="error", expanded=False)

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "📈 Market Analysis", "🔮 Weekly Forecasts", "🎯 Model Performance", "🎯 Model Accuracy", "📊 Advanced Analysis", "📅 Seasonality"]
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
            (signals_df['confidence'] > 0.6) & 
            (abs(signals_df['percent_change']) > 0.5)
        ].sort_values('confidence', ascending=False).head(10)
        
        if len(top_picks) == 0:
             top_picks = signals_df.sort_values('confidence', ascending=False).head(10)

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
            
            # Check if we have actual accuracy data
            has_accuracy_data = (acc_df['direction_accuracy'] > 0).any()
            
            if not has_accuracy_data:
                st.warning("""
                ⏳ **Accuracy Tracking in Progress**
                
                The system needs to accumulate prediction history to calculate accuracy metrics.
                
                **How it works:**
                1. Daily predictions are saved to `reports/history/`
                2. After 1-2 days, the system compares predictions with actual prices
                3. Accuracy metrics will appear here automatically
                
                **Current Status:** Collecting initial predictions...
                """)
            
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

    elif page == "🎯 Model Accuracy":
        import dashboard.accuracy_page as accuracy_page
        accuracy_page.render()

    elif page == "📊 Advanced Analysis":
        st.title("Advanced Market Analysis")
        
        # Load analysis data
        @st.cache_data(ttl=600)
        def load_analysis_data():
            analysis = {}
            
            # Sector analysis
            sector_path = Path("reports/sector_analysis.csv")
            if sector_path.exists():
                analysis['sectors'] = pd.read_csv(sector_path)
            
            # Momentum stocks
            momentum_path = Path("reports/momentum_stocks.csv")
            if momentum_path.exists():
                analysis['momentum'] = pd.read_csv(momentum_path)
            
            # Value stocks
            value_path = Path("reports/value_stocks.csv")
            if value_path.exists():
                analysis['value'] = pd.read_csv(value_path)
            
            # Dividend stocks
            dividend_path = Path("reports/dividend_stocks.csv")
            if dividend_path.exists():
                analysis['dividend'] = pd.read_csv(dividend_path)
            
            return analysis
        
        analysis_data = load_analysis_data()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏢 Sectors", "🚀 Momentum", "💎 Value", "💰 Dividends"])
        
        with tab1:
            st.subheader("Sector Performance")
            if 'sectors' in analysis_data:
                sectors_df = analysis_data['sectors']
                
                # Top/Bottom sectors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Top Performing Sectors**")
                    top_sectors = sectors_df.nlargest(5, 'change_pct')[['sector', 'change_pct']]
                    for _, row in top_sectors.iterrows():
                        st.metric(row['sector'][:30], f"{row['change_pct']:+.2f}%")
                
                with col2:
                    st.markdown("**🔴 Underperforming Sectors**")
                    bottom_sectors = sectors_df.nsmallest(5, 'change_pct')[['sector', 'change_pct']]
                    for _, row in bottom_sectors.iterrows():
                        st.metric(row['sector'][:30], f"{row['change_pct']:+.2f}%")
                
                # Full table
                st.dataframe(sectors_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run `python src/analysis/complete_analyzer.py` to generate sector analysis")
        
        with tab2:
            st.subheader("Momentum Leaders (52-Week Highs)")
            if 'momentum' in analysis_data:
                momentum_df = analysis_data['momentum'].head(20)
                
                # Chart
                fig = px.bar(
                    momentum_df.head(10),
                    x='symbol',
                    y='momentum_score',
                    title="Top 10 Momentum Stocks",
                    color='momentum_score',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    momentum_df[['symbol', 'close', 'change_pct', 'momentum_score']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Run `python src/analysis/valuation_analyzer.py` to generate momentum analysis")
        
        with tab3:
            st.subheader("Value Opportunities (52-Week Lows)")
            if 'value' in analysis_data:
                value_df = analysis_data['value'].head(20)
                
                # Chart
                fig = px.scatter(
                    value_df.head(15),
                    x='pe_ratio',
                    y='dividend_yield',
                    size='value_score',
                    color='value_score',
                    hover_data=['symbol'],
                    title="Value Stocks: P/E vs Dividend Yield",
                    labels={'pe_ratio': 'P/E Ratio', 'dividend_yield': 'Dividend Yield (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(
                    value_df[['symbol', 'close', 'pe_ratio', 'dividend_yield', 'value_score']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Run `python src/analysis/valuation_analyzer.py` to generate value analysis")
        
        with tab4:
            st.subheader("High Dividend Yields")
            if 'dividend' in analysis_data:
                dividend_df = analysis_data['dividend'].head(20)
                
                # Chart
                fig = px.bar(
                    dividend_df.head(10),
                    x='symbol',
                    y='dividend_yield',
                    title="Top 10 Dividend Stocks",
                    color='dividend_yield',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Table - dynamically select columns based on what's available
                display_cols = ['symbol', 'close', 'dividend_yield']
                if 'pe_ratio' in dividend_df.columns:
                    display_cols.append('pe_ratio')
                
                st.dataframe(
                    dividend_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Run `python src/analysis/valuation_analyzer.py` to generate dividend analysis")

    elif page == "📅 Seasonality":
        st.title("Seasonality Analysis")
        
        # Load seasonality data
        @st.cache_data(ttl=3600)
        def load_seasonality_data():
            data = {}
            
            # Summary patterns
            summary_path = Path("reports/seasonality_analysis.csv")
            if summary_path.exists():
                data['summary'] = pd.read_csv(summary_path)
            
            # Detailed stats
            stats_path = Path("reports/monthly_stats.json")
            if stats_path.exists():
                import json
                with open(stats_path, 'r') as f:
                    data['stats'] = json.load(f)
            
            return data
            
        seasonality_data = load_seasonality_data()
        
        tab1, tab2 = st.tabs(["📊 Seasonal Patterns", "🔍 Stock Search"])
        
        with tab1:
            st.subheader("Recurring Seasonal Patterns")
            
            if 'summary' in seasonality_data:
                df = seasonality_data['summary']
                
                # Show data info
                st.info(f"📊 Showing {len(df)} seasonal patterns from stocks with sufficient data (minimum 1 year required)")
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    pattern_type = st.selectbox("Pattern Type", ["All", "Bullish", "Bearish"])
                with col2:
                    min_win_rate = st.slider("Min Win Rate %", 0, 100, 70)
                
                # Apply filters
                filtered_df = df.copy()
                if pattern_type != "All":
                    filtered_df = filtered_df[filtered_df['pattern_type'] == pattern_type]
                
                if pattern_type == "Bearish":
                    filtered_df = filtered_df[filtered_df['win_rate'] <= (100 - min_win_rate)]
                else:
                    filtered_df = filtered_df[filtered_df['win_rate'] >= min_win_rate]
                
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "win_rate": st.column_config.ProgressColumn("Win Rate %", min_value=0, max_value=100, format="%.1f%%"),
                        "avg_return": st.column_config.NumberColumn("Avg Return %", format="%.2f%%")
                    }
                )
            else:
                st.info("No seasonality data found. Run `python src/analysis/seasonality_analyzer.py`")
                
        with tab2:
            st.subheader("Detailed Monthly Analysis")
            
            if 'stats' in seasonality_data:
                stats_dict = seasonality_data['stats']
                available_symbols = sorted(list(stats_dict.keys()))
                
                st.info(f"📊 {len(available_symbols)} stocks available for detailed analysis")
                
                symbol = st.selectbox("Select Stock", available_symbols)
                
                if symbol:
                    stock_stats = stats_dict[symbol]
                    stats_df = pd.DataFrame(stock_stats)
                    
                    # Heatmap-style Bar Chart
                    fig = px.bar(
                        stats_df,
                        x='month_name',
                        y='avg_return',
                        color='win_rate',
                        title=f"{symbol} - Average Monthly Returns & Win Rate",
                        labels={'avg_return': 'Average Return (%)', 'month_name': 'Month', 'win_rate': 'Win Rate (%)'},
                        color_continuous_scale='RdYlGn',
                        range_color=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Table
                    st.dataframe(
                        stats_df[['month_name', 'win_rate', 'avg_return', 'positive_years', 'negative_years', 'years_analyzed']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "win_rate": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f%%"),
                            "avg_return": st.column_config.NumberColumn("Avg Return", format="%.2f%%")
                        }
                    )
            else:
                st.info("No detailed stats available.")

if __name__ == "__main__":
    main()