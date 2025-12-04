import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

def render():
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
        st.subheader("Top Dividend Stocks")
        if 'dividend' in analysis_data:
            dividend_df = analysis_data['dividend'].head(20)
            
            # Chart
            fig = px.bar(
                dividend_df.head(10),
                x='symbol',
                y='dividend_yield',
                title="Top 10 Dividend Yields",
                color='dividend_quality',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(
                dividend_df[['symbol', 'close', 'dividend_yield', 'dividend_quality']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Run `python src/analysis/valuation_analyzer.py` to generate dividend analysis")

if __name__ == "__main__":
    render()
