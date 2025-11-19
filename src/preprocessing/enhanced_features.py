"""
Enhanced feature engineering with fundamentals and sentiment
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


class EnhancedFeatureEngineer:
    """Enhanced features with fundamental data and market sentiment"""
    
    def __init__(self):
        self.api = SarmayaAPI()
    
    def add_fundamental_features(self, df, symbol):
        """Add fundamental analysis features"""
        
        # Get stock info
        info = self.api.get_stock_info(symbol)
        
        if not info:
            return df
        
        # Add fundamental ratios (constant for recent period)
        df['market_cap'] = info.get('marketCap', np.nan)
        df['pe_ratio'] = info.get('peRatio', np.nan)
        df['eps'] = info.get('eps', np.nan)
        df['dividend_yield'] = info.get('dividendYield', np.nan)
        df['book_value'] = info.get('bookValue', np.nan)
        
        # Price to book ratio
        if 'book_value' in df.columns and df['book_value'].notna().any():
            df['price_to_book'] = df['close'] / df['book_value']
        
        # Forward fill fundamental data
        fundamental_cols = ['market_cap', 'pe_ratio', 'eps', 'dividend_yield', 'book_value']
        for col in fundamental_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def add_announcement_sentiment(self, df, symbol):
        """Add sentiment from corporate announcements"""
        
        # Get announcements for the period
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        
        announcements = self.api.get_announcements(symbol, start_date, end_date)
        
        if not announcements:
            df['announcement_count'] = 0
            df['days_since_announcement'] = 999
            return df
        
        # Create announcement dataframe
        ann_df = pd.DataFrame(announcements)
        ann_df['date'] = pd.to_datetime(ann_df['announcementDate'])
        
        # Count announcements per day
        ann_counts = ann_df.groupby('date').size().reset_index(name='announcement_count')
        
        # Merge with main df
        df = df.merge(ann_counts, on='date', how='left')
        df['announcement_count'] = df['announcement_count'].fillna(0)
        
        # Days since last announcement
        df['days_since_announcement'] = 0
        last_announcement = None
        
        for idx in df.index:
            if df.loc[idx, 'announcement_count'] > 0:
                last_announcement = df.loc[idx, 'date']
                df.loc[idx, 'days_since_announcement'] = 0
            elif last_announcement:
                days_diff = (df.loc[idx, 'date'] - last_announcement).days
                df.loc[idx, 'days_since_announcement'] = days_diff
            else:
                df.loc[idx, 'days_since_announcement'] = 999
        
        # Announcement momentum (rolling count)
        df['announcement_momentum_7d'] = df['announcement_count'].rolling(7).sum()
        df['announcement_momentum_30d'] = df['announcement_count'].rolling(30).sum()
        
        return df
    
    def add_market_sentiment(self, df):
        """Add overall market sentiment features"""
        
        market_view = self.api.get_market_view()
        
        if not market_view:
            return df
        
        # Add market indices
        df['kse100_value'] = market_view.get('kse100', {}).get('value', np.nan)
        df['kse100_change'] = market_view.get('kse100', {}).get('change', np.nan)
        df['kse100_change_pct'] = market_view.get('kse100', {}).get('changePct', np.nan)
        
        # Market breadth
        df['market_gainers'] = market_view.get('gainers', 0)
        df['market_losers'] = market_view.get('losers', 0)
        df['market_unchanged'] = market_view.get('unchanged', 0)
        
        # Market sentiment ratio
        total_stocks = df['market_gainers'] + df['market_losers'] + df['market_unchanged']
        if total_stocks > 0:
            df['market_sentiment'] = (df['market_gainers'] - df['market_losers']) / total_stocks
        else:
            df['market_sentiment'] = 0
        
        # Forward fill market data
        market_cols = ['kse100_value', 'kse100_change', 'kse100_change_pct', 
                       'market_gainers', 'market_losers', 'market_sentiment']
        for col in market_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        return df
    
    def add_relative_strength(self, df, symbol):
        """Add relative strength vs KSE-100"""
        
        # Get KSE-100 history (using a proxy like OGDC or direct index data)
        kse_df = self.api.get_price_history('OGDC', days=len(df))
        
        if kse_df is None or len(kse_df) == 0:
            return df
        
        kse_df = kse_df.rename(columns={'close': 'kse_close'})
        kse_df = kse_df[['date', 'kse_close']]
        
        # Merge
        df = df.merge(kse_df, on='date', how='left')
        df['kse_close'] = df['kse_close'].fillna(method='ffill')
        
        # Calculate relative strength
        df['relative_strength'] = (df['close'] / df['close'].iloc[0]) / (df['kse_close'] / df['kse_close'].iloc[0])
        
        # Relative strength momentum
        df['rs_momentum_5d'] = df['relative_strength'].pct_change(5)
        df['rs_momentum_20d'] = df['relative_strength'].pct_change(20)
        
        return df
    
    def create_all_enhanced_features(self, df, symbol):
        """Create all enhanced features"""
        
        print(f"\nCreating enhanced features for {symbol}...")
        
        # Add fundamental features
        print("  Adding fundamental data...")
        df = self.add_fundamental_features(df, symbol)
        
        # Add announcement sentiment
        print("  Adding announcement sentiment...")
        df = self.add_announcement_sentiment(df, symbol)
        
        # Add market sentiment
        print("  Adding market sentiment...")
        df = self.add_market_sentiment(df)
        
        # Add relative strength
        print("  Adding relative strength...")
        df = self.add_relative_strength(df, symbol)
        
        print(f"✓ Enhanced features created: {len(df.columns)} total columns")
        
        return df