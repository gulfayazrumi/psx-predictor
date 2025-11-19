"""
Feature Engineering Module
Creates technical indicators and features for stock prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("Warning: 'ta' library not installed. Installing fallback indicators.")


class FeatureEngineer:
    """Create features for stock prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.config = config
        else:
            from src.utils import load_config
            self.config = load_config()
        
        self.technical_indicators = self.config['features']['technical_indicators']
        self.lookback_periods = self.config['features']['lookback_periods']
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df.columns = df.columns.str.lower()
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain: {required_cols}")
        
        if TA_AVAILABLE:
            df = self._add_ta_indicators(df)
        else:
            df = self._add_manual_indicators(df)
        
        return df
    
    def _add_ta_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using ta library"""
        # Moving Averages
        df['SMA_5'] = SMAIndicator(close=df['close'], window=5).sma_indicator()
        df['SMA_10'] = SMAIndicator(close=df['close'], window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['EMA_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # RSI
        df['RSI_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        
        # ATR (Average True Range)
        df['ATR_14'] = AverageTrueRange(high=df['high'], low=df['low'], 
                                        close=df['close'], window=14).average_true_range()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ADX_14'] = adx.adx()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], 
                                      close=df['close'], window=14, smooth_window=3)
        df['Stochastic_K'] = stoch.stoch()
        df['Stochastic_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = WilliamsRIndicator(high=df['high'], low=df['low'], 
                                               close=df['close'], lbp=14).williams_r()
        
        # OBV (On Balance Volume)
        df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        return df
    
    def _add_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: Add indicators manually"""
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        df['RSI_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + (std_20 * 2)
        df['BB_middle'] = sma_20
        df['BB_lower'] = sma_20 - (std_20 * 2)
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df, 14)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R manually"""
        high_roll = df['high'].rolling(window=period).max()
        low_roll = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_roll - df['close']) / (high_roll - low_roll))
        return williams_r
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # High-Low spread
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_pct'] = (df['hl_spread'] / df['close']) * 100
        
        # Open-Close spread
        df['oc_spread'] = df['close'] - df['open']
        df['oc_spread_pct'] = (df['oc_spread'] / df['open']) * 100
        
        # Intraday range
        df['intraday_range'] = (df['high'] - df['low']) / df['low']
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        df = df.copy()
        
        # Volume moving averages
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Price-Volume correlation
        df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        df = df.copy()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   df['close'].shift(period)) * 100
        
        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features"""
        df = df.copy()
        
        for lag in lags:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistical features"""
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            
            # Rolling std
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            
            # Rolling min/max
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Percentage from high/low
            df[f'pct_from_high_{window}'] = (df['close'] - df[f'close_max_{window}']) / df[f'close_max_{window}'] * 100
            df[f'pct_from_low_{window}'] = (df['close'] - df[f'close_min_{window}']) / df[f'close_min_{window}'] * 100
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features at once"""
        print("Creating features...")
        
        df = df.copy()
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
            df.sort_values('date', inplace=True)
        
        # Add all features
        df = self.add_technical_indicators(df)
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.add_momentum_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        
        # Create target variable (next day close)
        df['target_next_close'] = df['close'].shift(-1)
        df['target_direction'] = (df['target_next_close'] > df['close']).astype(int)
        
        # Drop NaN rows
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        
        print(f"✓ Features created: {len(df.columns)} columns")
        print(f"✓ Rows: {initial_rows} → {final_rows} (dropped {initial_rows - final_rows} NaN rows)")
        
        return df


def main():
    """Test feature engineering"""
    print("=" * 50)
    print("FEATURE ENGINEERING TEST")
    print("=" * 50)
    
    # Load sample data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='B')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'date': dates,
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(10000, 100000, len(dates))
    })
    
    print(f"\nOriginal data shape: {df.shape}")
    print(df.head())
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\nFinal data shape: {df_features.shape}")
    print(f"\nFeatures created ({len(df_features.columns)}):")
    for i, col in enumerate(df_features.columns, 1):
        print(f"{i:3d}. {col}")
    
    print("\n" + "=" * 50)
    print("Feature engineering test completed!")


if __name__ == "__main__":
    main()
