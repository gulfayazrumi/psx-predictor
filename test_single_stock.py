"""
Test prediction on a single stock
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostDirectionPredictor
from src.models.ensemble import EnsemblePredictor


def test_stock(symbol='SYS'):
    """Test prediction on a single stock"""
    
    print(f"\n{'='*70}")
    print(f"TESTING PREDICTION FOR {symbol}")
    print(f"{'='*70}\n")
    
    # Check if models exist
    lstm_path = Path(f"models/saved/lstm_{symbol.lower()}.h5")
    xgb_path = Path(f"models/saved/xgboost_{symbol.lower()}.json")
    
    if not lstm_path.exists():
        print(f"❌ LSTM model not found for {symbol}")
        print(f"   Run: python train_all.py --stocks {symbol}")
        return
    
    if not xgb_path.exists():
        print(f"❌ XGBoost model not found for {symbol}")
        print(f"   Run: python train_all.py --stocks {symbol}")
        return
    
    print(f"✓ Found models for {symbol}")
    
    # Load data
    csv_path = Path(f"data/raw/historical/{symbol.lower()}.csv")
    
    if not csv_path.exists():
        print(f"❌ Data file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date')
    
    print(f"✓ Loaded {len(df)} rows of data")
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Clean data
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✓ Created {len(df_features.columns)} features")
    
    # Load models
    try:
        lstm = LSTMPredictor(lookback=60)
        lstm.load_model(f"models/saved/lstm_{symbol.lower()}")
        print("✓ Loaded LSTM model")
        
        xgb = XGBoostDirectionPredictor()
        xgb.load_model(f"models/saved/xgboost_{symbol.lower()}")
        print("✓ Loaded XGBoost model")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Make prediction
    try:
        ensemble = EnsemblePredictor()
        ensemble.set_models(lstm, xgb)
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 
                                     'target_direction', 'close', 'open', 
                                     'high', 'low', 'volume']]
        
        prediction = ensemble.predict_next_day(df_features, feature_cols)
        
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*70}")
        print(f"Symbol:           {symbol}")
        print(f"Current Price:    PKR {prediction['current_price']:.2f}")
        print(f"Predicted Price:  PKR {prediction['predicted_price']:.2f}")
        print(f"Expected Change:  {prediction['predicted_change_pct']:+.2f}%")
        print(f"Direction:        {prediction['direction']}")
        print(f"Confidence:       {prediction['confidence']:.1%}")
        print(f"Recommendation:   {prediction['recommendation']}")
        print(f"{'='*70}\n")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    
    symbol = 'SYS'
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    
    test_stock(symbol)