import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostDirectionPredictor
from src.models.ensemble import EnsemblePredictor
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils import load_config

def predict_single_stock(symbol, models_dir):
    """Generate prediction for a single stock"""
    try:
        # Load data
        csv_path = Path(f"data/raw/historical/{symbol}.csv")
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Parse date
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date')
        
        # Need at least 60 days for lookback
        if len(df) < 60:
            return None
            
        # Feature Engineering
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Check if models exist
        lstm_path = models_dir / f"lstm_{symbol.lower()}.h5"
        xgb_path = models_dir / f"xgboost_{symbol.lower()}.json"
        
        if not lstm_path.exists() or not xgb_path.exists():
            print(f"Missing models for {symbol}: LSTM={lstm_path.exists()}, XGB={xgb_path.exists()}")
            return None
            
        # Load models
        lstm = LSTMPredictor(lookback=60, model_type='v12')
        if not lstm.load_model(lstm_path.parent / f"lstm_{symbol.lower()}"):
            return None
            
        xgb = XGBoostDirectionPredictor()
        xgb.load_model(str(models_dir / f"xgboost_{symbol.lower()}"))
        
        # Ensemble
        ensemble = EnsemblePredictor()
        ensemble.set_models(lstm, xgb)
        
        # Prepare features for prediction
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 'target_direction']]
                       
        feature_cols_xgb = [col for col in df_features.columns 
                           if col not in ['date', 'time', 'target_next_close', 
                                         'target_direction', 'close', 'open', 'high', 'low', 'volume']]
        
        # Predict
        prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)
        
        # Add metadata
        prediction['symbol'] = symbol
        prediction['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return prediction
        
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("🔮 GENERATING v12 PREDICTIONS")
    print("="*70)
    
    models_dir = Path("models/v12")
    if not models_dir.exists():
        print("❌ No v12 models found. Please run training first.")
        return
        
    # Get list of stocks with models
    model_files = list(models_dir.glob("lstm_*.h5"))
    stocks = [f.stem.replace("lstm_", "").upper() for f in model_files]
    
    print(f"Found models for {len(stocks)} stocks")
    
    predictions = []
    
    for i, symbol in enumerate(stocks, 1):
        print(f"[{i}/{len(stocks)}] Predicting {symbol}...", end=' ', flush=True)
        
        pred = predict_single_stock(symbol, models_dir)
        
        if pred:
            predictions.append(pred)
            print("✓")
        else:
            print("✗")
            
    if predictions:
        # Save to CSV
        df = pd.DataFrame(predictions)
        
        # Ensure columns order
        cols = ['symbol', 'current_price', 'predicted_price', 'predicted_change_pct', 
                'direction', 'confidence', 'recommendation', 'last_update']
        
        # Rename predicted_change_pct to percent_change to match dashboard expectation
        df = df.rename(columns={'predicted_change_pct': 'percent_change'})
        
        # Save
        output_path = Path("reports/trading_signals.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(predictions)} predictions to {output_path}")
    else:
        print("\n⚠️ No predictions generated.")

if __name__ == "__main__":
    main()
