"""
Simple training script that works with raw CSV files
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, create_directories, save_dataframe
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMPredictor, create_train_test_split
from src.models.xgboost_model import XGBoostDirectionPredictor, prepare_classification_data
from src.models.ensemble import EnsemblePredictor

def train_stock(symbol='HBL'):
    """Train models on a stock using raw CSV data"""
    
    print(f"\n{'='*70}")
    print(f"PSX STOCK PREDICTOR - TRAINING: {symbol}")
    print(f"{'='*70}")
    print(f"Start Time: {datetime.now()}\n")
    
    # Step 1: Load raw data
    print("STEP 1: Loading Data")
    print("-" * 70)
    
    csv_path = f"data/raw/historical/{symbol.lower()}.csv"
    
    if not Path(csv_path).exists():
        print(f"✗ File not found: {csv_path}")
        print(f"\nAvailable stocks:")
        historical_path = Path("data/raw/historical")
        if historical_path.exists():
            stocks = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
            for i, stock in enumerate(stocks[:20], 1):
                print(f"  {i}. {stock}")
            print(f"  ... and {len(stocks)-20} more")
        return False
    
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Clean column names
    df.columns = df.columns.str.lower()
    
    # Parse date
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    elif 'date' not in df.columns:
        print("✗ No date column found")
        return False
    
    df.sort_values('date', inplace=True)
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Engineer features
    print("\nSTEP 2: Feature Engineering")
    print("-" * 70)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    if len(df_features) < 100:
        print(f"✗ Insufficient data after feature engineering: {len(df_features)} rows")
        return False
    
    # Save features
    save_dataframe(df_features, f"{symbol.lower()}_features.csv", data_type='features')
    
    # Step 3: Train LSTM
    print(f"\nSTEP 3: Training LSTM Model")
    print("-" * 70)
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]
    
    print(f"Using {len(feature_cols)} features")
    
    try:
        data_split = create_train_test_split(
            df_features,
            feature_cols,
            target_col='close',
            test_size=0.2,
            lookback=60
        )
        
        lstm = LSTMPredictor(lookback=60)
        lstm.scaler = data_split['scaler']
        lstm.feature_scaler = data_split['feature_scaler']
        
        history = lstm.train(
            data_split['X_train'],
            data_split['y_train'],
            data_split['X_test'],
            data_split['y_test'],
            model_save_path=f"models/saved/lstm_{symbol.lower()}.h5"
        )
        
        # Evaluate
        predictions = lstm.predict(data_split['X_test'])
        predictions_actual = data_split['scaler'].inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        actuals = data_split['scaler'].inverse_transform(
            data_split['y_test'].reshape(-1, 1)
        ).flatten()
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(actuals, predictions_actual)
        mae = mean_absolute_error(actuals, predictions_actual)
        r2 = r2_score(actuals, predictions_actual)
        
        print(f"\nLSTM Results:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Save
        lstm.save_model(f"models/saved/lstm_{symbol.lower()}")
        print(f"✓ LSTM model saved")
        
    except Exception as e:
        print(f"✗ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        lstm = None
    
    # Step 4: Train XGBoost
    print(f"\nSTEP 4: Training XGBoost Model")
    print("-" * 70)
    
    feature_cols_xgb = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 
                                     'target_direction', 'close', 'open', 'high', 'low', 'volume']]
    
    try:
        data_split_xgb = prepare_classification_data(
            df_features,
            feature_cols_xgb,
            target_col='target_direction',
            test_size=0.2
        )
        
        xgb = XGBoostDirectionPredictor()
        results = xgb.train(
            data_split_xgb['X_train'],
            data_split_xgb['y_train'],
            data_split_xgb['X_test'],
            data_split_xgb['y_test']
        )
        
        eval_results = xgb.evaluate(
            data_split_xgb['X_test'],
            data_split_xgb['y_test'],
            feature_names=feature_cols_xgb[:10]  # Show top 10 features
        )
        
        xgb.save_model(f"models/saved/xgboost_{symbol.lower()}")
        print(f"✓ XGBoost model saved")
        
    except Exception as e:
        print(f"✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        xgb = None
    
    # Step 5: Create ensemble and test prediction
    if lstm and xgb:
        print(f"\nSTEP 5: Creating Ensemble & Testing")
        print("-" * 70)
        
        ensemble = EnsemblePredictor()
        ensemble.set_models(lstm, xgb)
        
        try:
            prediction = ensemble.predict_next_day(df_features, feature_cols)
            
            print(f"\n{'='*70}")
            print("PREDICTION TEST")
            print(f"{'='*70}")
            print(f"Symbol:           {symbol}")
            print(f"Current Price:    PKR {prediction['current_price']:.2f}")
            print(f"Predicted Price:  PKR {prediction['predicted_price']:.2f}")
            print(f"Change:           {prediction['predicted_change_pct']:+.2f}%")
            print(f"Direction:        {prediction['direction']}")
            print(f"Confidence:       {prediction['confidence']:.1%}")
            print(f"Signal Strength:  {prediction['signal_strength']}")
            print(f"Recommendation:   {prediction['recommendation']}")
            print(f"Models Agree:     {'Yes' if prediction['models_agree'] else 'No'}")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"✓ Training completed for {symbol}")
    print(f"End Time: {datetime.now()}")
    print(f"{'='*70}\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PSX Stock Predictor (Simple)')
    parser.add_argument('--symbol', type=str, default='HBL', help='Stock symbol (default: HBL)')
    parser.add_argument('--list', action='store_true', help='List available stocks')
    
    args = parser.parse_args()
    
    create_directories()
    
    if args.list:
        print("\nAvailable stocks in data/raw/historical/:")
        print("=" * 70)
        historical_path = Path("data/raw/historical")
        if historical_path.exists():
            stocks = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
            for i, stock in enumerate(stocks, 1):
                print(f"  {i:3d}. {stock}")
            print(f"\nTotal: {len(stocks)} stocks")
        return
    
    success = train_stock(args.symbol.upper())
    
    if success:
        print("\n🎉 Success! Next steps:")
        print("  1. Run dashboard: streamlit run dashboard/app.py")
        print("  2. Try another stock: python train_simple.py --symbol OGDC")
        print("  3. Train multiple: python train_simple.py --symbol MCB")


if __name__ == "__main__":
    main()