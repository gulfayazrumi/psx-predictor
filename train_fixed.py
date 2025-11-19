"""
Fixed training script with data cleaning
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


def clean_data(df):
    """Clean data by removing infinite and extreme values"""
    print("\nCleaning data...")
    
    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check for issues
    nan_counts = df.isnull().sum()
    problematic_cols = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
    
    if problematic_cols:
        print(f"  Dropping {len(problematic_cols)} columns with >50% NaN")
        df = df.drop(columns=problematic_cols)
    
    # Fill remaining NaN with forward fill, then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Clip extreme values (beyond 99.9 percentile)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['date', 'time']]
    
    for col in numeric_cols:
        if df[col].std() > 0:  # Only clip if there's variance
            lower = df[col].quantile(0.001)
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(lower, upper)
    
    # Final check
    remaining_nans = df.isnull().sum().sum()
    remaining_infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    
    print(f"  ✓ Remaining NaN: {remaining_nans}")
    print(f"  ✓ Remaining Inf: {remaining_infs}")
    print(f"  ✓ Final shape: {df.shape}")
    
    return df


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
        return False
    
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Clean column names
    df.columns = df.columns.str.lower()
    
    # Parse date
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    
    df.sort_values('date', inplace=True)
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Engineer features
    print("\nSTEP 2: Feature Engineering")
    print("-" * 70)
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    # Step 3: Clean data
    df_features = clean_data(df_features)
    
    if len(df_features) < 100:
        print(f"✗ Insufficient data after cleaning: {len(df_features)} rows")
        return False
    
    # Save features
    save_dataframe(df_features, f"{symbol.lower()}_features.csv", data_type='features')
    
    # Step 4: Train LSTM
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
        
        print("\nTraining LSTM (this may take a few minutes)...")
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
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions_actual) / (actuals + 1e-8))) * 100
        
        print(f"\n{'='*50}")
        print("LSTM Evaluation Results")
        print(f"{'='*50}")
        print(f"  MSE:   {mse:10.4f}")
        print(f"  RMSE:  {rmse:10.4f}")
        print(f"  MAE:   {mae:10.4f}")
        print(f"  R²:    {r2:10.4f}")
        print(f"  MAPE:  {mape:10.2f}%")
        print(f"{'='*50}")
        
        # Save
        lstm.save_model(f"models/saved/lstm_{symbol.lower()}")
        print(f"✓ LSTM model saved")
        
    except Exception as e:
        print(f"✗ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()
        lstm = None
    
    # Step 5: Train XGBoost
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
        
        print("\nTraining XGBoost...")
        results = xgb.train(
            data_split_xgb['X_train'],
            data_split_xgb['y_train'],
            data_split_xgb['X_test'],
            data_split_xgb['y_test']
        )
        
        print(f"\n{'='*50}")
        print("XGBoost Evaluation")
        print(f"{'='*50}")
        
        eval_results = xgb.evaluate(
            data_split_xgb['X_test'],
            data_split_xgb['y_test'],
            feature_names=feature_cols_xgb[:10]
        )
        
        xgb.save_model(f"models/saved/xgboost_{symbol.lower()}")
        
    except Exception as e:
        print(f"✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        xgb = None
    
    # Step 6: Create ensemble and test prediction
    if lstm and xgb:
        print(f"\nSTEP 5: Creating Ensemble & Making Prediction")
        print("-" * 70)
        
        ensemble = EnsemblePredictor()
        ensemble.set_models(lstm, xgb)
        
        try:
            prediction = ensemble.predict_next_day(df_features, feature_cols)
            
            print(f"\n{'='*70}")
            print("🎯 NEXT-DAY PREDICTION")
            print(f"{'='*70}")
            print(f"Symbol:              {symbol}")
            print(f"Current Price:       PKR {prediction['current_price']:,.2f}")
            print(f"Predicted Price:     PKR {prediction['predicted_price']:,.2f}")
            print(f"Expected Change:     {prediction['predicted_change_pct']:+.2f}%")
            print(f"Direction:           {prediction['direction']}")
            print(f"Confidence:          {prediction['confidence']:.1%}")
            print(f"Signal Strength:     {prediction['signal_strength']}")
            print(f"Recommendation:      {prediction['recommendation']}")
            print(f"Models Agree:        {'✓ Yes' if prediction['models_agree'] else '✗ No'}")
            print(f"{'='*70}")
            
            # Trading signal
            if prediction['recommendation'] == 'BUY' and prediction['confidence'] > 0.65:
                print(f"\n🟢 STRONG BUY SIGNAL")
            elif prediction['recommendation'] == 'SELL' and prediction['confidence'] > 0.65:
                print(f"\n🔴 STRONG SELL SIGNAL")
            elif prediction['recommendation'] == 'HOLD':
                print(f"\n🟡 HOLD - Wait for better opportunity")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed successfully for {symbol}!")
    print(f"End Time: {datetime.now()}")
    print(f"{'='*70}\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PSX Stock Predictor (Fixed)')
    parser.add_argument('--symbol', type=str, default='HBL', help='Stock symbol')
    parser.add_argument('--list', action='store_true', help='List available stocks')
    
    args = parser.parse_args()
    
    create_directories()
    
    if args.list:
        print("\nAvailable stocks:")
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
        print("\n🎉 Next Steps:")
        print("  1. Launch dashboard:  streamlit run dashboard/app.py")
        print("  2. Start API server:  python src/api/api_server.py")
        print(f"  3. Train more stocks: python train_fixed.py --symbol OGDC")
        print(f"  4. See all stocks:    python train_fixed.py --list")


if __name__ == "__main__":
    main()