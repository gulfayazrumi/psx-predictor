"""
Main Training Script for PSX Stock Predictor
Orchestrates data collection, feature engineering, model training, and evaluation
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from src.utils import (load_config, setup_logging, create_directories, 
                      save_dataframe, load_dataframe, calculate_metrics)
from src.data_collection.sarmaaya_api import SarmayaAPIClient
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMPredictor, create_train_test_split
from src.models.xgboost_model import XGBoostDirectionPredictor, prepare_classification_data
from src.models.ensemble import EnsemblePredictor, backtest_ensemble


def step1_collect_data(args):
    """Step 1: Collect historical data"""
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    
    api_client = SarmayaAPIClient()
    
    if args.symbol:
        # Single stock
        print(f"\nFetching data for {args.symbol}...")
        df = api_client.get_price_history(args.symbol, days=args.days)
        
        if df is not None:
            filename = f"{args.symbol.lower()}_raw.csv"
            save_dataframe(df, filename, data_type='raw')
            print(f"✓ Saved {len(df)} records")
            return {args.symbol: df}
        else:
            print(f"✗ Failed to fetch data for {args.symbol}")
            return {}
    
    else:
        # Multiple stocks or KSE-100
        if args.kse100:
            print("\nFetching KSE-100 stocks...")
            return api_client.get_all_kse100_data(days=args.days)
        else:
            print("\nPlease specify --symbol or --kse100")
            return {}


def step2_engineer_features(data_dict, args):
    """Step 2: Engineer features"""
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    engineer = FeatureEngineer()
    processed_data = {}
    
    for symbol, df in data_dict.items():
        print(f"\nProcessing {symbol}...")
        
        try:
            # Create features
            df_features = engineer.create_all_features(df)
            
            # Save
            filename = f"{symbol.lower()}_features.csv"
            save_dataframe(df_features, filename, data_type='features')
            
            processed_data[symbol] = df_features
            
        except Exception as e:
            print(f"✗ Failed to process {symbol}: {e}")
    
    return processed_data


def step3_train_lstm(df, symbol, args):
    """Step 3: Train LSTM model"""
    print("\n" + "="*70)
    print(f"STEP 3: TRAINING LSTM MODEL FOR {symbol}")
    print("="*70)
    
    # Prepare data
    feature_cols = [col for col in df.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Create train/test split
    data_split = create_train_test_split(
        df, 
        feature_cols, 
        target_col='close',
        test_size=0.2,
        lookback=60
    )
    
    # Initialize and train model
    lstm = LSTMPredictor(lookback=60)
    lstm.scaler = data_split['scaler']
    lstm.feature_scaler = data_split['feature_scaler']
    
    # Train
    history = lstm.train(
        data_split['X_train'], 
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test'],
        model_save_path=f"models/saved/lstm_{symbol.lower()}.h5"
    )
    
    # Evaluate
    print("\n--- LSTM Evaluation ---")
    predictions = lstm.predict(data_split['X_test'])
    
    # Denormalize predictions
    predictions_actual = data_split['scaler'].inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()
    
    actuals = data_split['scaler'].inverse_transform(
        data_split['y_test'].reshape(-1, 1)
    ).flatten()
    
    metrics = calculate_metrics(actuals, predictions_actual)
    
    for metric, value in metrics.items():
        print(f"{metric:8s}: {value:10.4f}")
    
    # Save model
    lstm.save_model(f"models/saved/lstm_{symbol.lower()}")
    
    return lstm, metrics


def step4_train_xgboost(df, symbol, args):
    """Step 4: Train XGBoost model"""
    print("\n" + "="*70)
    print(f"STEP 4: TRAINING XGBOOST MODEL FOR {symbol}")
    print("="*70)
    
    # Prepare data
    feature_cols = [col for col in df.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction', 
                                 'close', 'open', 'high', 'low', 'volume']]
    
    data_split = prepare_classification_data(
        df,
        feature_cols,
        target_col='target_direction',
        test_size=0.2
    )
    
    # Initialize and train model
    xgb = XGBoostDirectionPredictor()
    
    results = xgb.train(
        data_split['X_train'],
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test']
    )
    
    # Detailed evaluation
    eval_results = xgb.evaluate(
        data_split['X_test'],
        data_split['y_test'],
        feature_names=feature_cols
    )
    
    # Save model
    xgb.save_model(f"models/saved/xgboost_{symbol.lower()}")
    
    return xgb, eval_results


def step5_create_ensemble(lstm, xgb, df, symbol, args):
    """Step 5: Create and test ensemble"""
    print("\n" + "="*70)
    print(f"STEP 5: CREATING ENSEMBLE MODEL FOR {symbol}")
    print("="*70)
    
    ensemble = EnsemblePredictor()
    ensemble.set_models(lstm, xgb)
    
    # Test prediction
    feature_cols = [col for col in df.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]
    
    print("\n--- Making Sample Prediction ---")
    prediction = ensemble.predict_next_day(df, feature_cols)
    
    print(f"\nCurrent Price: PKR {prediction['current_price']:.2f}")
    print(f"Predicted Price: PKR {prediction['predicted_price']:.2f}")
    print(f"Change: {prediction['predicted_change_pct']:+.2f}%")
    print(f"Direction: {prediction['direction']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Signal Strength: {prediction['signal_strength']}")
    print(f"Recommendation: {prediction['recommendation']}")
    print(f"Models Agree: {prediction['models_agree']}")
    
    return ensemble


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train PSX Stock Predictor')
    
    parser.add_argument('--symbol', type=str, help='Stock symbol (e.g., HBL)')
    parser.add_argument('--kse100', action='store_true', help='Train on all KSE-100 stocks')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data (default: 730)')
    parser.add_argument('--skip-data', action='store_true', help='Skip data collection')
    parser.add_argument('--load-symbol', type=str, help='Load existing data for symbol')
    
    args = parser.parse_args()
    
    # Setup
    create_directories()
    logger = setup_logging('INFO', 'logs/training.log')
    
    print("\n" + "="*70)
    print("PSX STOCK PREDICTOR - TRAINING PIPELINE")
    print("="*70)
    print(f"Start Time: {datetime.now()}")
    
    # Load or collect data
    if args.skip_data and args.load_symbol:
        print(f"\nLoading existing data for {args.load_symbol}...")
        df = load_dataframe(f"{args.load_symbol.lower()}_features.csv", data_type='features')
        data_dict = {args.load_symbol: df}
        processed_data = data_dict
    else:
        # Step 1: Collect data
        data_dict = step1_collect_data(args)
        
        if not data_dict:
            print("\n✗ No data collected. Exiting.")
            return
        
        # Step 2: Engineer features
        processed_data = step2_engineer_features(data_dict, args)
    
    if not processed_data:
        print("\n✗ No processed data. Exiting.")
        return
    
    # Train models for each stock
    for symbol, df in processed_data.items():
        if len(df) < 100:
            print(f"\n✗ Insufficient data for {symbol} ({len(df)} rows). Skipping.")
            continue
        
        print(f"\n\n{'='*70}")
        print(f"TRAINING MODELS FOR {symbol}")
        print(f"{'='*70}")
        
        try:
            # Step 3: Train LSTM
            lstm, lstm_metrics = step3_train_lstm(df, symbol, args)
            
            # Step 4: Train XGBoost
            xgb, xgb_metrics = step4_train_xgboost(df, symbol, args)
            
            # Step 5: Create ensemble
            ensemble = step5_create_ensemble(lstm, xgb, df, symbol, args)
            
            print(f"\n✓ Successfully trained models for {symbol}")
        
        except Exception as e:
            print(f"\n✗ Failed to train models for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print(f"End Time: {datetime.now()}")
    print("="*70)
    
    print("\n📊 Next Steps:")
    print("1. Start API: python src/api/api_server.py")
    print("2. Start Dashboard: streamlit run dashboard/app.py")
    print("3. Test predictions via API or dashboard")


if __name__ == "__main__":
    main()
