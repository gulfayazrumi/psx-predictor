"""
Batch training script - Train models for all stocks
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, create_directories, save_dataframe
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMPredictor, create_train_test_split
from src.models.xgboost_model import XGBoostDirectionPredictor, prepare_classification_data
from src.models.ensemble import EnsemblePredictor


def clean_data(df):
    """Clean data by removing infinite and extreme values"""
    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Check for problematic columns
    nan_counts = df.isnull().sum()
    problematic_cols = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
    
    if problematic_cols:
        df = df.drop(columns=problematic_cols)
    
    # Fill remaining NaN
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Clip extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['date', 'time']]
    
    for col in numeric_cols:
        if df[col].std() > 0:
            lower = df[col].quantile(0.001)
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(lower, upper)
    
    return df


def train_single_stock(symbol, show_details=False):
    """Train models for a single stock"""
    
    try:
        # Load data
        csv_path = f"data/raw/historical/{symbol.lower()}.csv"
        
        if not Path(csv_path).exists():
            return {'symbol': symbol, 'status': 'SKIP', 'reason': 'File not found'}
        
        df = pd.read_csv(csv_path)
        
        if len(df) < 200:
            return {'symbol': symbol, 'status': 'SKIP', 'reason': f'Insufficient data ({len(df)} rows)'}
        
        # Parse date
        df.columns = df.columns.str.lower()
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        
        df.sort_values('date', inplace=True)
        
        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Clean data
        df_features = clean_data(df_features)
        
        if len(df_features) < 100:
            return {'symbol': symbol, 'status': 'FAIL', 'reason': 'Insufficient data after cleaning'}
        
        # Save features
        save_dataframe(df_features, f"{symbol.lower()}_features.csv", data_type='features')
        
        # Prepare feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 'target_direction']]
        
        # Train LSTM
        lstm_success = False
        try:
            data_split = create_train_test_split(
                df_features, feature_cols, target_col='close',
                test_size=0.2, lookback=60
            )
            
            lstm = LSTMPredictor(lookback=60)
            lstm.scaler = data_split['scaler']
            lstm.feature_scaler = data_split['feature_scaler']
            
            # Train with minimal output
            lstm.train(
                data_split['X_train'], data_split['y_train'],
                data_split['X_test'], data_split['y_test'],
                model_save_path=f"models/saved/lstm_{symbol.lower()}.h5"
            )
            
            lstm.save_model(f"models/saved/lstm_{symbol.lower()}")
            lstm_success = True
            
        except Exception as e:
            if show_details:
                print(f"    LSTM failed: {str(e)[:50]}")
        
        # Train XGBoost
        xgb_success = False
        try:
            feature_cols_xgb = [col for col in df_features.columns 
                               if col not in ['date', 'time', 'target_next_close', 
                                             'target_direction', 'close', 'open', 'high', 'low', 'volume']]
            
            data_split_xgb = prepare_classification_data(
                df_features, feature_cols_xgb,
                target_col='target_direction', test_size=0.2
            )
            
            xgb = XGBoostDirectionPredictor()
            xgb.train(
                data_split_xgb['X_train'], data_split_xgb['y_train'],
                data_split_xgb['X_test'], data_split_xgb['y_test']
            )
            
            xgb.save_model(f"models/saved/xgboost_{symbol.lower()}")
            xgb_success = True
            
        except Exception as e:
            if show_details:
                print(f"    XGBoost failed: {str(e)[:50]}")
        
        # Make prediction if both models succeeded
        prediction_result = None
        if lstm_success and xgb_success:
            try:
                ensemble = EnsemblePredictor()
                ensemble.set_models(lstm, xgb)
                #prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)
                prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)
                #prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)
                
                prediction_result = {
                    'current_price': prediction['current_price'],
                    'predicted_price': prediction['predicted_price'],
                    'change_pct': prediction['predicted_change_pct'],
                    'direction': prediction['direction'],
                    'confidence': prediction['confidence'],
                    'recommendation': prediction['recommendation']
                }
            except:
                pass
        
        # Determine status
        if lstm_success and xgb_success:
            status = 'SUCCESS'
        elif lstm_success or xgb_success:
            status = 'PARTIAL'
        else:
            status = 'FAIL'
        
        return {
            'symbol': symbol,
            'status': status,
            'rows': len(df),
            'features': len(df_features),
            'lstm': lstm_success,
            'xgboost': xgb_success,
            'prediction': prediction_result
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'status': 'ERROR',
            'reason': str(e)[:100]
        }


def train_all_stocks(stock_list=None, max_stocks=None, show_details=False):
    """Train models for all stocks"""
    
    print("\n" + "="*70)
    print("PSX STOCK PREDICTOR - BATCH TRAINING")
    print("="*70)
    print(f"Start Time: {datetime.now()}\n")
    
    create_directories()
    
    # Get stock list
    historical_path = Path("data/raw/historical")
    if stock_list:
        stocks = stock_list
    else:
        stocks = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
    
    if max_stocks:
        stocks = stocks[:max_stocks]
    
    total_stocks = len(stocks)
    print(f"Training models for {total_stocks} stocks...\n")
    
    results = []
    start_time = time.time()
    
    for i, symbol in enumerate(stocks, 1):
        print(f"[{i}/{total_stocks}] Processing {symbol}...", end=' ', flush=True)
        
        result = train_single_stock(symbol, show_details=show_details)
        results.append(result)
        
        # Print result
        if result['status'] == 'SUCCESS':
            print(f"✓ SUCCESS")
        elif result['status'] == 'PARTIAL':
            print(f"⚠ PARTIAL")
        elif result['status'] == 'SKIP':
            print(f"⊘ SKIPPED ({result.get('reason', 'N/A')})")
        elif result['status'] == 'FAIL':
            print(f"✗ FAILED ({result.get('reason', 'N/A')})")
        else:
            print(f"✗ ERROR")
        
        # Show time estimate
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (total_stocks - i) * avg_time
        print(f"    Time: {elapsed/60:.1f}m elapsed | ~{remaining/60:.1f}m remaining")
    
    # Summary
    elapsed_total = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    partial_count = sum(1 for r in results if r['status'] == 'PARTIAL')
    fail_count = sum(1 for r in results if r['status'] in ['FAIL', 'ERROR'])
    skip_count = sum(1 for r in results if r['status'] == 'SKIP')
    
    print(f"Total Stocks:      {total_stocks}")
    print(f"✓ Success:         {success_count}")
    print(f"⚠ Partial:         {partial_count}")
    print(f"✗ Failed:          {fail_count}")
    print(f"⊘ Skipped:         {skip_count}")
    print(f"\nTotal Time:        {elapsed_total/60:.1f} minutes")
    print(f"Average per stock: {elapsed_total/total_stocks:.1f} seconds")
    
    # Show top predictions
    successful_predictions = [r for r in results if r['status'] == 'SUCCESS' and r.get('prediction')]
    
    if successful_predictions:
        print("\n" + "="*70)
        print("TOP OPPORTUNITIES (Strong Buy Signals)")
        print("="*70)
        
        # Sort by confidence and positive change
        buy_signals = [r for r in successful_predictions 
                      if r['prediction']['recommendation'] == 'BUY' 
                      and r['prediction']['confidence'] > 0.65]
        
        buy_signals.sort(key=lambda x: x['prediction']['confidence'], reverse=True)
        
        if buy_signals:
            print(f"\n{'Symbol':<8} {'Current':<10} {'Predicted':<10} {'Change':<8} {'Confidence':<12} {'Signal'}")
            print("-" * 70)
            
            for r in buy_signals[:10]:
                p = r['prediction']
                print(f"{r['symbol']:<8} "
                      f"PKR {p['current_price']:>6.2f}  "
                      f"PKR {p['predicted_price']:>6.2f}  "
                      f"{p['change_pct']:>+6.2f}%  "
                      f"{p['confidence']:>5.1%}       "
                      f"🟢 BUY")
        else:
            print("No strong buy signals found.")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = Path("data/predictions/batch_training_results.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "="*70)
    print("✅ BATCH TRAINING COMPLETED")
    print("="*70 + "\n")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch train models for all stocks')
    parser.add_argument('--max', type=int, help='Maximum number of stocks to train (for testing)')
    parser.add_argument('--stocks', type=str, help='Comma-separated list of specific stocks (e.g., HBL,OGDC,PPL)')
    parser.add_argument('--list', action='store_true', help='List all available stocks')
    parser.add_argument('--details', action='store_true', help='Show detailed error messages')
    
    args = parser.parse_args()
    
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
    
    stock_list = None
    if args.stocks:
        stock_list = [s.strip().upper() for s in args.stocks.split(',')]
        print(f"Training specific stocks: {', '.join(stock_list)}")
    
    results = train_all_stocks(
        stock_list=stock_list,
        max_stocks=args.max,
        show_details=args.details
    )
    
    print("\n🎉 Next Steps:")
    print("  1. Review results: data/predictions/batch_training_results.csv")
    print("  2. Launch dashboard: streamlit run dashboard/app.py")
    print("  3. Start API server: python src/api/api_server.py")


if __name__ == "__main__":
    main()