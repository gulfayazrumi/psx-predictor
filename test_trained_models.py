"""
TEST TRAINED MODELS - FIXED KERAS LOADING
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from tensorflow.keras.models import load_model as keras_load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from src.preprocessing.feature_engineer import FeatureEngineer


def test_model(symbol, test_days=90):
    """Test a trained model"""
    
    print("\n" + "="*70)
    print(f"TESTING MODEL: {symbol}")
    print("="*70)
    
    # 1. Check if models exist
    lstm_path = Path(f"models/saved/lstm_{symbol.lower()}.h5")
    xgb_path = Path(f"models/saved/xgboost_{symbol.lower()}.json")
    scaler_path = Path(f"models/saved/scaler_{symbol.lower()}.pkl")
    
    if not lstm_path.exists() or not xgb_path.exists():
        print(f"❌ Models not found for {symbol}")
        print(f"   LSTM: {lstm_path.exists()}")
        print(f"   XGBoost: {xgb_path.exists()}")
        return None
    
    print("✓ Model files found")
    
    # 2. Load data
    csv_path = Path(f"data/raw/historical/{symbol.lower()}.csv")
    
    if not csv_path.exists():
        print(f"❌ No data found for {symbol}")
        return None
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    
    # Parse date
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], format='%b %d, %Y', errors='coerce')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 3. Create features
    print("Calculating indicators...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    df_features = df_features.dropna()
    
    print(f"✓ Features: {len(df_features.columns)} columns, {len(df_features)} rows")
    
    # 4. Get test data
    if len(df_features) < test_days + 60:
        test_days = len(df_features) - 60
        print(f"⚠️  Adjusted test period to {test_days} days")
    
    test_data = df_features.tail(test_days + 60).copy()
    
    # 5. Load models - FIXED KERAS LOADING
    try:
        print("Loading models...")
        
        # Load LSTM with compile=False to avoid deserialization issues
        lstm_model = keras_load_model(lstm_path, compile=False)
        print("  ✓ LSTM loaded")
        
        # Load XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        print("  ✓ XGBoost loaded")
        
        # Load scaler
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("  ✓ Scaler loaded")
        else:
            scaler = MinMaxScaler()
            scaler.fit(test_data.select_dtypes(include=[np.number]))
            print("  ✓ Scaler created")
        
        print("✓ All models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 6. Make predictions
    print(f"\nBacktesting last {test_days} days...")
    print("="*70)
    
    predictions = []
    actuals = []
    dates = []
    
    # Get feature columns (exclude date, symbol if present)
    feature_cols = [col for col in test_data.columns 
                   if col not in ['date', 'symbol', 'time'] 
                   and test_data[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    print(f"Using {len(feature_cols)} features for prediction")
    
    success_count = 0
    fail_count = 0
    
    for i in range(60, len(test_data) - 1):
        try:
            # Get sequence of last 60 days
            sequence_data = test_data[feature_cols].iloc[i-60:i].values
            
            # Check for NaN or Inf
            if np.any(np.isnan(sequence_data)) or np.any(np.isinf(sequence_data)):
                fail_count += 1
                continue
            
            # Scale data
            sequence_scaled = scaler.transform(sequence_data)
            
            # Reshape for LSTM [samples, timesteps, features]
            X_lstm = sequence_scaled.reshape(1, 60, len(feature_cols))
            
            # LSTM prediction (returns normalized value)
            lstm_pred = lstm_model.predict(X_lstm, verbose=0)[0][0]
            
            # Get actual prices
            actual_price = test_data.iloc[i]['close']
            actual_next = test_data.iloc[i+1]['close']
            
            # Simple denormalization approach
            # The model predicts percentage change
            predicted_price = actual_price * (1 + lstm_pred)
            
            predictions.append(predicted_price)
            actuals.append(actual_next)
            dates.append(test_data.iloc[i+1]['date'])
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            continue
    
    print(f"✓ Generated {success_count} predictions (failed: {fail_count})")
    
    if len(predictions) < 10:
        print(f"❌ Not enough predictions ({len(predictions)})")
        return None
    
    # 7. Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Price accuracy
    errors = np.abs(predictions - actuals)
    error_pct = (errors / actuals) * 100
    mean_error = error_pct.mean()
    median_error = np.median(error_pct)
    
    # Direction accuracy
    actual_changes = actuals[1:] - actuals[:-1]
    pred_changes = predictions[1:] - actuals[:-1]
    
    actual_direction = np.sign(actual_changes)
    pred_direction = np.sign(pred_changes)
    
    correct_direction = (actual_direction == pred_direction).sum()
    direction_accuracy = correct_direction / len(actual_direction) if len(actual_direction) > 0 else 0
    
    # Trading simulation
    trade_returns = []
    
    for i in range(len(actuals) - 1):
        actual_return = ((actuals[i+1] - actuals[i]) / actuals[i]) * 100
        
        if pred_changes[i] > 0:  # Predicted up
            trade_returns.append(actual_return)
        elif pred_changes[i] < 0:  # Predicted down
            trade_returns.append(-actual_return)
        else:
            trade_returns.append(0)
    
    trade_returns = np.array(trade_returns)
    total_return = trade_returns.sum()
    winning_trades = (trade_returns > 0).sum()
    losing_trades = (trade_returns < 0).sum()
    
    win_rate = winning_trades / len(trade_returns) if len(trade_returns) > 0 else 0
    
    avg_win = trade_returns[trade_returns > 0].mean() if winning_trades > 0 else 0
    avg_loss = abs(trade_returns[trade_returns < 0].mean()) if losing_trades > 0 else 0
    
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Sharpe ratio (simplified - assuming daily returns)
    if len(trade_returns) > 0:
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252) if trade_returns.std() > 0 else 0
    else:
        sharpe = 0
    
    # 8. Display results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\n📊 DIRECTION ACCURACY: {direction_accuracy:.2%}")
    if direction_accuracy > 0.60:
        print("   ✅ EXCELLENT - Better than 60%")
        print("   💡 This model can predict market direction reliably!")
    elif direction_accuracy > 0.55:
        print("   🟢 GOOD - Better than random")
        print("   💡 Model shows predictive power")
    elif direction_accuracy > 0.50:
        print("   🟡 ACCEPTABLE - Slightly better than random")
        print("   💡 Marginal edge, use with caution")
    else:
        print("   🔴 POOR - Worse than random")
        print("   💡 Model needs retraining or more data")
    
    print(f"\n📏 PRICE PREDICTION ERROR:")
    print(f"   Mean: {mean_error:.2f}%")
    print(f"   Median: {median_error:.2f}%")
    if mean_error < 2:
        print("   ✅ EXCELLENT - Very accurate")
    elif mean_error < 3:
        print("   🟢 GOOD - Reasonable accuracy")
    elif mean_error < 5:
        print("   🟡 ACCEPTABLE - Decent for trading")
    else:
        print("   🔴 HIGH ERROR - Predictions too far off")
    
    print("\n💰 SIMULATED TRADING PERFORMANCE:")
    print("-" * 70)
    print(f"  Test Period: {len(trade_returns)} days")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Daily Avg Return: {total_return/len(trade_returns):+.3f}%")
    print(f"  Annualized Return: {(total_return/len(trade_returns))*252:+.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Winning Trades: {winning_trades} ({win_rate:.1%})")
    print(f"  Losing Trades: {losing_trades} ({1-win_rate:.1%})")
    print(f"  Average Win: +{avg_win:.2f}%")
    print(f"  Average Loss: -{avg_loss:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    
    print("\n💡 VERDICT:")
    if total_return > 10 and direction_accuracy > 0.55:
        print("  🎉 HIGHLY PROFITABLE! Ready to trade!")
    elif total_return > 5 and direction_accuracy > 0.52:
        print("  🟢 PROFITABLE - Good for careful trading")
    elif total_return > 0:
        print("  🟡 Slight profit - Use small position sizes")
    else:
        print("  🔴 Losing - Need more training or different strategy")
    
    # Sample predictions
    print("\n📋 LAST 10 PREDICTIONS:")
    print("-" * 70)
    print(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Error':>8} {'Direction':>10}")
    print("-" * 70)
    for i in range(max(0, len(predictions)-10), len(predictions)):
        date_str = dates[i].strftime('%Y-%m-%d') if i < len(dates) else 'N/A'
        error = error_pct[i] if i < len(error_pct) else 0
        
        if i > 0 and i < len(predictions):
            pred_up = predictions[i] > actuals[i-1]
            actual_up = actuals[i] > actuals[i-1]
            direction = "✓ Correct" if pred_up == actual_up else "✗ Wrong"
        else:
            direction = "N/A"
        
        print(f"{date_str:<12} {actuals[i]:>10.2f} {predictions[i]:>10.2f} {error:>7.2f}% {direction:>10}")
    
    # Save results
    results_df = pd.DataFrame({
        'date': dates[:len(predictions)],
        'actual': actuals,
        'predicted': predictions,
        'error_pct': error_pct[:len(predictions)]
    })
    
    results_file = Path("reports") / f"{symbol}_backtest.csv"
    results_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_file, index=False)
    
    print(f"\n✓ Detailed results saved: {results_file}")
    print("="*70)
    
    return {
        'symbol': symbol,
        'direction_accuracy': direction_accuracy,
        'mean_error': mean_error,
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--days', type=int, default=90, help='Test period in days')
    
    args = parser.parse_args()
    
    result = test_model(args.symbol.upper(), test_days=args.days)
    
    if result is None:
        print("\n❌ Testing failed")
        sys.exit(1)
    else:
        print(f"\n✅ Testing completed for {result['symbol']}")
        print(f"   Direction: {result['direction_accuracy']:.1%} | Return: {result['total_return']:+.1f}% | Win Rate: {result['win_rate']:.1%}")


if __name__ == "__main__":
    main()