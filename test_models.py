"""
TEST TRAINED MODELS DIRECTLY
Works with your saved model files without needing ensemble_predictor
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
    df_features = engineer.create_features(df)
    df_features = df_features.dropna()
    
    print(f"✓ Features: {len(df_features.columns)} columns, {len(df_features)} rows")
    
    # 4. Get test data
    if len(df_features) < test_days + 60:
        test_days = len(df_features) - 60
        print(f"⚠️  Adjusted test period to {test_days} days")
    
    test_data = df_features.tail(test_days + 60).copy()
    
    # 5. Load models
    try:
        print("Loading models...")
        lstm_model = keras_load_model(lstm_path)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = MinMaxScaler()
            scaler.fit(test_data.select_dtypes(include=[np.number]))
        
        print("✓ Models loaded")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
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
                   and test_data[col].dtype in [np.float64, np.int64]]
    
    print(f"Using {len(feature_cols)} features for prediction")
    
    for i in range(60, len(test_data) - 1):
        try:
            # Get sequence of last 60 days
            sequence_data = test_data[feature_cols].iloc[i-60:i].values
            
            # Scale data
            sequence_scaled = scaler.transform(sequence_data)
            
            # Reshape for LSTM [samples, timesteps, features]
            X_lstm = sequence_scaled.reshape(1, 60, len(feature_cols))
            
            # LSTM prediction (price)
            lstm_pred = lstm_model.predict(X_lstm, verbose=0)[0][0]
            
            # Get actual next day price
            actual_price = test_data.iloc[i]['close']
            actual_next = test_data.iloc[i+1]['close']
            
            # Denormalize LSTM prediction
            # (assuming it predicts normalized values)
            predicted_price = lstm_pred * actual_price  # Simple scaling
            
            predictions.append(predicted_price)
            actuals.append(actual_next)
            dates.append(test_data.iloc[i+1]['date'])
            
        except Exception as e:
            # Skip this prediction if error
            continue
    
    if len(predictions) < 10:
        print(f"❌ Not enough predictions ({len(predictions)}). Need at least 10.")
        return None
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    # 7. Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Price accuracy
    errors = np.abs(predictions - actuals)
    error_pct = (errors / actuals) * 100
    mean_error = error_pct.mean()
    
    # Direction accuracy
    actual_changes = actuals[1:] - actuals[:-1]
    pred_changes = predictions[1:] - actuals[:-1]
    
    actual_direction = np.sign(actual_changes)
    pred_direction = np.sign(pred_changes)
    
    correct_direction = (actual_direction == pred_direction).sum()
    direction_accuracy = correct_direction / len(actual_direction) if len(actual_direction) > 0 else 0
    
    # Trading simulation
    trade_returns = []
    current_price = actuals[0]
    
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
    
    # 8. Display results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\n📊 Direction Accuracy: {direction_accuracy:.2%}")
    if direction_accuracy > 0.60:
        print("   ✅ EXCELLENT - Better than 60%")
    elif direction_accuracy > 0.55:
        print("   🟢 GOOD - Better than random")
    elif direction_accuracy > 0.50:
        print("   🟡 ACCEPTABLE")
    else:
        print("   🔴 POOR - Needs improvement")
    
    print(f"\n📏 Average Price Error: {mean_error:.2f}%")
    if mean_error < 2:
        print("   ✅ EXCELLENT")
    elif mean_error < 3:
        print("   🟢 GOOD")
    elif mean_error < 5:
        print("   🟡 ACCEPTABLE")
    else:
        print("   🔴 HIGH ERROR")
    
    print("\n💰 SIMULATED TRADING PERFORMANCE:")
    print("-" * 70)
    print(f"  Test Period: {len(trade_returns)} days")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Winning Trades: {winning_trades} ({win_rate:.1%})")
    print(f"  Losing Trades: {losing_trades} ({1-win_rate:.1%})")
    print(f"  Average Win: +{avg_win:.2f}%")
    print(f"  Average Loss: -{avg_loss:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    
    if total_return > 10:
        print("\n  🎉 HIGHLY PROFITABLE!")
    elif total_return > 5:
        print("\n  🟢 PROFITABLE")
    elif total_return > 0:
        print("\n  🟡 Slight profit")
    else:
        print("\n  🔴 Losing")
    
    # Sample predictions
    print("\n📋 SAMPLE PREDICTIONS (Last 10 days):")
    print("-" * 70)
    print(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Error':>8}")
    print("-" * 70)
    for i in range(max(0, len(predictions)-10), len(predictions)):
        date_str = dates[i].strftime('%Y-%m-%d') if i < len(dates) else 'N/A'
        error = error_pct[i] if i < len(error_pct) else 0
        print(f"{date_str:<12} {actuals[i]:>10.2f} {predictions[i]:>10.2f} {error:>7.2f}%")
    
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
    
    print(f"\n✓ Results saved: {results_file}")
    print("="*70)
    
    return {
        'symbol': symbol,
        'direction_accuracy': direction_accuracy,
        'mean_error': mean_error,
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor
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


if __name__ == "__main__":
    main()