"""
WORKING MODEL TESTER - HANDLES INF VALUES
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pickle

sys.path.append(str(Path(__file__).parent))
from src.preprocessing.feature_engineer import FeatureEngineer


def clean_data(data):
    """Remove inf and nan values"""
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())
    data = data.fillna(0)  # If column is all NaN, fill with 0
    return data


def test_stock(symbol, test_days=90):
    print("\n" + "="*70)
    print(f"TESTING: {symbol}")
    print("="*70)
    
    # Paths
    lstm_path = Path(f"models/saved/lstm_{symbol.lower()}.h5")
    xgb_path = Path(f"models/saved/xgboost_{symbol.lower()}.json")
    scaler_path = Path(f"models/saved/scaler_{symbol.lower()}.pkl")
    
    # Check files exist
    if not lstm_path.exists():
        print(f"❌ LSTM model not found")
        return None
    if not xgb_path.exists():
        print(f"❌ XGBoost model not found")
        return None
    
    print("✓ Model files found")
    
    # Load models
    try:
        print("Loading models...")
        lstm_model = load_model(str(lstm_path), compile=False)
        print("  ✓ LSTM loaded")
        
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))
        print("  ✓ XGBoost loaded")
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("  ✓ Scaler loaded")
        else:
            scaler = MinMaxScaler()
            print("  ⚠ Creating new scaler")
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None
    
    # Load data
    csv_path = Path(f"data/raw/historical/{symbol.lower()}.csv")
    if not csv_path.exists():
        print(f"❌ Data not found")
        return None
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'], format='%b %d, %Y', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    print(f"✓ Data loaded: {len(df)} days")
    
    # Create features
    print("Creating features...")
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    df_features = df_features.dropna()
    print(f"✓ Features created: {len(df_features)} rows")
    
    # Test data
    if len(df_features) < test_days + 60:
        test_days = len(df_features) - 60
        if test_days < 30:
            print(f"❌ Not enough data (need 90+ days, have {len(df_features)})")
            return None
    
    test_data = df_features.tail(test_days + 60).copy()
    
    # Get feature columns
    feature_cols = [c for c in test_data.columns 
                   if c not in ['date', 'time', 'symbol']
                   and pd.api.types.is_numeric_dtype(test_data[c])]
    
    print(f"Using {len(feature_cols)} features")
    
    # Clean data - CRITICAL FIX
    print("Cleaning data...")
    test_data[feature_cols] = clean_data(test_data[feature_cols])
    
    # Fit scaler if needed
    if not hasattr(scaler, 'n_features_in_'):
        print("Fitting scaler...")
        scaler.fit(test_data[feature_cols])
    
    # Make predictions
    print(f"\nMaking predictions for last {test_days} days...")
    predictions = []
    actuals = []
    dates = []
    
    success = 0
    failed = 0
    
    for i in range(60, len(test_data) - 1):
        try:
            X = test_data[feature_cols].iloc[i-60:i].values
            
            # Check for invalid data
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                failed += 1
                continue
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Reshape for LSTM
            X_lstm = X_scaled.reshape(1, 60, len(feature_cols))
            
            # Predict
            pred = lstm_model.predict(X_lstm, verbose=0)[0][0]
            actual = test_data.iloc[i+1]['close']
            current = test_data.iloc[i]['close']
            
            # Denormalize prediction
            predictions.append(pred * current)
            actuals.append(actual)
            dates.append(test_data.iloc[i+1]['date'])
            success += 1
            
        except Exception as e:
            failed += 1
            continue
    
    print(f"✓ Generated {success} predictions (failed: {failed})")
    
    if len(predictions) < 10:
        print(f"❌ Only {len(predictions)} predictions - not enough")
        return None
    
    # Calculate metrics
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
    
    correct = (np.sign(actual_changes) == np.sign(pred_changes)).sum()
    direction_acc = correct / len(actual_changes)
    
    # Trading simulation
    returns = []
    for i in range(len(actual_changes)):
        actual_ret = (actual_changes[i] / actuals[i]) * 100
        if pred_changes[i] > 0:
            returns.append(actual_ret)
        elif pred_changes[i] < 0:
            returns.append(-actual_ret)
        else:
            returns.append(0)
    
    returns = np.array(returns)
    total_return = returns.sum()
    wins = (returns > 0).sum()
    losses = (returns < 0).sum()
    win_rate = wins / len(returns) if len(returns) > 0 else 0
    
    avg_win = returns[returns > 0].mean() if wins > 0 else 0
    avg_loss = abs(returns[returns < 0].mean()) if losses > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print(f"\n📊 DIRECTION ACCURACY: {direction_acc:.2%}")
    if direction_acc > 0.60:
        print("   ✅ EXCELLENT")
    elif direction_acc > 0.55:
        print("   🟢 GOOD")
    elif direction_acc > 0.50:
        print("   🟡 ACCEPTABLE")
    else:
        print("   🔴 POOR")
    
    print(f"\n📏 PRICE ERROR:")
    print(f"   Mean: {mean_error:.2f}%")
    print(f"   Median: {median_error:.2f}%")
    
    print(f"\n💰 TRADING PERFORMANCE:")
    print(f"   Test Period: {len(returns)} days")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Avg Win: +{avg_win:.2f}%")
    print(f"   Avg Loss: -{avg_loss:.2f}%")
    print(f"   Profit Factor: {profit_factor:.2f}")
    
    print("\n💡 VERDICT:")
    if total_return > 10 and direction_acc > 0.55:
        print("   🎉 HIGHLY PROFITABLE!")
    elif total_return > 5 and direction_acc > 0.52:
        print("   🟢 PROFITABLE")
    elif total_return > 0:
        print("   🟡 Slight profit")
    else:
        print("   🔴 Losing")
    
    # Sample predictions
    print("\n📋 LAST 10 PREDICTIONS:")
    print("-" * 70)
    print(f"{'Date':<12} {'Actual':>10} {'Predicted':>10} {'Error':>8}")
    print("-" * 70)
    for i in range(max(0, len(predictions)-10), len(predictions)):
        date_str = dates[i].strftime('%Y-%m-%d')
        print(f"{date_str:<12} {actuals[i]:>10.2f} {predictions[i]:>10.2f} {error_pct[i]:>7.2f}%")
    
    # Save results
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': predictions,
        'error_pct': error_pct
    })
    
    results_file = Path("reports") / f"{symbol}_test_results.csv"
    results_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
    print("="*70)
    
    return {
        'symbol': symbol,
        'direction_accuracy': direction_acc,
        'mean_error': mean_error,
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = input("Enter stock symbol: ").strip().upper()
    
    result = test_stock(symbol)
    
    if result:
        print(f"\n✅ Test completed for {result['symbol']}")
        print(f"   Direction: {result['direction_accuracy']:.1%}")
        print(f"   Return: {result['total_return']:+.1f}%")
        print(f"   Win Rate: {result['win_rate']:.1%}")