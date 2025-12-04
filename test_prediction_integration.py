"""
Test script to verify prediction model integration and live data usage
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_collection.sarmaaya_api import SarmayaAPI
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostDirectionPredictor
from src.models.ensemble import EnsemblePredictor
from src.preprocessing.feature_engineer import FeatureEngineer

def test_prediction_with_live_data(symbol="OGDC"):
    """Test if predictions are using live data and coming from models"""
    
    print("\n" + "="*70)
    print(f"TESTING PREDICTION INTEGRATION FOR {symbol}")
    print("="*70)
    
    # Step 1: Get live price
    print(f"\n[1] Fetching live price for {symbol}...")
    api = SarmayaAPI()
    try:
        stocks = api.get_all_stocks(page=1, limit=50)
        live_stock = next((s for s in stocks if s['symbol'] == symbol), None)
        
        if not live_stock:
            print(f"ERROR: {symbol} not found in live data")
            return
        
        live_price = live_stock['close']
        print(f"SUCCESS: Live price = PKR {live_price}")
    except Exception as e:
        print(f"ERROR fetching live price: {e}")
        return
    
    # Step 2: Load historical data
    print(f"\n[2] Loading historical data...")
    csv_path = Path(f"data/raw/historical/{symbol}.csv")
    if not csv_path.exists():
        print(f"ERROR: Historical data not found")
        return
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date')
    
    last_historical_price = df['close'].iloc[-1]
    last_historical_date = df['date'].iloc[-1]
    
    print(f"SUCCESS: Last historical price = PKR {last_historical_price} (Date: {last_historical_date.date()})")
    print(f"Historical data rows: {len(df)}")
    
    # Step 3: Test WITHOUT live data injection
    print(f"\n[3] Testing prediction WITHOUT live data...")
    
    engineer = FeatureEngineer()
    df_features_old = engineer.create_all_features(df.copy())
    
    models_dir = Path("models/v12")
    lstm = LSTMPredictor(lookback=60, model_type='v12')
    lstm.load_model(models_dir / f"lstm_{symbol.lower()}")
    
    xgb = XGBoostDirectionPredictor()
    xgb.load_model(str(models_dir / f"xgboost_{symbol.lower()}"))
    
    ensemble = EnsemblePredictor()
    ensemble.set_models(lstm, xgb)
    
    feature_cols = [col for col in df_features_old.columns 
                   if col not in ['date', 'time', 'target_next_close', 
                                 'target_direction', 'close', 'open', 'high', 'low', 'volume']]
    
    pred_old = ensemble.predict_next_day(df_features_old, feature_cols)
    
    if pred_old:
        print(f"SUCCESS: Prediction (old data) = PKR {pred_old['predicted_price']:.2f}")
        print(f"Confidence: {pred_old['confidence']:.2%}")
    else:
        print(f"ERROR: Prediction failed")
        return
    
    # Step 4: Test WITH live data injection
    print(f"\n[4] Testing prediction WITH live data injection...")
    
    current_date = pd.Timestamp.now().normalize()
    
    # Inject live data
    new_row = {
        'date': current_date,
        'open': float(live_stock.get('open', live_price)),
        'high': float(live_stock.get('high', live_price)),
        'low': float(live_stock.get('low', live_price)),
        'close': float(live_price),
        'volume': float(live_stock.get('volume', 0))
    }
    
    df_with_live = df.copy()
    
    # If last row is today, update it. Otherwise append.
    if last_historical_date.date() == current_date.date():
        print(f"Updating today's row with live data")
        for col, val in new_row.items():
            if col in df_with_live.columns:
                df_with_live.iloc[-1, df_with_live.columns.get_loc(col)] = val
    else:
        print(f"Appending new row with live data")
        df_with_live = pd.concat([df_with_live, pd.DataFrame([new_row])], ignore_index=True)
    
    print(f"SUCCESS: Live data injected. New last price = PKR {df_with_live['close'].iloc[-1]}")
    
    # Create features with live data
    df_features_new = engineer.create_all_features(df_with_live)
    
    pred_new = ensemble.predict_next_day(df_features_new, feature_cols)
    
    if pred_new:
        print(f"SUCCESS: Prediction (with live) = PKR {pred_new['predicted_price']:.2f}")
        print(f"Confidence: {pred_new['confidence']:.2%}")
    else:
        print(f"ERROR: Prediction failed")
        return
    
    # Step 5: Analysis
    print(f"\n[5] ANALYSIS")
    print("="*70)
    
    print(f"\nPrice Comparison:")
    print(f"  Last Historical: PKR {last_historical_price:.2f}")
    print(f"  Current Live:    PKR {live_price:.2f}")
    print(f"  Difference:      PKR {live_price - last_historical_price:.2f} ({((live_price - last_historical_price) / last_historical_price * 100):+.2f}%)")
    
    print(f"\nPrediction Comparison:")
    print(f"  Without Live Data: PKR {pred_old['predicted_price']:.2f}")
    print(f"  With Live Data:    PKR {pred_new['predicted_price']:.2f}")
    print(f"  Difference:        PKR {pred_new['predicted_price'] - pred_old['predicted_price']:.2f}")
    
    print(f"\nPrediction vs Current:")
    print(f"  Current Price:     PKR {live_price:.2f}")
    print(f"  Predicted (Next):  PKR {pred_new['predicted_price']:.2f}")
    print(f"  Expected Change:   {((pred_new['predicted_price'] - live_price) / live_price * 100):+.2f}%")
    
    # Verify model is actually being used
    print(f"\nVERIFICATION RESULTS:")
    if abs(pred_new['predicted_price'] - pred_old['predicted_price']) > 0.01:
        print(f"  [PASS] Predictions ARE different with live data (model is responsive)")
    else:
        print(f"  [WARN] Predictions are SAME (model may not be using live data effectively)")
    
    if pred_new['predicted_price'] != live_price:
        print(f"  [PASS] Prediction is NOT just copying current price (model is working)")
    else:
        print(f"  [FAIL] Prediction equals current price (suspicious!)")
    
    # Check if prediction is reasonable
    change_pct = abs((pred_new['predicted_price'] - live_price) / live_price * 100)
    if 0.1 < change_pct < 10:
        print(f"  [PASS] Prediction change is reasonable ({change_pct:.2f}%)")
    elif change_pct < 0.1:
        print(f"  [WARN] Prediction change is very small ({change_pct:.2f}%) - model may be too conservative")
    else:
        print(f"  [WARN] Prediction change is large ({change_pct:.2f}%) - verify data quality")
    
    print("\n" + "="*70)
    
    return {
        'symbol': symbol,
        'live_price': live_price,
        'historical_price': last_historical_price,
        'pred_without_live': pred_old['predicted_price'],
        'pred_with_live': pred_new['predicted_price'],
        'expected_change_pct': ((pred_new['predicted_price'] - live_price) / live_price * 100)
    }

if __name__ == "__main__":
    # Test with a few stocks
    test_stocks = ["OGDC", "PPL", "ENGRO"]
    
    results = []
    for stock in test_stocks:
        try:
            result = test_prediction_with_live_data(stock)
            if result:
                results.append(result)
            print("\n" + "-"*70 + "\n")
        except Exception as e:
            print(f"ERROR testing {stock}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print("\n" + "="*70)
        print("SUMMARY OF ALL TESTS")
        print("="*70)
        for r in results:
            print(f"\n{r['symbol']}:")
            print(f"  Live Price: PKR {r['live_price']:.2f}")
            print(f"  Prediction: PKR {r['pred_with_live']:.2f} ({r['expected_change_pct']:+.2f}%)")
