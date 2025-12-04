"""
Verify model predictions against actual market outcomes.
Calculates accuracy metrics and logs performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI

def verify_predictions():
    print("\n" + "="*70)
    print("🔍 VERIFYING PREDICTION ACCURACY")
    print("="*70)
    
    # 1. Find the most recent prediction file (before today)
    history_dir = Path("reports/history")
    if not history_dir.exists():
        print("❌ No prediction history found.")
        return
        
    # Get all prediction files
    pred_files = sorted(list(history_dir.glob("predictions_*.csv")))
    if not pred_files:
        print("❌ No prediction files found.")
        return
        
    # We want to verify the *previous* trading day's prediction against *today's* price
    # Or verify *yesterday's* prediction against *today's* price
    
    # For now, let's just take the last file that isn't today's (if possible)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    target_file = None
    for f in reversed(pred_files):
        if f.stem != f"predictions_{today_str}":
            target_file = f
            break
            
    if not target_file:
        print("⚠️ Only found today's predictions. Cannot verify yet (wait for tomorrow).")
        # Optional: Check if we can verify today's prediction against live price?
        # No, prediction is for "Next Day". So we need yesterday's prediction.
        return

    print(f"📄 Verifying predictions from: {target_file.name}")
    df_preds = pd.read_csv(target_file)
    
    # 2. Get Live/Actual Prices
    print("📡 Fetching current market prices...")
    api = SarmayaAPI()
    live_prices = {}
    
    # Fetch all stocks (simplified loop)
    for page in range(1, 11):
        try:
            stocks = api.get_all_stocks(page=page, limit=50)
            if not stocks: break
            for s in stocks:
                live_prices[s['symbol']] = s['close']
        except: break
        
    # 3. Compare
    results = []
    
    for _, row in df_preds.iterrows():
        symbol = row['symbol']
        predicted_price = row['predicted_price']
        predicted_direction = row['direction']
        
        if symbol in live_prices:
            actual_price = live_prices[symbol]
            
            # Calculate Error
            error = actual_price - predicted_price
            abs_error = abs(error)
            pct_error = (abs_error / actual_price) * 100
            
            # Calculate Direction Accuracy
            # We need the "previous close" (the price at the time of prediction) to know actual direction
            # The prediction file has 'current_price' which was the price *at prediction time*
            prev_close = row['current_price']
            
            actual_change = actual_price - prev_close
            actual_direction = 'UP' if actual_change > 0 else 'DOWN'
            
            direction_correct = (predicted_direction == actual_direction)
            
            results.append({
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'prediction_date': target_file.stem.replace('predictions_', ''),
                'prev_close': prev_close,
                'predicted_price': predicted_price,
                'actual_price': actual_price,
                'error': error,
                'pct_error': pct_error,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'direction_correct': direction_correct
            })
            
    if not results:
        print("❌ No matching stocks found in live data.")
        return
        
    # 4. Aggregate Metrics
    df_res = pd.DataFrame(results)
    
    avg_mape = df_res['pct_error'].mean()
    direction_accuracy = (df_res['direction_correct'].sum() / len(df_res)) * 100
    
    print(f"\n📊 ACCURACY REPORT ({len(df_res)} stocks)")
    print(f"   Directional Accuracy: {direction_accuracy:.1f}%")
    print(f"   Average Error (MAPE): {avg_mape:.2f}%")
    
    # 5. Save Report
    report_path = Path("reports/model_performance.csv")
    
    # Append to existing report if exists
    if report_path.exists():
        df_res.to_csv(report_path, mode='a', header=False, index=False)
    else:
        df_res.to_csv(report_path, index=False)
        
    print(f"✅ Detailed report saved to {report_path}")

if __name__ == "__main__":
    verify_predictions()
