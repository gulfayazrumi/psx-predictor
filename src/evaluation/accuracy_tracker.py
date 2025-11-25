"""
Accuracy Tracker Module
Compares historical predictions with actual prices to calculate accuracy metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob

class AccuracyTracker:
    def __init__(self):
        self.base_path = Path(".")
        self.history_dir = self.base_path / "reports/history"
        self.historical_data_dir = self.base_path / "data/raw/historical"
        
    def load_prediction_history(self):
        """Load all historical prediction files"""
        if not self.history_dir.exists():
            return pd.DataFrame()
            
        all_files = glob.glob(str(self.history_dir / "*.csv"))
        dfs = []
        
        for f in all_files:
            try:
                # Extract date from filename (YYYY-MM-DD.csv)
                date_str = Path(f).stem
                df = pd.read_csv(f)
                df['prediction_date'] = date_str
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def get_actual_price(self, symbol, date_str):
        """Get actual closing price for a specific date"""
        csv_path = self.historical_data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return None
            
        try:
            df = pd.read_csv(csv_path)
            # Handle different date column names
            date_col = None
            for col in ['Date', 'DATE', 'time', 'TIME']:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                return None
                
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            target_date = pd.to_datetime(date_str)
            
            # Find exact match
            row = df[df[date_col] == target_date]
            
            if not row.empty:
                # Handle different close column names
                close_col = 'Close' if 'Close' in df.columns else 'CLOSE'
                if close_col in df.columns:
                    return float(row[close_col].iloc[0])
            
            return None
        except:
            return None

    def calculate_accuracy(self):
        """Calculate accuracy metrics"""
        print("\n" + "="*70)
        print("📊 CALCULATING PREDICTION ACCURACY")
        print("="*70)
        
        predictions_df = self.load_prediction_history()
        
        if predictions_df.empty:
            print("⚠️ No prediction history found.")
            return None
            
        print(f"✓ Loaded {len(predictions_df)} historical predictions")
        
        results = []
        
        # Process each prediction
        for idx, row in predictions_df.iterrows():
            symbol = row['symbol']
            pred_date = row['prediction_date']
            predicted_price = row['predicted_price']
            predicted_direction = row['direction']
            
            # We predicted on pred_date for the NEXT day (usually)
            # Or if the file is named by the date it was generated, it predicts for the next trading day.
            # Let's assume filename date is the date prediction was made.
            # So we need to find the actual price for the next trading day?
            # Or maybe the prediction file contains the target date?
            # Usually prediction is for T+1.
            
            # Let's try to find the actual price for dates AFTER prediction date
            # Ideally we should look for the first trading day after prediction date.
            
            pred_dt = pd.to_datetime(pred_date)
            
            # Check next 5 days for a match (to handle weekends/holidays)
            actual_price = None
            actual_date = None
            
            for i in range(1, 6):
                target_date = pred_dt + timedelta(days=i)
                price = self.get_actual_price(symbol, target_date.strftime('%Y-%m-%d'))
                if price is not None:
                    actual_price = price
                    actual_date = target_date
                    break
            
            if actual_price is not None:
                # Calculate error
                error = abs(actual_price - predicted_price)
                error_pct = (error / actual_price) * 100
                
                # Calculate direction accuracy
                # We need previous day price to know actual direction
                prev_price = self.get_actual_price(symbol, pred_date)
                
                direction_correct = False
                if prev_price:
                    actual_change = actual_price - prev_price
                    actual_direction = 'UP' if actual_change > 0 else 'DOWN'
                    direction_correct = (predicted_direction == actual_direction)
                
                results.append({
                    'symbol': symbol,
                    'prediction_date': pred_date,
                    'actual_date': actual_date.strftime('%Y-%m-%d'),
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'error_abs': error,
                    'error_pct': error_pct,
                    'direction_correct': direction_correct,
                    'confidence': row.get('confidence', 0)
                })
        
        if not results:
            print("⚠️ No matching actual data found yet (predictions might be too recent).")
            return None
            
        results_df = pd.DataFrame(results)
        
        # Save detailed report
        report_path = self.base_path / "reports/accuracy_report_detailed.csv"
        results_df.to_csv(report_path, index=False)
        
        # Calculate summary stats
        summary = {
            'total_predictions': len(results_df),
            'avg_error_pct': results_df['error_pct'].mean(),
            'direction_accuracy': results_df['direction_correct'].mean() * 100,
            'high_confidence_accuracy': results_df[results_df['confidence'] > 0.7]['direction_correct'].mean() * 100
        }
        
        print(f"✓ Matched {len(results_df)} predictions with actuals")
        print(f"✓ Average Error: {summary['avg_error_pct']:.2f}%")
        print(f"✓ Direction Accuracy: {summary['direction_accuracy']:.1f}%")
        
        # Save summary per stock
        stock_summary = results_df.groupby('symbol').agg({
            'error_pct': 'mean',
            'direction_correct': 'mean',
            'prediction_date': 'count'
        }).reset_index()
        
        stock_summary.columns = ['symbol', 'avg_error_pct', 'direction_accuracy', 'count']
        stock_summary['direction_accuracy'] = stock_summary['direction_accuracy'] * 100
        
        stock_summary.to_csv(self.base_path / "reports/accuracy_report.csv", index=False)
        print(f"✓ Report saved to reports/accuracy_report.csv")
        
        return summary

if __name__ == "__main__":
    tracker = AccuracyTracker()
    tracker.calculate_accuracy()
