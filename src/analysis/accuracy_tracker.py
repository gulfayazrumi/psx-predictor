"""
Accuracy Tracker Module
Calculates the actual performance of past predictions by comparing them with subsequent market data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class AccuracyTracker:
    def __init__(self, history_dir="reports/history", data_dir="data/raw/historical"):
        self.history_dir = Path(history_dir)
        self.data_dir = Path(data_dir)
        self.report_path = Path("reports/accuracy_report.json")
        
    def calculate_accuracy(self):
        """
        Process all prediction history files and calculate accuracy metrics.
        """
        prediction_files = sorted(list(self.history_dir.glob("predictions_*.csv")))
        
        if not prediction_files:
            logger.warning("No prediction history files found.")
            return {}
            
        all_results = []
        
        print(f"🔍 Analyzing {len(prediction_files)} prediction history files...")
        
        for file_path in prediction_files:
            try:
                # Parse date from filename: predictions_YYYY-MM-DD.csv
                date_str = file_path.stem.replace("predictions_", "")
                pred_date = pd.to_datetime(date_str)
                
                df_preds = pd.read_csv(file_path)
                
                # Check if file has necessary columns
                required_cols = ['symbol', 'current_price', 'predicted_price', 'direction', 'confidence']
                if not all(col in df_preds.columns for col in required_cols):
                    continue
                    
                # For each prediction, find the actual outcome
                for _, row in df_preds.iterrows():
                    symbol = row['symbol']
                    result = self._verify_prediction(symbol, row, pred_date)
                    if result:
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        # Aggregate results
        if not all_results:
            return {}
            
        df_results = pd.DataFrame(all_results)
        
        # Calculate metrics
        total_predictions = len(df_results)
        correct_direction = len(df_results[df_results['direction_correct']])
        accuracy = (correct_direction / total_predictions) * 100 if total_predictions > 0 else 0
        
        avg_price_error = df_results['price_error_pct'].mean()
        
        # Group by confidence
        by_confidence = df_results.groupby('confidence_bucket').agg({
            'direction_correct': ['count', 'sum', 'mean'],
            'price_error_pct': 'mean'
        })
        
        # Prepare report
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_predictions': int(total_predictions),
                'correct_predictions': int(correct_direction),
                'accuracy_pct': round(accuracy, 2),
                'avg_price_error_pct': round(avg_price_error, 2)
            },
            'detailed_results': df_results.to_dict(orient='records'),
            'by_confidence': {}
        }
        
        # Format confidence stats
        for bucket, stats in by_confidence.iterrows():
            report['by_confidence'][bucket] = {
                'total': int(stats['direction_correct']['count']),
                'correct': int(stats['direction_correct']['sum']),
                'accuracy': round(stats['direction_correct']['mean'] * 100, 2),
                'avg_error': round(stats['price_error_pct']['mean'], 2)
            }
            
        # Save report
        with open(self.report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        print(f"✅ Accuracy report generated: {accuracy:.2f}% accuracy over {total_predictions} predictions")
        return report

    def _verify_prediction(self, symbol, pred_row, pred_date):
        """
        Compare a single prediction with actual historical data.
        """
        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return None
            
        try:
            df_hist = pd.read_csv(csv_path)
            
            # Normalize columns
            df_hist.columns = df_hist.columns.str.strip().str.lower()
            date_col = 'time' if 'time' in df_hist.columns else 'date'
            df_hist['date'] = pd.to_datetime(df_hist[date_col])
            df_hist = df_hist.sort_values('date')
            
            # Find the NEXT trading day after prediction date
            # We look for the first record where date > pred_date
            # Note: pred_date is from the filename (e.g., 2025-11-27)
            # If prediction was made on 27th (evening), it's for the 28th.
            
            # Filter for dates strictly greater than pred_date
            future_data = df_hist[df_hist['date'] > pred_date]
            
            if future_data.empty:
                return None # Outcome not yet known
                
            # The actual outcome is the FIRST day after the prediction
            actual_row = future_data.iloc[0]
            actual_date = actual_row['date']
            actual_close = float(actual_row['close'])
            
            # If the gap is too large (e.g. > 5 days), maybe it's not valid?
            # But weekends/holidays exist. Let's assume < 7 days is fine.
            if (actual_date - pred_date).days > 7:
                return None
                
            # Calculate Accuracy
            pred_price = float(pred_row['predicted_price'])
            current_price_at_pred = float(pred_row['current_price'])
            pred_direction = pred_row['direction'] # UP or DOWN
            
            # Actual movement
            if actual_close > current_price_at_pred:
                actual_direction = 'UP'
            elif actual_close < current_price_at_pred:
                actual_direction = 'DOWN'
            else:
                actual_direction = 'FLAT'
                
            is_correct = (pred_direction == actual_direction)
            
            # Price Error
            error_pct = abs(pred_price - actual_close) / actual_close * 100
            
            # Confidence Bucket
            conf = float(pred_row['confidence'])
            if conf >= 0.7:
                bucket = 'High (70%+)'
            elif conf >= 0.6:
                bucket = 'Medium (60-70%)'
            else:
                bucket = 'Low (<60%)'
                
            return {
                'symbol': symbol,
                'prediction_date': pred_date.strftime('%Y-%m-%d'),
                'target_date': actual_date.strftime('%Y-%m-%d'),
                'predicted_price': pred_price,
                'current_price_at_pred': current_price_at_pred,
                'actual_price': actual_close,
                'predicted_direction': pred_direction,
                'actual_direction': actual_direction,
                'direction_correct': is_correct,
                'price_error_pct': error_pct,
                'confidence': conf,
                'confidence_bucket': bucket
            }
            
        except Exception as e:
            # logger.error(f"Error verifying {symbol}: {e}")
            return None

if __name__ == "__main__":
    tracker = AccuracyTracker()
    tracker.calculate_accuracy()
