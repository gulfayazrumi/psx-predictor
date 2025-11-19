"""
Compare predictions over time to track accuracy
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def compare_predictions():
    """Compare yesterday's predictions with today's actual prices"""
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    history_dir = Path("reports/history")
    
    # Load yesterday's predictions
    yesterday_file = history_dir / f"{yesterday}.csv"
    if not yesterday_file.exists():
        print(f"No predictions found for {yesterday}")
        return
    
    yesterday_df = pd.read_csv(yesterday_file)
    
    # Load today's actual prices
    today_file = history_dir / f"{today}.csv"
    if not today_file.exists():
        print(f"No data for {today} yet")
        return
    
    today_df = pd.read_csv(today_file)
    
    # Merge
    comparison = yesterday_df[['symbol', 'predicted_price']].merge(
        today_df[['symbol', 'current_price']],
        on='symbol',
        how='inner'
    )
    
    # Calculate accuracy
    comparison['predicted'] = comparison['predicted_price']
    comparison['actual'] = comparison['current_price']
    comparison['error_pct'] = abs(
        (comparison['actual'] - comparison['predicted']) / comparison['actual'] * 100
    )
    
    # Stats
    avg_error = comparison['error_pct'].mean()
    
    print("\n" + "="*70)
    print(f"📊 PREDICTION ACCURACY - {yesterday} → {today}")
    print("="*70)
    print(f"\nAverage Error: {avg_error:.2f}%")
    print(f"Best Predictions (lowest error):")
    print(comparison.nsmallest(5, 'error_pct')[['symbol', 'predicted', 'actual', 'error_pct']])
    
    # Save accuracy report
    comparison.to_csv(f"reports/accuracy_{today}.csv", index=False)
    print(f"\n✓ Accuracy report saved: reports/accuracy_{today}.csv")

if __name__ == "__main__":
    compare_predictions()