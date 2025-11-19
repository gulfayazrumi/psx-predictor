"""
Simple script to view your trained model results
"""
import pandas as pd
from pathlib import Path

def view_results():
    """View batch training results"""
    
    results_file = Path("data/predictions/batch_training_results.csv")
    
    if not results_file.exists():
        print("❌ No results file found!")
        print("Your models are trained but predictions weren't saved properly.")
        print("\nRun: python train_all.py --max 5")
        print("This will retrain 5 stocks and generate predictions.")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    print("\n" + "="*70)
    print("YOUR TRAINING RESULTS")
    print("="*70)
    
    print(f"\nTotal Stocks: {len(df)}")
    print(f"Successful: {len(df[df['status'] == 'SUCCESS'])}")
    print(f"Failed: {len(df[df['status'].isin(['FAIL', 'ERROR'])])}")
    print(f"Skipped: {len(df[df['status'] == 'SKIP'])}")
    
    # Show successful stocks
    successful = df[df['status'] == 'SUCCESS']
    
    if len(successful) > 0:
        print(f"\n✅ Successfully trained models:")
        print("-"*70)
        for _, row in successful.head(20).iterrows():
            lstm_status = "✓" if row.get('lstm') else "✗"
            xgb_status = "✓" if row.get('xgboost') else "✗"
            print(f"  {row['symbol']:<8} LSTM:{lstm_status}  XGBoost:{xgb_status}  Rows:{row.get('rows', 0)}")
    
    print("\n" + "="*70)
    print("\n📊 Your models are ready!")
    print("   Next: streamlit run dashboard/app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    view_results()