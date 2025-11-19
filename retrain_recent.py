"""
Retrain models for stocks with updated data
Quick retrain (no full training)
"""
import subprocess
import sys
from pathlib import Path

def retrain_top_stocks():
    """Retrain the 50 stocks we generate signals for"""
    
    # List of stocks with trained models
    models_dir = Path("models/saved")
    lstm_models = list(models_dir.glob("lstm_*.h5"))
    
    # Get first 50 (or less)
    stocks_to_retrain = [m.stem.replace('lstm_', '') for m in lstm_models[:50]]
    
    print("\n" + "="*70)
    print("🔄 RETRAINING TOP 50 STOCKS")
    print("="*70)
    print(f"Stocks to retrain: {len(stocks_to_retrain)}")
    
    # Train each
    for i, symbol in enumerate(stocks_to_retrain, 1):
        print(f"\n[{i}/{len(stocks_to_retrain)}] Training {symbol}...")
        
        result = subprocess.run(
            ['python', 'train_single.py', symbol],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ {symbol} trained")
        else:
            print(f"  ✗ {symbol} failed")
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    retrain_top_stocks()