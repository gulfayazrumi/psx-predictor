"""
Incremental Training - Train only stocks that don't have models yet
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from train_v12 import train_single_stock_v12, create_directories

def get_untrained_stocks(max_stocks=None):
    """Get list of stocks that don't have v12 models yet"""
    
    # Get all available stocks
    historical_path = Path("data/raw/historical")
    all_stocks = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
    
    # Get stocks with existing models
    models_path = Path("models/v12")
    if models_path.exists():
        trained_stocks = set([f.stem.replace("lstm_", "").upper() 
                             for f in models_path.glob("lstm_*.h5")])
    else:
        trained_stocks = set()
    
    # Find untrained stocks
    untrained = [s for s in all_stocks if s not in trained_stocks]
    
    if max_stocks:
        untrained = untrained[:max_stocks]
    
    return untrained, len(trained_stocks), len(all_stocks)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train only untrained stocks')
    parser.add_argument('--max', type=int, help='Maximum number of stocks to train in this run')
    parser.add_argument('--details', action='store_true', help='Show detailed error messages')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔄 INCREMENTAL v12 TRAINING")
    print("="*70)
    
    create_directories()
    
    # Get untrained stocks
    untrained, trained_count, total_count = get_untrained_stocks(args.max)
    
    print(f"\n📊 Training Status:")
    print(f"   Total stocks: {total_count}")
    print(f"   Already trained: {trained_count}")
    print(f"   Remaining: {len(untrained)}")
    
    if not untrained:
        print("\n✅ All stocks already trained!")
        return
    
    print(f"\n🚀 Training {len(untrained)} stocks in this batch...")
    print("="*70)
    
    results = []
    
    for i, symbol in enumerate(untrained, trained_count + 1):
        print(f"\n[{i}/{total_count}] Processing {symbol}...", end=' ', flush=True)
        
        result = train_single_stock_v12(symbol, show_details=args.details)
        results.append(result)
        
        # Print result
        if result['status'] == 'SUCCESS':
            print(f"✓ SUCCESS")
        elif result['status'] == 'PARTIAL':
            print(f"⚠ PARTIAL")
        elif result['status'] == 'SKIP':
            print(f"⊘ SKIPPED ({result.get('reason', 'N/A')})")
        else:
            print(f"✗ FAILED")
    
    # Summary
    print("\n" + "="*70)
    print("📊 BATCH SUMMARY")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print(f"Trained in this batch: {success_count}")
    print(f"Total trained now: {trained_count + success_count}/{total_count}")
    print(f"Remaining: {total_count - (trained_count + success_count)}")
    
    if total_count - (trained_count + success_count) > 0:
        print(f"\n💡 Run again to train more stocks:")
        print(f"   python train_incremental.py --max 50")
    else:
        print(f"\n🎉 ALL STOCKS TRAINED!")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
