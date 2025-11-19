"""
FINAL FIX - Clears everything and generates fresh signals
RUN THIS ONCE
"""
import shutil
from pathlib import Path
import subprocess
import pandas as pd

def final_fix():
    print("\n" + "="*70)
    print("🔧 FINAL FIX - CLEARING CACHE & REGENERATING")
    print("="*70)
    
    # Step 1: Delete OLD signal files
    print("\n[1/4] Deleting old signals...")
    reports_dir = Path("reports")
    if reports_dir.exists():
        for file in reports_dir.glob("*.csv"):
            file.unlink()
            print(f"  ✓ Deleted {file.name}")
    
    # Step 2: Verify data has today's date
    print("\n[2/4] Verifying data freshness...")
    sample_file = Path("data/raw/historical/786.csv")
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        date_col = 'TIME' if 'TIME' in df.columns else 'Date'
        latest_date = df[date_col].iloc[0]
        latest_price = df['CLOSE'].iloc[0]
        print(f"  ✓ Latest: {latest_date} - Price: {latest_price}")
    
    # Step 3: Generate FRESH signals
    print("\n[3/4] Generating fresh signals...")
    result = subprocess.run(['python', 'generate_signals.py'])
    
    if result.returncode == 0:
        print("  ✓ Signals generated")
    else:
        print("  ✗ Failed to generate signals")
        return False
    
    # Step 4: Verify new signals
    print("\n[4/4] Verifying new signals...")
    signals_file = Path("reports/trading_signals.csv")
    if signals_file.exists():
        df = pd.read_csv(signals_file)
        
        # Check 786 stock specifically
        stock_786 = df[df['symbol'] == '786']
        if not stock_786.empty:
            current = stock_786['current_price'].iloc[0]
            predicted = stock_786['predicted_price'].iloc[0]
            print(f"\n  786 Stock:")
            print(f"    Current:   PKR {current:.2f}")
            print(f"    Predicted: PKR {predicted:.2f}")
            
            if abs(current - 13.08) < 0.1:
                print("\n  ✅ SUCCESS! Data is fresh (13.08)")
            else:
                print(f"\n  ⚠️  Current price {current} - Expected 13.08")
        
        print(f"\n  ✓ Total signals: {len(df)}")
    
    print("\n" + "="*70)
    print("✅ FIX COMPLETE!")
    print("\nNext: Restart dashboard and press Ctrl+Shift+R in browser")
    print("="*70)
    
    return True

if __name__ == "__main__":
    final_fix()