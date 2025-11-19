"""
Collect FRESH data from Sarmaaya API
Run this in the MORNING when market opens (9 AM - 11 AM PKT)
"""
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


def verify_data_freshness(df, symbol):
    """Check if data is recent"""
    if df is None or len(df) == 0:
        return False, "No data"
    
    latest_date = df['date'].max()
    days_old = (datetime.now() - latest_date).days
    
    if days_old > 7:
        return False, f"Data is {days_old} days old"
    else:
        return True, f"Fresh ({days_old} days old)"


def collect_current_data(symbols_list, days_back=730):
    """
    Collect last 2 years of data (730 days) for all stocks
    This ensures we have recent + enough historical data
    """
    
    api = SarmayaAPI()
    
    print("\n" + "="*70)
    print("COLLECTING FRESH PSX DATA")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Collecting {days_back} days of data for {len(symbols_list)} stocks")
    print("="*70 + "\n")
    
    Path("data/raw/historical").mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    fresh = 0
    stale = 0
    
    for i, symbol in enumerate(symbols_list, 1):
        print(f"[{i}/{len(symbols_list)}] {symbol}...", end=' ')
        
        try:
            # Fetch data
            df = api.get_price_history(symbol, days=days_back)
            
            if df is not None and len(df) > 0:
                # Verify freshness
                is_fresh, msg = verify_data_freshness(df, symbol)
                
                # Save to CSV
                csv_path = Path(f"data/raw/historical/{symbol.lower()}.csv")
                df.to_csv(csv_path, index=False)
                
                latest = df['date'].max().strftime('%Y-%m-%d')
                
                if is_fresh:
                    print(f"✓ {len(df)} rows | Latest: {latest} | {msg}")
                    success += 1
                    fresh += 1
                else:
                    print(f"⚠ {len(df)} rows | Latest: {latest} | {msg}")
                    success += 1
                    stale += 1
            else:
                print("✗ No data returned")
                failed += 1
            
            # Rate limiting - be nice to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total Stocks:    {len(symbols_list)}")
    print(f"✓ Success:       {success}")
    print(f"  - Fresh data:  {fresh} (within 7 days)")
    print(f"  - Stale data:  {stale} (>7 days old)")
    print(f"✗ Failed:        {failed}")
    print("="*70 + "\n")
    
    if fresh > len(symbols_list) * 0.5:  # At least 50% fresh
        print("✅ DATA IS CURRENT - Ready to retrain models!")
        return True
    else:
        print("⚠️  WARNING: Most data is stale. Try again when market is open.")
        return False


def main():
    """Main function"""
    
    # Get list of stocks from existing files or API
    historical_path = Path("data/raw/historical")
    
    if historical_path.exists():
        symbols = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
        print(f"\nFound {len(symbols)} existing stocks to update")
    else:
        print("No existing data found. Will collect KSE-100")
        api = SarmayaAPI()
        symbols = api.get_kse100_tickers()
    
    if not symbols:
        print("❌ No symbols to collect!")
        return
    
    # Confirm
    print(f"\nAbout to collect data for {len(symbols)} stocks")
    print("This will take approximately 1-2 hours")
    print("Best time: 9 AM - 11 AM PKT (when market is open)")
    
    choice = input("\nContinue? (yes/no): ").strip().lower()
    
    if choice != 'yes':
        print("Cancelled.")
        return
    
    # Collect
    is_current = collect_current_data(symbols, days_back=730)
    
    if is_current:
        print("\n🎉 Next Step: Retrain models with fresh data")
        print("   Run: python train_all.py --max 50")
    else:
        print("\n⚠️  Data collection incomplete")
        print("   Try again during market hours (9 AM - 5 PM PKT)")


if __name__ == "__main__":
    main()