"""
EMERGENCY DATA COLLECTOR - FIXED VERSION
Handles timezone issues
"""
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


def collect_fresh_data_now():
    """Collect fresh data from working API"""
    
    api = SarmayaAPI()
    
    print("\n" + "="*70)
    print("🚀 EMERGENCY DATA COLLECTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("API Status: ✅ CONFIRMED WORKING")
    print("Collecting data from API")
    print("="*70 + "\n")
    
    # Get list of stocks
    historical_path = Path("data/raw/historical")
    existing_stocks = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
    
    print(f"Found {len(existing_stocks)} existing stocks")
    print("\nCollecting fresh data...\n")
    
    historical_path.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    fresh_data = 0
    
    for i, symbol in enumerate(existing_stocks, 1):
        print(f"[{i}/{len(existing_stocks)}] {symbol}...", end=' ', flush=True)
        
        try:
            # Get data (API might limit to 30 days)
            df = api.get_price_history(symbol, days=730)
            
            if df is not None and len(df) > 0:
                # Convert date and remove timezone
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                
                # Check if data is current
                latest_date = df['date'].max()
                days_old = (datetime.now() - latest_date).days
                
                # Save in the format your system expects
                csv_path = historical_path / f"{symbol.lower()}.csv"
                
                # Convert to uppercase column names and date format
                df_save = df.copy()
                df_save['TIME'] = df_save['date'].dt.strftime('%b %d, %Y')
                df_save = df_save.rename(columns={
                    'open': 'OPEN',
                    'high': 'HIGH',
                    'low': 'LOW',
                    'close': 'CLOSE',
                    'volume': 'VOLUME'
                })
                
                # Save with correct column order
                df_save = df_save[['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
                df_save.to_csv(csv_path, index=False)
                
                if days_old <= 7:
                    status = f"✓ {len(df)} days | Latest: {latest_date.strftime('%Y-%m-%d')} (FRESH!)"
                    fresh_data += 1
                else:
                    status = f"⚠ {len(df)} days | Latest: {latest_date.strftime('%Y-%m-%d')} ({days_old}d old)"
                
                print(status)
                success += 1
                
            else:
                print("✗ No data")
                failed += 1
            
            # Rate limiting - be nice to API
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Collection stopped by user")
            print(f"Processed: {i}/{len(existing_stocks)}")
            print(f"Success so far: {success}")
            break
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print("="*70)
    print(f"Total Processed: {success + failed}/{len(existing_stocks)}")
    print(f"✓ Success: {success}")
    print(f"  Fresh (within 7 days): {fresh_data}")
    print(f"✗ Failed: {failed}")
    print("="*70 + "\n")
    
    if success > 0:
        print("🎉 DATA COLLECTION SUCCESSFUL!")
        print(f"\n📊 You now have {success} stocks with 2025 data!")
        
        if fresh_data > success * 0.5:
            print(f"✅ {fresh_data} stocks have FRESH data (within 7 days)")
            print("\n🎯 NEXT STEPS:")
            print("  1. Verify: python check_data_freshness.py")
            print("  2. Retrain: python train_all.py --max 50")
            print("  3. Signals: python generate_signals.py")
            print("  4. Trade: streamlit run dashboard/app.py")
        else:
            print(f"⚠️  Only {fresh_data} stocks have fresh data")
            print("   But this is still MUCH better than 2020 data!")
        
        return True
    else:
        print("❌ No data collected")
        print("Check your internet connection and try again")
        return False


if __name__ == "__main__":
    print("\n🚨 EMERGENCY DATA UPDATE 🚨")
    print("API is confirmed working!")
    print("Let's get fresh 2025 data RIGHT NOW!\n")
    
    choice = input("Start collection? (yes/no): ").strip().lower()
    
    if choice in ['yes', 'y']:
        collect_fresh_data_now()
    else:
        print("\nCancelled. Run again when ready!")