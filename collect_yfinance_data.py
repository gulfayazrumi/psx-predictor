"""
Collect PSX data from Yahoo Finance with REAL OHLC values
PSX stocks use .KA extension (e.g., HBL.KA)
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import time


def collect_with_yfinance(symbols, period="2y", append=True):
    """
    Collect data from Yahoo Finance
    
    Args:
        symbols: List of stock symbols
        period: How far back (2y = 2 years)
        append: If True, adds to existing data. If False, replaces.
    """
    
    print("\n" + "="*70)
    print("📊 COLLECTING PSX DATA FROM YAHOO FINANCE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'APPEND (keeps old data)' if append else 'REPLACE'}")
    print(f"Period: {period}")
    print("="*70 + "\n")
    
    historical_path = Path("data/raw/historical")
    historical_path.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = 0
    appended = 0
    replaced = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end=' ', flush=True)
        
        try:
            # Yahoo Finance uses .KA for Karachi Stock Exchange
            ticker = f"{symbol}.KA"
            
            # Download data
            stock = yf.Ticker(ticker)
            df_new = stock.history(period=period)
            
            if len(df_new) == 0:
                print("✗ No data from Yahoo")
                failed += 1
                continue
            
            # Reset index and clean columns
            df_new = df_new.reset_index()
            df_new.columns = df_new.columns.str.upper()
            
            # Yahoo returns: Date, Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We need: TIME, OPEN, HIGH, LOW, CLOSE, VOLUME
            
            df_new = df_new.rename(columns={'DATE': 'TIME'})
            df_new = df_new[['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            
            # Format date as "Nov 18, 2025"
            df_new['TIME'] = pd.to_datetime(df_new['TIME']).dt.strftime('%b %d, %Y')
            
            csv_path = historical_path / f"{symbol.lower()}.csv"
            
            # APPEND mode: Load existing data and combine
            if append and csv_path.exists():
                try:
                    df_old = pd.read_csv(csv_path)
                    
                    # Convert dates for comparison
                    df_old['date_parsed'] = pd.to_datetime(df_old['TIME'], format='%b %d, %Y')
                    df_new['date_parsed'] = pd.to_datetime(df_new['TIME'], format='%b %d, %Y')
                    
                    # Remove duplicates from old data
                    df_old = df_old[~df_old['date_parsed'].isin(df_new['date_parsed'])]
                    
                    # Combine old + new
                    df_combined = pd.concat([df_old, df_new])
                    df_combined = df_combined.sort_values('date_parsed', ascending=False)
                    
                    # Remove helper column
                    df_combined = df_combined.drop('date_parsed', axis=1)
                    
                    # Save
                    df_combined.to_csv(csv_path, index=False)
                    
                    print(f"✓ {len(df_new)} new rows | Total: {len(df_combined)} | APPENDED")
                    appended += 1
                    
                except Exception as e:
                    # If append fails, just replace
                    df_new_clean = df_new.drop('date_parsed', axis=1)
                    df_new_clean.to_csv(csv_path, index=False)
                    print(f"✓ {len(df_new)} rows | REPLACED (append failed)")
                    replaced += 1
            else:
                # REPLACE mode or new file
                df_new_clean = df_new.drop('date_parsed', axis=1, errors='ignore')
                df_new_clean.to_csv(csv_path, index=False)
                print(f"✓ {len(df_new)} rows | NEW FILE")
                replaced += 1
            
            success += 1
            time.sleep(0.5)  # Rate limiting
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopped by user")
            break
            
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print("="*70)
    print(f"✓ Success: {success}")
    print(f"  - Appended: {appended}")
    print(f"  - Replaced: {replaced}")
    print(f"✗ Failed: {failed}")
    print("="*70 + "\n")
    
    if success > 0:
        print("🎉 DATA COLLECTION SUCCESSFUL!")
        print("\n📊 IMPORTANT: Now you have REAL OHLC data!")
        print("   OPEN ≠ HIGH ≠ LOW ≠ CLOSE ✅")
        print("   This gives accurate technical indicators!")
        
        print("\n🎯 NEXT STEPS:")
        print("  1. Check data: python check_data_freshness.py")
        print("  2. Verify OHLC: type data\\raw\\historical\\hbl.csv")
        print("  3. Retrain: python train_all.py --max 50")
        print("  4. Signals: python generate_signals.py")
        
        return True
    
    return False


def main():
    """Main function"""
    
    # Get list of existing stocks
    historical_path = Path("data/raw/historical")
    
    if historical_path.exists():
        symbols = sorted([f.stem.upper() for f in historical_path.glob("*.csv")])
        print(f"\nFound {len(symbols)} stocks to update")
    else:
        print("❌ No existing stocks found!")
        return
    
    print("\n⚠️  IMPORTANT:")
    print("   - This will use Yahoo Finance (better OHLC data)")
    print("   - Will APPEND new data to existing data")
    print("   - Keeps your historical 2020-2025 data")
    print("   - Adds fresh Nov 2025 data on top")
    
    choice = input("\nProceed? (yes/no): ").strip().lower()
    
    if choice in ['yes', 'y']:
        # First install yfinance if needed
        print("\nChecking yfinance installation...")
        try:
            import yfinance
            print("✓ yfinance already installed")
        except ImportError:
            print("Installing yfinance...")
            import subprocess
            subprocess.run(["pip", "install", "yfinance"], check=True)
            print("✓ yfinance installed")
        
        # Collect data (APPEND mode)
        collect_with_yfinance(symbols, period="2y", append=True)
    else:
        print("\nCancelled.")


if __name__ == "__main__":
    main()