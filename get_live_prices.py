"""
FIXED: Update historical data with live prices
Handles TIME column and reverse chronological order
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data_collection.sarmaaya_api import SarmayaAPI

def update_historical_with_live():
    """Update historical CSV files with today's live price"""
    
    print("\n" + "="*70)
    print("🔄 UPDATING HISTORICAL DATA WITH LIVE PRICES")
    print("="*70)
    
    # Check if live prices exist
    live_file = Path("data/live/latest_prices.csv")
    if not live_file.exists():
        print("✗ No live prices found. Run get_live_prices.py first!")
        return
    
    # Get live prices
    live_df = pd.read_csv(live_file)
    
    historical_dir = Path("data/raw/historical")
    updated = 0
    errors = 0
    
    for _, live_row in live_df.iterrows():
        symbol = live_row['symbol']
        csv_file = historical_dir / f"{symbol}.csv"
        
        if not csv_file.exists():
            continue
        
        try:
            # Read historical data
            df = pd.read_csv(csv_file)
            
            # Handle both 'Date' and 'TIME' column names
            date_col = 'TIME' if 'TIME' in df.columns else 'Date'
            
            if date_col not in df.columns:
                print(f"  ⚠️  {symbol}: No date column found")
                errors += 1
                continue
            
            # Parse dates - handle both formats
            try:
                # Try parsing as "Nov 17, 2025" format
                df[date_col] = pd.to_datetime(df[date_col], format='%b %d, %Y')
            except:
                try:
                    # Try standard format
                    df[date_col] = pd.to_datetime(df[date_col])
                except:
                    print(f"  ⚠️  {symbol}: Could not parse dates")
                    errors += 1
                    continue
            
            # Get today's date
            today = datetime.now().date()
            today_str = today.strftime('%b %d, %Y')  # Match format
            
            # Check if today exists
            today_exists = (df[date_col].dt.date == today).any()
            
            if today_exists:
                # Update today's row
                mask = df[date_col].dt.date == today
                df.loc[mask, 'CLOSE'] = live_row['close']
                df.loc[mask, 'OPEN'] = live_row.get('open', live_row['close'])
                df.loc[mask, 'HIGH'] = live_row.get('high', live_row['close'])
                df.loc[mask, 'LOW'] = live_row.get('low', live_row['close'])
                df.loc[mask, 'VOLUME'] = live_row.get('volume', 0)
            else:
                # Add new row for today
                new_row = {
                    date_col: pd.Timestamp(today),
                    'OPEN': live_row.get('open', live_row['close']),
                    'HIGH': live_row.get('high', live_row['close']),
                    'LOW': live_row.get('low', live_row['close']),
                    'CLOSE': live_row['close'],
                    'VOLUME': live_row.get('volume', 0)
                }
                df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
            
            # Sort by date (newest first to match existing format)
            df = df.sort_values(date_col, ascending=False)
            
            # Format dates back to original format
            df[date_col] = df[date_col].dt.strftime('%b %d, %Y')
            
            # Save updated data
            df.to_csv(csv_file, index=False)
            updated += 1
            
            if updated % 50 == 0:
                print(f"  ✓ Updated {updated} stocks...")
        
        except Exception as e:
            print(f"  ✗ {symbol}: {str(e)}")
            errors += 1
    
    print("\n" + "="*70)
    print(f"✓ Updated: {updated} stocks")
    print(f"✗ Errors: {errors} stocks")
    print("="*70)
    
    return updated


def verify_update(symbol="HBL"):
    """Verify that update worked"""
    
    csv_file = Path(f"data/raw/historical/{symbol}.csv")
    
    if not csv_file.exists():
        print(f"✗ {symbol}.csv not found")
        return
    
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*70)
    print(f"📊 VERIFICATION: {symbol}")
    print("="*70)
    
    date_col = 'TIME' if 'TIME' in df.columns else 'Date'
    
    print(f"\nMost recent data:")
    print(df.head(3).to_string())
    
    # Check if today exists
    today = datetime.now().strftime('%b %d, %Y')
    
    if df[date_col].iloc[0] == today:
        print(f"\n✅ SUCCESS! Today's date ({today}) is present!")
        print(f"   Close price: {df['CLOSE'].iloc[0]}")
    else:
        print(f"\n⚠️  Today's date ({today}) NOT found")
        print(f"   Latest date: {df[date_col].iloc[0]}")
    
    print("="*70)


if __name__ == "__main__":
    # Update all stocks
    updated = update_historical_with_live()
    
    # Verify with HBL
    if updated > 0:
        verify_update("HBL")
        
        # Also check a few more
        print("\n📋 Spot checking other stocks:")
        for symbol in ['OGDC', 'LUCK', 'FFC']:
            csv_file = Path(f"data/raw/historical/{symbol}.csv")
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                date_col = 'TIME' if 'TIME' in df.columns else 'Date'
                print(f"  {symbol}: Latest = {df[date_col].iloc[0]}")