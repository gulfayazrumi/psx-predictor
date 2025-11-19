"""
Fix data format and update with live prices
Handles DD-MMM-YY format correctly
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data_collection.sarmaaya_api import SarmayaAPI
import time

def get_live_prices():
    """Fetch today's live prices"""
    
    api = SarmayaAPI()
    
    print("\n" + "="*70)
    print("📊 FETCHING LIVE PRICES FROM SARMAAYA")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_stocks = []
    page = 1
    
    while page <= 10:  # Max 10 pages = ~500 stocks
        stocks = api.get_all_stocks(page=page, limit=50)
        if not stocks:
            break
        all_stocks.extend(stocks)
        print(f"  Page {page}: {len(stocks)} stocks")
        page += 1
        time.sleep(0.3)
    
    print(f"\n✓ Total: {len(all_stocks)} stocks")
    
    # Convert to dict for easy lookup
    live_prices = {}
    for stock in all_stocks:
        live_prices[stock['symbol']] = {
            'close': stock['close'],
            'open': stock.get('open', stock['close']),
            'high': stock.get('high', stock['close']),
            'low': stock.get('low', stock['close']),
            'volume': stock.get('volume', 0),
            'change': stock.get('change', 0),
            'changePercent': stock.get('changePercent', 0)
        }
    
    return live_prices


def update_csv_with_live(symbol, live_data):
    """Update a single CSV file with live data"""
    
    csv_file = Path(f"data/raw/historical/{symbol}.csv")
    
    if not csv_file.exists():
        return False
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Find date column (TIME or Date)
        date_col = None
        for col in ['TIME', 'Date', 'time', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return False
        
        # Parse dates - try multiple formats
        df[date_col] = df[date_col].astype(str)
        
        # Today's date in DD-MMM-YY format (matching your CSV)
        today = datetime.now()
        today_str = today.strftime('%d-%b-%y')  # "18-Nov-25"
        
        # Check if today already exists
        if today_str in df[date_col].values:
            # Update existing row
            mask = df[date_col] == today_str
            df.loc[mask, 'CLOSE'] = live_data['close']
            df.loc[mask, 'OPEN'] = live_data['open']
            df.loc[mask, 'HIGH'] = live_data['high']
            df.loc[mask, 'LOW'] = live_data['low']
            df.loc[mask, 'VOLUME'] = live_data['volume']
        else:
            # Insert new row at top (newest first)
            new_row = {
                date_col: today_str,
                'OPEN': live_data['open'],
                'HIGH': live_data['high'],
                'LOW': live_data['low'],
                'CLOSE': live_data['close'],
                'VOLUME': live_data['volume']
            }
            df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
        
        # Save back
        df.to_csv(csv_file, index=False)
        return True
        
    except Exception as e:
        print(f"  ✗ {symbol}: {e}")
        return False


def main():
    """Main update function"""
    
    print("\n" + "="*70)
    print("🔄 UPDATING ALL STOCK DATA WITH LIVE PRICES")
    print("="*70)
    
    # Get live prices
    live_prices = get_live_prices()
    
    if not live_prices:
        print("✗ No live prices fetched!")
        return
    
    # Update each CSV
    print("\nUpdating CSV files...")
    updated = 0
    errors = 0
    
    for symbol, data in live_prices.items():
        if update_csv_with_live(symbol, data):
            updated += 1
            if updated % 50 == 0:
                print(f"  ✓ Updated {updated} stocks...")
        else:
            errors += 1
    
    print("\n" + "="*70)
    print(f"✅ COMPLETE!")
    print(f"   Updated: {updated} stocks")
    print(f"   Errors: {errors} stocks")
    print("="*70)
    
    # Verify a few stocks
    print("\n📋 Verification (sample stocks):")
    for symbol in ['786', 'HBL', 'OGDC', 'LUCK']:
        csv_file = Path(f"data/raw/historical/{symbol}.csv")
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            date_col = 'TIME' if 'TIME' in df.columns else 'Date'
            latest_date = df[date_col].iloc[0]
            latest_price = df['CLOSE'].iloc[0]
            print(f"   {symbol:<8} Date: {latest_date}  Close: {latest_price:.2f}")


if __name__ == "__main__":
    main()