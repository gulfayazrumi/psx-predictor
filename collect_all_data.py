"""
Complete data collection from Sarmaaya API
"""
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from src.data_collection.sarmaaya_api import SarmayaAPI
from src.utils import save_dataframe, create_directories


def collect_kse100_historical_data(days=730):
    """Collect 2 years of historical data for all KSE-100 stocks"""
    
    print("\n" + "="*70)
    print("KSE-100 HISTORICAL DATA COLLECTION")
    print("="*70)
    print(f"Start Time: {datetime.now()}\n")
    
    create_directories()
    api = SarmayaAPI()
    
    # Get KSE-100 tickers
    print("Fetching KSE-100 constituents...")
    symbols = api.get_kse100_tickers()
    
    if not symbols:
        print("✗ Failed to fetch KSE-100 tickers")
        return
    
    print(f"✓ Found {len(symbols)} KSE-100 stocks\n")
    
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {symbol}...", end=' ')
        
        try:
            df = api.get_price_history(symbol, days=days)
            
            if df is not None and len(df) > 0:
                # Save to historical folder
                save_dataframe(df, f"{symbol.lower()}.csv", data_type='historical')
                print(f"✓ {len(df)} records saved")
                successful += 1
            else:
                print(f"✗ No data")
                failed += 1
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total Stocks:  {len(symbols)}")
    print(f"✓ Successful:  {successful}")
    print(f"✗ Failed:      {failed}")
    print(f"\nEnd Time: {datetime.now()}")
    print("="*70 + "\n")


# def collect_market_overview():
#     """Collect market overview data"""
    
#     print("\nCollecting market overview...")
#     api = SarmayaAPI()
    
#     # Market view
#     market = api.get_market_view()
#     if market:
#         df = pd.DataFrame([market])
#         df['timestamp'] = datetime.now()
#         save_dataframe(df, 'market_overview.csv', data_type='raw')
#         print("✓ Market overview saved")
    
#     # Top gainers
#     gainers = api.get_market_performers('gainers', limit=20)
#     if gainers:
#         df = pd.DataFrame(gainers)
#         df['timestamp'] = datetime.now()
#         save_dataframe(df, 'top_gainers.csv', data_type='raw')
#         print("✓ Top gainers saved")
    
#     # Top losers
#     losers = api.get_market_performers('losers', limit=20)
#     if losers:
#         df = pd.DataFrame(losers)
#         df['timestamp'] = datetime.now()
#         save_dataframe(df, 'top_losers.csv', data_type='raw')
#         print("✓ Top losers saved")
    
#     # Most active
#     active = api.get_market_performers('active', limit=20)
#     if active:
#         df = pd.DataFrame(active)
#         df['timestamp'] = datetime.now()
#         save_dataframe(df, 'most_active.csv', data_type='raw')
#         print("✓ Most active saved")

def collect_market_overview():
    """Collect market overview data"""
    print("\nCollecting market overview...")
    
    api = SarmayaAPI()
    
    try:
        # Get market view
        market = api.get_market_view()
        
        if market:
            print("✓ Market overview collected")
            
            # Save market data
            market_file = Path("data/market_overview.json")
            import json
            with open(market_file, 'w') as f:
                json.dump(market, f, indent=2)
            
            print(f"✓ Market data saved to {market_file}")
        else:
            print("⚠️  Could not fetch market overview")
        
        # Try to get market performers
        try:
            gainers = api.get_market_performers('gainers', limit=20)
            losers = api.get_market_performers('losers', limit=20)
            
            if gainers:
                print(f"✓ Top gainers: {len(gainers)}")
            if losers:
                print(f"✓ Top losers: {len(losers)}")
        except Exception as e:
            print(f"⚠️  Could not fetch market performers: {e}")
    
    except Exception as e:
        print(f"✗ Market overview collection failed: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect data from Sarmaaya API')
    parser.add_argument('--kse100', action='store_true', help='Collect KSE-100 historical data')
    parser.add_argument('--market', action='store_true', help='Collect market overview')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data (default: 730)')
    parser.add_argument('--all', action='store_true', help='Collect everything')
    
    args = parser.parse_args()
    
    if args.all or args.kse100:
        collect_kse100_historical_data(days=args.days)
    
    if args.all or args.market:
        collect_market_overview()
    
    if not (args.kse100 or args.market or args.all):
        print("Please specify what to collect:")
        print("  --kse100  : Collect KSE-100 historical data")
        print("  --market  : Collect market overview")
        print("  --all     : Collect everything")


if __name__ == "__main__":
    main()