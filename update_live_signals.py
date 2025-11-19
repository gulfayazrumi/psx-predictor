"""
Update trading signals with LIVE CURRENT prices from Sarmaaya
Run this every morning or before checking dashboard
"""
import pandas as pd
from pathlib import Path
from src.data_collection.sarmaaya_api import SarmayaAPI
import time

def update_with_live_prices():
    """Update signals with real-time prices"""
    
    print("\n" + "="*70)
    print("📊 UPDATING SIGNALS WITH LIVE PRICES")
    print("="*70)
    
    # Get ALL live prices from Sarmaaya
    api = SarmayaAPI()
    live_prices = {}
    
    print("\nFetching live prices from Sarmaaya...")
    for page in range(1, 11):  # 10 pages = ~500 stocks
        try:
            stocks = api.get_all_stocks(page=page, limit=50)
            if not stocks:
                break
            
            for stock in stocks:
                live_prices[stock['symbol']] = stock['close']
            
            print(f"  Page {page}: {len(stocks)} stocks")
            time.sleep(0.3)
        except:
            break
    
    print(f"\n✓ Got live prices for {len(live_prices)} stocks")
    
    # Verify 786 price
    if '786' in live_prices:
        print(f"\n  786 LIVE PRICE: PKR {live_prices['786']:.2f}")
    
    # Read existing signals
    signals_file = Path("reports/trading_signals.csv")
    
    if not signals_file.exists():
        print("\n✗ No signals file found. Run generate_signals.py first!")
        return
    
    signals = pd.read_csv(signals_file)
    print(f"\n✓ Loaded {len(signals)} signals")
    
    # Update EVERY stock with live price
    updated = 0
    for idx, row in signals.iterrows():
        symbol = row['symbol']
        
        if symbol in live_prices:
            old_price = row['current_price']
            new_price = live_prices[symbol]
            
            # Always update to live price
            signals.at[idx, 'current_price'] = new_price
            
            # Recalculate with new price
            predicted = row['predicted_price']
            price_change = predicted - new_price
            percent_change = (price_change / new_price) * 100
            
            signals.at[idx, 'price_change'] = price_change
            signals.at[idx, 'percent_change'] = percent_change
            
            # Update direction
            if percent_change > 0:
                signals.at[idx, 'direction'] = 'UP'
            elif percent_change < 0:
                signals.at[idx, 'direction'] = 'DOWN'
            else:
                signals.at[idx, 'direction'] = 'NEUTRAL'
            
            if abs(old_price - new_price) > 0.01:
                updated += 1
    
    # Save
    signals.to_csv(signals_file, index=False)
    
    print(f"\n✓ Updated {updated} prices with LIVE data")
    
    # Show 786 specifically
    stock_786 = signals[signals['symbol'] == '786']
    if not stock_786.empty:
        print("\n" + "="*70)
        print("📊 786 STOCK - UPDATED")
        print("="*70)
        print(f"   Current (LIVE):  PKR {stock_786['current_price'].iloc[0]:.2f}")
        print(f"   Predicted:       PKR {stock_786['predicted_price'].iloc[0]:.2f}")
        print(f"   Change:          {stock_786['percent_change'].iloc[0]:+.2f}%")
        print(f"   Direction:       {stock_786['direction'].iloc[0]}")
        print("="*70)
    
    # Show top movers
    print("\n🔝 TOP PRICE CHANGES:")
    #big_changes = signals.nlargest(5, lambda x: abs(x['percent_change']))
   # REPLACE WITH:
    signals['abs_change'] = signals['percent_change'].abs()
    big_changes = signals.nlargest(5, 'abs_change')
    for _, row in big_changes.head(5).iterrows():
        print(f"   {row['symbol']:<8} PKR {row['current_price']:>8.2f}  "
              f"({row['percent_change']:>+6.2f}%)  {row['direction']}")
    
    print("\n" + "="*70)
    print("✅ SIGNALS UPDATED WITH LIVE PRICES!")
    print("\nRestart dashboard to see changes:")
    print("  streamlit run dashboard/app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    update_with_live_prices()