"""
Generate signals using LATEST CSV data (not old model data)
UPDATED: Uses trading_signals.csv
"""
import pandas as pd
from pathlib import Path
import numpy as np

def get_latest_prices():
    """Read latest prices directly from CSV files"""
    
    data_dir = Path("data/raw/historical")
    latest_prices = {}
    
    # Get all CSV files
    for csv_file in data_dir.glob("*.csv"):
        symbol = csv_file.stem
        
        try:
            df = pd.read_csv(csv_file)
            
            # Get FIRST row (latest date)
            if len(df) > 0:
                latest_price = float(df['CLOSE'].iloc[0])
                latest_prices[symbol] = latest_price
            
        except Exception as e:
            pass  # Skip errors
    
    return latest_prices


def generate_corrected_signals():
    """Generate signals with correct current prices"""
    
    print("\n" + "="*70)
    print("📊 GENERATING SIGNALS WITH LATEST CSV PRICES")
    print("="*70)
    
    # Get latest prices from CSV
    latest_prices = get_latest_prices()
    print(f"\n✓ Read {len(latest_prices)} prices from CSV files")
    
    # Verify 786 specifically
    if '786' in latest_prices:
        print(f"   786 CSV price: PKR {latest_prices['786']:.2f}")
    
    # Check for existing signals
    signals_file = Path("reports/trading_signals.csv")
    
    if not signals_file.exists():
        print("\n✗ No trading_signals.csv found!")
        print("\nGenerating fresh signals...")
        
        # Import and run generate_signals
        import subprocess
        subprocess.run(['python', 'generate_signals.py'])
        
        if not signals_file.exists():
            print("✗ Failed to generate signals")
            return
    
    # Read signals
    signals = pd.read_csv(signals_file)
    print(f"\n✓ Loaded {len(signals)} existing signals")
    
    # Update current prices from CSV
    updated = 0
    for idx, row in signals.iterrows():
        symbol = row['symbol']
        if symbol in latest_prices:
            old_price = row['current_price']
            new_price = latest_prices[symbol]
            
            if abs(old_price - new_price) > 0.01:  # Price changed
                signals.at[idx, 'current_price'] = new_price
                
                # Recalculate changes
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
                
                updated += 1
    
    print(f"✓ Updated {updated} prices from CSV")
    
    # Save corrected signals
    output_file = Path("reports/trading_signals.csv")
    signals.to_csv(output_file, index=False)
    
    print(f"\n✅ Corrected signals saved: {output_file}")
    
    # Verify 786
    stock_786 = signals[signals['symbol'] == '786']
    if not stock_786.empty:
        print(f"\n" + "="*70)
        print(f"📊 786 STOCK VERIFICATION")
        print("="*70)
        print(f"   Current Price:  PKR {stock_786['current_price'].iloc[0]:.2f}")
        print(f"   Predicted:      PKR {stock_786['predicted_price'].iloc[0]:.2f}")
        print(f"   Change:         {stock_786['percent_change'].iloc[0]:+.2f}%")
        print(f"   Direction:      {stock_786['direction'].iloc[0]}")
        print(f"   Confidence:     {stock_786['confidence'].iloc[0]:.1f}%")
        print(f"   Action:         {stock_786['action'].iloc[0]}")
        print("="*70)
    
    # Show top changes
    print(f"\n🔝 TOP 5 PRICE UPDATES:")
    if updated > 0:
        # Calculate which stocks had biggest price differences
        for idx, row in signals.head(5).iterrows():
            print(f"   {row['symbol']:<8} PKR {row['current_price']:>8.2f}  "
                  f"({row['percent_change']:>+6.2f}%)  {row['direction']}")
    
    print(f"\n" + "="*70)
    print(f"✅ DONE! Restart dashboard to see updated prices")
    print("="*70)


if __name__ == "__main__":
    generate_corrected_signals()