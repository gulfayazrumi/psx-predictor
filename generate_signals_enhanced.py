"""
Enhanced signal generator with current price display
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data_collection.sarmaaya_api import SarmayaAPI

def get_current_prices():
    """Get current market prices"""
    
    api = SarmayaAPI()
    all_stocks = []
    
    for page in range(1, 11):
        stocks = api.get_all_stocks(page=page, limit=50)
        if not stocks:
            break
        all_stocks.extend(stocks)
    
    current_prices = {}
    for stock in all_stocks:
        current_prices[stock['symbol']] = {
            'current_price': stock['close'],
            'change': stock.get('change', 0),
            'change_percent': stock.get('changePercent', 0),
            'volume': stock.get('volume', 0)
        }
    
    return current_prices


def main():
    """Generate signals with current prices"""
    
    print("\n" + "="*70)
    print("📊 GENERATING ENHANCED TRADING SIGNALS")
    print("="*70)
    
    # Get current market prices
    print("\nFetching current market prices...")
    current_prices = get_current_prices()
    print(f"✓ Got prices for {len(current_prices)} stocks")
    
    # Read existing signals
    signals_file = Path("reports/trading_signals.csv")
    
    if not signals_file.exists():
        print("\n✗ No signals found. Run 'python generate_signals.py' first!")
        return
    
    signals = pd.read_csv(signals_file)
    
    # Add current market data
    signals['current_price'] = signals['symbol'].map(
        lambda x: current_prices.get(x, {}).get('current_price', 0)
    )
    signals['market_change_%'] = signals['symbol'].map(
        lambda x: current_prices.get(x, {}).get('change_percent', 0)
    )
    signals['volume'] = signals['symbol'].map(
        lambda x: current_prices.get(x, {}).get('volume', 0)
    )
    
    # Calculate prediction accuracy (current vs predicted)
    signals['prediction_diff_%'] = (
        (signals['predicted_price'] - signals['current_price']) / 
        signals['current_price'] * 100
    )
    
    # Save enhanced signals
    output_file = Path("reports/signals_enhanced.csv")
    signals.to_csv(output_file, index=False)
    
    print(f"\n✅ Enhanced signals saved: {output_file}")
    
    # Display summary
    print("\n" + "="*70)
    print("📈 MARKET SUMMARY")
    print("="*70)
    
    # Top movers
    if 'market_change_%' in signals.columns:
        gainers = signals.nlargest(5, 'market_change_%')
        losers = signals.nsmallest(5, 'market_change_%')
        
        print("\n🚀 TOP GAINERS TODAY:")
        for _, row in gainers.iterrows():
            print(f"   {row['symbol']:<8} PKR {row['current_price']:>8.2f}  "
                  f"({row['market_change_%']:>+6.2f}%)")
        
        print("\n📉 TOP LOSERS TODAY:")
        for _, row in losers.iterrows():
            print(f"   {row['symbol']:<8} PKR {row['current_price']:>8.2f}  "
                  f"({row['market_change_%']:>+6.2f}%)")
    
    # Best predictions
    strong_signals = signals[
        (abs(signals['prediction_diff_%']) > 2) & 
        (signals['confidence'] > 40)
    ]
    
    print(f"\n🎯 STRONG SIGNALS: {len(strong_signals)}")
    
    if len(strong_signals) > 0:
        print("\nTop Opportunities:")
        top = strong_signals.nlargest(5, 'confidence')
        for _, row in top.iterrows():
            direction = "📈 BUY" if row['direction'] == 'UP' else "📉 SELL"
            print(f"   {direction}  {row['symbol']:<8} "
                  f"Current: {row['current_price']:>7.2f}  "
                  f"Target: {row['predicted_price']:>7.2f}  "
                  f"Confidence: {row['confidence']:.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()