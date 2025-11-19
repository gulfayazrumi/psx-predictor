"""
Generate trading signals from trained models
FIXED VERSION - Correctly parses predictions
"""
import pandas as pd
from pathlib import Path
import ast

def generate_trading_signals():
    """Generate signals from batch training results"""
    
    results_file = Path("data/predictions/batch_training_results.csv")
    
    if not results_file.exists():
        print("❌ No training results found!")
        print("Run: python train_all.py first")
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Filter successful predictions
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    print("\n" + "="*70)
    print("TRADING SIGNALS REPORT")
    print("="*70)
    print(f"Total Stocks Analyzed: {len(df)}")
    print(f"Successful Predictions: {len(successful)}")
    print(f"Success Rate: {len(successful)/len(df)*100:.1f}%\n")
    
    # Parse predictions column
    signals = []
    
    for idx, row in successful.iterrows():
        try:
            pred_str = row.get('prediction', '{}')
            
            # Skip if no prediction
            if pd.isna(pred_str) or pred_str == '{}' or pred_str == '':
                continue
            
            # Parse the prediction dictionary
            pred_dict = ast.literal_eval(pred_str)
            
            symbol = row['symbol']
            current = pred_dict.get('current_price', 0)
            predicted = pred_dict.get('predicted_price', 0)
            change_pct = pred_dict.get('change_pct', 0)
            direction = pred_dict.get('direction', 'HOLD')
            confidence = pred_dict.get('confidence', 0)
            recommendation = pred_dict.get('recommendation', 'HOLD')
            
            if current > 0 and predicted > 0:
                signals.append({
                    'symbol': symbol,
                    'current_price': current,
                    'predicted_price': predicted,
                    'change_pct': change_pct,
                    'direction': direction,
                    'confidence': confidence,
                    'recommendation': recommendation
                })
        except Exception as e:
            continue
    
    if not signals:
        print("⚠️  No signals could be parsed from predictions.")
        print("This might indicate a data format issue.\n")
        return
    
    signals_df = pd.DataFrame(signals)
    
    print(f"✓ Successfully parsed {len(signals_df)} predictions\n")
    
    # Filter by change threshold
    print("="*70)
    print("FILTERING CRITERIA")
    print("="*70)
    print("Minimum change: ±2.0%")
    print("Minimum confidence: 30%\n")
    
    # Strong buy signals (>2% up, >30% confidence)
    strong_buys = signals_df[
        (signals_df['change_pct'] > 2.0) & 
        (signals_df['confidence'] > 0.30)
    ].sort_values('change_pct', ascending=False)
    
    # Strong sell signals (<-2% down, >30% confidence)
    strong_sells = signals_df[
        (signals_df['change_pct'] < -2.0) & 
        (signals_df['confidence'] > 0.30)
    ].sort_values('change_pct')
    
    # Display results
    if len(strong_buys) > 0:
        print("="*70)
        print("🟢 BUY SIGNALS (Expected Gain > 2%)")
        print("="*70)
        print(f"{'Symbol':<8} {'Current':<12} {'Predicted':<12} {'Change':<10} {'Confidence'}")
        print("-"*70)
        
        for _, row in strong_buys.head(20).iterrows():
            print(f"{row['symbol']:<8} "
                  f"PKR {row['current_price']:>7.2f}  "
                  f"PKR {row['predicted_price']:>7.2f}  "
                  f"{row['change_pct']:>+6.2f}%   "
                  f"{row['confidence']:>5.1%}")
        
        print(f"\nTotal BUY opportunities: {len(strong_buys)}")
    else:
        print("\n⚠️  No strong BUY signals found (>2% gain, >30% confidence)")
    
    if len(strong_sells) > 0:
        print("\n" + "="*70)
        print("🔴 SELL SIGNALS (Expected Loss > 2%)")
        print("="*70)
        print(f"{'Symbol':<8} {'Current':<12} {'Predicted':<12} {'Change':<10} {'Confidence'}")
        print("-"*70)
        
        for _, row in strong_sells.head(20).iterrows():
            print(f"{row['symbol']:<8} "
                  f"PKR {row['current_price']:>7.2f}  "
                  f"PKR {row['predicted_price']:>7.2f}  "
                  f"{row['change_pct']:>+6.2f}%   "
                  f"{row['confidence']:>5.1%}")
        
        print(f"\nTotal SELL warnings: {len(strong_sells)}")
    else:
        print("\n⚠️  No strong SELL signals found (<-2% loss, >30% confidence)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("MARKET SENTIMENT SUMMARY")
    print("="*70)
    
    bullish = len(signals_df[signals_df['change_pct'] > 0])
    bearish = len(signals_df[signals_df['change_pct'] < 0])
    neutral = len(signals_df) - bullish - bearish
    
    print(f"Bullish (Up):    {bullish} stocks ({bullish/len(signals_df)*100:.1f}%)")
    print(f"Bearish (Down):  {bearish} stocks ({bearish/len(signals_df)*100:.1f}%)")
    print(f"Neutral:         {neutral} stocks ({neutral/len(signals_df)*100:.1f}%)")
    
    avg_change = signals_df['change_pct'].mean()
    print(f"\nAverage Expected Change: {avg_change:+.2f}%")
    
    if avg_change > 1:
        print("📊 Market Outlook: BULLISH 🟢")
    elif avg_change < -1:
        print("📊 Market Outlook: BEARISH 🔴")
    else:
        print("📊 Market Outlook: NEUTRAL 🟡")
    
    print("="*70)
    
    # Save to CSV
    signals_df.to_csv('reports/trading_signals.csv', index=False)
    print("\n✓ Trading signals saved to: reports/trading_signals.csv")
    
    # Save filtered signals
    if len(strong_buys) > 0:
        strong_buys.to_csv('reports/buy_signals.csv', index=False)
        print("✓ Buy signals saved to: reports/buy_signals.csv")
    
    if len(strong_sells) > 0:
        strong_sells.to_csv('reports/sell_signals.csv', index=False)
        print("✓ Sell signals saved to: reports/sell_signals.csv")
    
    print("="*70 + "\n")
    
    return signals_df


if __name__ == "__main__":
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    generate_trading_signals()