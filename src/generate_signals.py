"""
Generate trading signals from trained models
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

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
    
    # Parse predictions
    signals = []
    
    for idx, row in successful.iterrows():
        symbol = row['symbol']
        
        # Try to parse prediction (might be stored as string)
        try:
            if pd.notna(row.get('predicted_price')):
                current = row.get('current_price', 0)
                predicted = row.get('predicted_price', 0)
                
                if current > 0 and predicted > 0:
                    change_pct = (predicted - current) / current * 100
                    
                    # Determine signal
                    if change_pct > 2:
                        signal = 'STRONG BUY'
                    elif change_pct > 0.5:
                        signal = 'BUY'
                    elif change_pct < -2:
                        signal = 'STRONG SELL'
                    elif change_pct < -0.5:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    signals.append({
                        'symbol': symbol,
                        'current_price': current,
                        'predicted_price': predicted,
                        'change_pct': change_pct,
                        'signal': signal
                    })
        except:
            continue
    
    if not signals:
        print("⚠️  No signals generated. This might be due to the ensemble bug.")
        print("Make sure you ran fix_ensemble_bug.py and retested.\n")
        return
    
    signals_df = pd.DataFrame(signals)
    
    # Strong buy signals
    strong_buys = signals_df[signals_df['signal'].str.contains('BUY')].sort_values('change_pct', ascending=False)
    
    if len(strong_buys) > 0:
        print("🟢 BUY SIGNALS")
        print("-"*70)
        print(f"{'Symbol':<8} {'Current':<12} {'Predicted':<12} {'Change':<10} {'Signal'}")
        print("-"*70)
        
        for _, row in strong_buys.head(20).iterrows():
            print(f"{row['symbol']:<8} "
                  f"PKR {row['current_price']:>7.2f}  "
                  f"PKR {row['predicted_price']:>7.2f}  "
                  f"{row['change_pct']:>+6.2f}%   "
                  f"{row['signal']}")
    
    # Strong sell signals
    strong_sells = signals_df[signals_df['signal'].str.contains('SELL')].sort_values('change_pct')
    
    if len(strong_sells) > 0:
        print("\n🔴 SELL SIGNALS")
        print("-"*70)
        print(f"{'Symbol':<8} {'Current':<12} {'Predicted':<12} {'Change':<10} {'Signal'}")
        print("-"*70)
        
        for _, row in strong_sells.head(10).iterrows():
            print(f"{row['symbol']:<8} "
                  f"PKR {row['current_price']:>7.2f}  "
                  f"PKR {row['predicted_price']:>7.2f}  "
                  f"{row['change_pct']:>+6.2f}%   "
                  f"{row['signal']}")
    
    print("\n" + "="*70)
    
    # Save signals
    signals_df.to_csv('reports/trading_signals.csv', index=False)
    print("✓ Trading signals saved to: reports/trading_signals.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    generate_trading_signals()