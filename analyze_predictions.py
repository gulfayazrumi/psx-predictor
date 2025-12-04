"""
Analyze trading signals to verify prediction quality
"""
import pandas as pd
from pathlib import Path

# Load trading signals
signals_path = Path("reports/trading_signals.csv")
df = pd.read_csv(signals_path)

print("\n" + "="*80)
print("PREDICTION MODEL VERIFICATION REPORT")
print("="*80)

print(f"\nTotal Signals: {len(df)}")

# Analysis 1: Are predictions different from current price?
df['price_diff'] = df['predicted_price'] - df['current_price']
df['abs_diff'] = abs(df['price_diff'])

same_price = df[df['abs_diff'] < 0.01]
print(f"\n[1] Predictions identical to current price: {len(same_price)} ({len(same_price)/len(df)*100:.1f}%)")
if len(same_price) > 0:
    print("   WARNING: Model may be copying current price!")
    print(f"   Examples: {same_price['symbol'].head(5).tolist()}")
else:
    print("   PASS: All predictions are different from current price")

# Analysis 2: Distribution of predicted changes
print(f"\n[2] Predicted Change Distribution:")
print(f"   Mean change: {df['percent_change'].mean():.2f}%")
print(f"   Median change: {df['percent_change'].median():.2f}%")
print(f"   Std dev: {df['percent_change'].std():.2f}%")
print(f"   Min change: {df['percent_change'].min():.2f}%")
print(f"   Max change: {df['percent_change'].max():.2f}%")

# Analysis 3: Directional distribution
up_count = len(df[df['direction'] == 'UP'])
down_count = len(df[df['direction'] == 'DOWN'])
print(f"\n[3] Directional Predictions:")
print(f"   UP:   {up_count} ({up_count/len(df)*100:.1f}%)")
print(f"   DOWN: {down_count} ({down_count/len(df)*100:.1f}%)")

if abs(up_count - down_count) / len(df) > 0.8:
    print("   WARNING: Heavily biased predictions!")
else:
    print("   PASS: Reasonable directional balance")

# Analysis 4: Confidence distribution
print(f"\n[4] Confidence Scores:")
print(f"   Mean: {df['confidence'].mean():.2%}")
print(f"   Median: {df['confidence'].median():.2%}")
print(f"   Min: {df['confidence'].min():.2%}")
print(f"   Max: {df['confidence'].max():.2%}")

high_conf = len(df[df['confidence'] > 0.7])
print(f"   High confidence (>70%): {high_conf} ({high_conf/len(df)*100:.1f}%)")

# Analysis 5: Sample predictions
print(f"\n[5] Sample Predictions (First 10):")
print("-"*80)
sample = df.head(10)[['symbol', 'current_price', 'predicted_price', 'percent_change', 'confidence', 'recommendation']]
for _, row in sample.iterrows():
    print(f"{row['symbol']:8s} | Current: {row['current_price']:8.2f} | Predicted: {row['predicted_price']:8.2f} | Change: {row['percent_change']:+6.2f}% | Conf: {row['confidence']:.2%} | {row['recommendation']}")

# Analysis 6: Extreme predictions
print(f"\n[6] Extreme Predictions:")
top_gains = df.nlargest(5, 'percent_change')[['symbol', 'current_price', 'predicted_price', 'percent_change', 'confidence']]
print("\nTop 5 Predicted Gains:")
for _, row in top_gains.iterrows():
    print(f"  {row['symbol']:8s}: {row['percent_change']:+6.2f}% (Current: {row['current_price']:.2f}, Predicted: {row['predicted_price']:.2f}, Conf: {row['confidence']:.2%})")

top_losses = df.nsmallest(5, 'percent_change')[['symbol', 'current_price', 'predicted_price', 'percent_change', 'confidence']]
print("\nTop 5 Predicted Losses:")
for _, row in top_losses.iterrows():
    print(f"  {row['symbol']:8s}: {row['percent_change']:+6.2f}% (Current: {row['current_price']:.2f}, Predicted: {row['predicted_price']:.2f}, Conf: {row['confidence']:.2%})")

# Analysis 7: Reasonableness check
unreasonable = df[abs(df['percent_change']) > 15]
print(f"\n[7] Unreasonable Predictions (>15% change): {len(unreasonable)}")
if len(unreasonable) > 0:
    print("   WARNING: Some predictions may be unrealistic")
    for _, row in unreasonable.iterrows():
        print(f"   {row['symbol']}: {row['percent_change']:+.2f}%")

# Final verdict
print(f"\n" + "="*80)
print("VERDICT:")
print("="*80)

issues = []
if len(same_price) > len(df) * 0.1:
    issues.append("- Too many predictions identical to current price")

if abs(up_count - down_count) / len(df) > 0.8:
    issues.append("- Heavily biased directional predictions")

if df['percent_change'].std() < 0.5:
    issues.append("- Predictions are too conservative (low variance)")

if len(unreasonable) > len(df) * 0.05:
    issues.append("- Too many unreasonable predictions")

if len(issues) == 0:
    print("[PASS] Predictions appear to be coming from the model correctly")
    print("       - Predictions differ from current prices")
    print("       - Reasonable directional balance")
    print("       - Appropriate variance in predictions")
    print("       - Confidence scores are distributed")
else:
    print("[ISSUES FOUND]")
    for issue in issues:
        print(f"  {issue}")

print("\n" + "="*80)
