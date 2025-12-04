import json

# Load monthly top picks
with open('reports/monthly_top_picks.json', 'r') as f:
    picks = json.load(f)

# Show summary
print("\n📊 MONTHLY TOP PICKS SUMMARY")
print("="*70)
for month, stocks in picks.items():
    if stocks:
        top = stocks[0]
        print(f"{month:12s}: {top['symbol']:8s} | Score: {top['probability_score']:5.1f} | Win Rate: {top['win_rate']:5.1f}% | Avg Return: {top['avg_return']:+6.2f}%")
    else:
        print(f"{month:12s}: No data")

print("\n✅ Total stocks with monthly data: {}")
print(f"✅ Data covers years: 2015 onwards")
