"""
Master analyzer that runs all analysis modules
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI
from src.analysis.sector_analyzer import SectorAnalyzer
from src.analysis.valuation_analyzer import ValuationAnalyzer


def run_complete_market_analysis():
    """Run comprehensive market analysis"""
    
    print("\n" + "="*70)
    print("COMPLETE PSX MARKET ANALYSIS")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize
    api = SarmayaAPI()
    sector_analyzer = SectorAnalyzer()
    valuation_analyzer = ValuationAnalyzer()
    
    reports = []
    
    # 1. Market Overview
    print("\n[1/5] Market Overview")
    print("-"*70)
    
    market = api.get_market_view()
    if market:
        kse100 = market.get('kse100', {})
        print(f"KSE-100:        {kse100.get('value', 'N/A'):>10}")
        print(f"Change:         {kse100.get('change', 0):>+10.2f} ({kse100.get('changePct', 0):>+6.2f}%)")
        print(f"Gainers:        {market.get('gainers', 0):>10}")
        print(f"Losers:         {market.get('losers', 0):>10}")
        print(f"Unchanged:      {market.get('unchanged', 0):>10}")
        reports.append("✓ Market overview collected")
    else:
        reports.append("✗ Market overview failed")
    
    # 2. Sector Analysis
    print("\n[2/5] Sector Analysis")
    print("-"*70)
    
    try:
        sectors = sector_analyzer.get_sector_momentum()
        if sectors is not None and len(sectors) > 0:
            print(f"\nTop 3 Sectors:")
            for _, row in sectors.head(3).iterrows():
                print(f"  {row['sector'][:30]:<30} {row['change_pct']:>+7.2f}%")
            
            print(f"\nBottom 3 Sectors:")
            for _, row in sectors.tail(3).iterrows():
                print(f"  {row['sector'][:30]:<30} {row['change_pct']:>+7.2f}%")
            
            sectors.to_csv('reports/sector_analysis.csv', index=False)
            reports.append("✓ Sector analysis saved")
        else:
            reports.append("✗ Sector analysis failed")
    except Exception as e:
        print(f"Error: {e}")
        reports.append("✗ Sector analysis error")
    
    # 3. Momentum Stocks
    print("\n[3/5] Momentum Stocks (52-Week Highs)")
    print("-"*70)
    
    try:
        momentum = valuation_analyzer.analyze_momentum_stocks()
        if momentum is not None and len(momentum) > 0:
            print(f"\nTop 5 Momentum Leaders:")
            for _, row in momentum.head(5).iterrows():
                print(f"  {row.get('symbol', 'N/A'):<8} PKR {row.get('currentPrice', 0):>7.2f}  {row.get('change_pct', 0):>+6.2f}%")
            
            momentum.to_csv('reports/momentum_stocks.csv', index=False)
            reports.append("✓ Momentum analysis saved")
        else:
            reports.append("✗ Momentum analysis failed")
    except Exception as e:
        print(f"Error: {e}")
        reports.append("✗ Momentum analysis error")
    
    # 4. Value Opportunities
    print("\n[4/5] Value Opportunities (52-Week Lows)")
    print("-"*70)
    
    try:
        value = valuation_analyzer.analyze_value_stocks()
        if value is not None and len(value) > 0:
            print(f"\nTop 5 Value Picks:")
            for _, row in value.head(5).iterrows():
                print(f"  {row.get('symbol', 'N/A'):<8} PKR {row.get('currentPrice', 0):>7.2f}  P/E: {row.get('pe_ratio', 0):>5.1f}")
            
            value.to_csv('reports/value_stocks.csv', index=False)
            reports.append("✓ Value analysis saved")
        else:
            reports.append("✗ Value analysis failed")
    except Exception as e:
        print(f"Error: {e}")
        reports.append("✗ Value analysis error")
    
    # 5. Dividend Opportunities
    print("\n[5/5] Dividend Opportunities")
    print("-"*70)
    
    try:
        div_stocks, _ = valuation_analyzer.analyze_dividend_opportunities()
        if div_stocks is not None and len(div_stocks) > 0:
            print(f"\nTop 5 Dividend Stocks:")
            for _, row in div_stocks.head(5).iterrows():
                print(f"  {row.get('symbol', 'N/A'):<8} Yield: {row.get('dividendYield', 0):>5.2f}%")
            
            div_stocks.to_csv('reports/dividend_stocks.csv', index=False)
            reports.append("✓ Dividend analysis saved")
        else:
            reports.append("✗ Dividend analysis failed")
    except Exception as e:
        print(f"Error: {e}")
        reports.append("✗ Dividend analysis error")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    for report in reports:
        print(report)
    
    print("\n✅ Analysis complete!")
    print("Reports saved in: reports/")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    run_complete_market_analysis()