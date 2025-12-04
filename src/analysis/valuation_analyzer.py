"""
Advanced valuation analysis and stock screening
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


class ValuationAnalyzer:
    """Analyze stock valuations and identify opportunities"""
    
    def __init__(self):
        self.api = SarmayaAPI()
    
    def analyze_momentum_stocks(self):
        """Analyze stocks at 52-week highs (momentum strategy)"""
        
        high_stocks = self.api.get_52week_high_stocks(limit=50)
        
        if len(high_stocks) == 0:
            return None
        
        # Already has basic data, just add momentum score
        if 'changePercent' in high_stocks.columns:
            high_stocks['change_pct'] = pd.to_numeric(high_stocks['changePercent'], errors='coerce')
        if 'volume' in high_stocks.columns:
            high_stocks['volume'] = pd.to_numeric(high_stocks['volume'], errors='coerce')
        
        # Sort by strength
        high_stocks['momentum_score'] = high_stocks.get('change_pct', 0).fillna(0)
        high_stocks = high_stocks.sort_values('momentum_score', ascending=False)
        
        return high_stocks
    
    def analyze_value_stocks(self):
        """Analyze stocks at 52-week lows (value/contrarian strategy)"""
        
        low_stocks = self.api.get_52week_low_stocks(limit=50)
        
        if len(low_stocks) == 0:
            return None
        
        # Add basic metrics
        if 'peRatio' in low_stocks.columns:
            low_stocks['pe_ratio'] = pd.to_numeric(low_stocks['peRatio'], errors='coerce').fillna(999)
        if 'dividendYield' in low_stocks.columns:
            low_stocks['dividend_yield'] = pd.to_numeric(low_stocks['dividendYield'], errors='coerce').fillna(0)
        
        # Value score (lower P/E + higher dividend = better value)
        # Ensure columns exist
        if 'pe_ratio' not in low_stocks.columns:
            low_stocks['pe_ratio'] = 20
        if 'dividend_yield' not in low_stocks.columns:
            low_stocks['dividend_yield'] = 0
            
        low_stocks['value_score'] = (
            (20 - low_stocks['pe_ratio'].clip(0, 40)) / 20 * 0.6 +
            low_stocks['dividend_yield'] / 10 * 0.4
        )
        
        low_stocks = low_stocks.sort_values('value_score', ascending=False)
        
        return low_stocks
    
    def analyze_dividend_opportunities(self):
        """Find best dividend opportunities"""
        
        div_stocks = self.api.get_top_dividend_stocks(limit=50)
        
        if len(div_stocks) == 0:
            return None, None
        
        # Add dividend quality score
        if 'dividendYield' in div_stocks.columns:
            div_stocks['dividend_yield'] = pd.to_numeric(div_stocks['dividendYield'], errors='coerce').fillna(0)
        if 'marketCap' in div_stocks.columns:
            div_stocks['market_cap'] = pd.to_numeric(div_stocks['marketCap'], errors='coerce').fillna(0)
        
        # Dividend quality score
        max_cap = div_stocks['market_cap'].max() if 'market_cap' in div_stocks.columns else 1
        div_stocks['dividend_quality'] = (
            div_stocks.get('dividend_yield', 0) * 0.6 +
            (div_stocks.get('market_cap', 0) / max_cap * 10) * 0.4
        )
        
        div_stocks = div_stocks.sort_values('dividend_quality', ascending=False)
        
        return div_stocks, None
    
    def find_blue_chip_opportunities(self):
        """Find opportunities in large cap stocks (blue chips)"""
        
        large_caps = self.api.get_large_cap_stocks(limit=50)
        
        if len(large_caps) == 0:
            return None
        
        # Add quality score
        if 'dividendYield' in large_caps.columns:
            large_caps['dividend_yield'] = pd.to_numeric(large_caps['dividendYield'], errors='coerce').fillna(0)
        if 'peRatio' in large_caps.columns:
            large_caps['pe_ratio'] = pd.to_numeric(large_caps['peRatio'], errors='coerce').fillna(20)
        if 'marketCap' in large_caps.columns:
            large_caps['market_cap'] = pd.to_numeric(large_caps['marketCap'], errors='coerce').fillna(0)
        
        # Blue chip quality score
        max_cap = large_caps['market_cap'].max() if 'market_cap' in large_caps.columns else 1
        
        # Ensure columns exist
        if 'pe_ratio' not in large_caps.columns:
            large_caps['pe_ratio'] = 20
        if 'dividend_yield' not in large_caps.columns:
            large_caps['dividend_yield'] = 0
            
        large_caps['quality_score'] = (
            large_caps['dividend_yield'] / 10 * 0.3 +
            ((20 - large_caps['pe_ratio'].clip(0, 40)) / 20 * 10) * 0.4 +
            (large_caps.get('market_cap', 0) / max_cap * 10) * 0.3
        )
        
        large_caps = large_caps.sort_values('quality_score', ascending=False)
        
        return large_caps


def generate_valuation_report():
    """Generate comprehensive valuation report"""
    
    analyzer = ValuationAnalyzer()
    
    print("\n" + "="*70)
    print("VALUATION ANALYSIS REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Momentum Stocks
    print("\n🚀 MOMENTUM LEADERS (At 52-Week Highs)")
    print("-"*70)
    
    momentum = analyzer.analyze_momentum_stocks()
    
    if momentum is not None and len(momentum) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'Change %':<12} {'Momentum Score'}")
        print("-"*70)
        
        for _, row in momentum.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('close', 0):>7.2f}  "
                  f"{row.get('change_pct', 0):>+7.2f}%    "
                  f"{row.get('momentum_score', 0):>7.2f}")
        momentum.to_csv('reports/momentum_stocks.csv', index=False)
    else:
        print("No data available")
    
    # 2. Value Stocks
    print("\n💎 VALUE OPPORTUNITIES (At 52-Week Lows)")
    print("-"*70)
    
    value = analyzer.analyze_value_stocks()
    
    if value is not None and len(value) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'P/E':<8} {'Div Yield':<12} {'Value Score'}")
        print("-"*70)
        
        for _, row in value.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('close', 0):>7.2f}  "
                  f"{row.get('pe_ratio', 0):>5.1f}  "
                  f"{row.get('dividend_yield', 0):>6.2f}%     "
                  f"{row.get('value_score', 0):>7.2f}")
        value.to_csv('reports/value_stocks.csv', index=False)
    else:
        print("No data available")
    
    # 3. Dividend Opportunities
    print("\n💰 TOP DIVIDEND STOCKS")
    print("-"*70)
    
    div_stocks, _ = analyzer.analyze_dividend_opportunities()
    
    if div_stocks is not None and len(div_stocks) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'Div Yield':<12} {'Quality Score'}")
        print("-"*70)
        
        for _, row in div_stocks.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('close', 0):>7.2f}  "
                  f"{row.get('dividend_yield', 0):>6.2f}%     "
                  f"{row.get('dividend_quality', 0):>7.2f}")
        div_stocks.to_csv('reports/dividend_stocks.csv', index=False)
    else:
        print("No data available")
    
    # 4. Blue Chip Opportunities
    print("\n🏆 BLUE CHIP OPPORTUNITIES (Large Cap)")
    print("-"*70)
    
    blue_chips = analyzer.find_blue_chip_opportunities()
    
    if blue_chips is not None and len(blue_chips) > 0:
        print(f"{'Symbol':<8} {'Market Cap':<15} {'P/E':<8} {'Div Yield':<12} {'Quality'}")
        print("-"*70)
        
        for _, row in blue_chips.head(10).iterrows():
            market_cap_b = row.get('market_cap', 0) / 1e9
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"{market_cap_b:>7.2f}B       "
                  f"{row.get('pe_ratio', 0):>5.1f}  "
                  f"{row.get('dividend_yield', 0):>6.2f}%     "
                  f"{row.get('quality_score', 0):>7.2f}")
        blue_chips.to_csv('reports/blue_chip_stocks.csv', index=False)
    else:
        print("No data available")
    
    print("\n" + "="*70)
    print("✓ Valuation reports saved to reports/ directory")
    print("="*70 + "\n")


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    generate_valuation_report()