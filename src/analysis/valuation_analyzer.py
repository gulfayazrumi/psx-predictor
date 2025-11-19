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
        
        # Enrich with additional data
        for idx, row in high_stocks.iterrows():
            symbol = row.get('symbol')
            info = self.api.get_stock_info(symbol)
            
            if info:
                high_stocks.loc[idx, 'volume'] = info.get('volume', 0)
                high_stocks.loc[idx, 'change_pct'] = info.get('changePct', 0)
                high_stocks.loc[idx, 'market_cap'] = info.get('marketCap', 0)
        
        # Sort by strength
        high_stocks['momentum_score'] = (
            high_stocks.get('change_pct', 0) * 0.5 +
            (high_stocks.get('volume', 0) / high_stocks.get('volume', 1).max() * 10) * 0.5
        )
        
        high_stocks = high_stocks.sort_values('momentum_score', ascending=False)
        
        return high_stocks
    
    def analyze_value_stocks(self):
        """Analyze stocks at 52-week lows (value/contrarian strategy)"""
        
        low_stocks = self.api.get_52week_low_stocks(limit=50)
        
        if len(low_stocks) == 0:
            return None
        
        # Enrich with fundamentals
        for idx, row in low_stocks.iterrows():
            symbol = row.get('symbol')
            info = self.api.get_stock_info(symbol)
            about = self.api.get_stock_about(symbol)
            
            if info:
                low_stocks.loc[idx, 'pe_ratio'] = info.get('peRatio', 999)
                low_stocks.loc[idx, 'dividend_yield'] = info.get('dividendYield', 0)
                low_stocks.loc[idx, 'market_cap'] = info.get('marketCap', 0)
            
            if about:
                low_stocks.loc[idx, 'sector'] = about.get('sector', 'Unknown')
        
        # Value score (lower P/E + higher dividend = better value)
        low_stocks['value_score'] = (
            (20 - low_stocks.get('pe_ratio', 20).clip(0, 40)) / 20 * 0.6 +
            low_stocks.get('dividend_yield', 0) / 10 * 0.4
        )
        
        low_stocks = low_stocks.sort_values('value_score', ascending=False)
        
        return low_stocks
    
    def analyze_dividend_opportunities(self):
        """Find best dividend opportunities"""
        
        div_stocks = self.api.get_top_dividend_stocks(limit=50)
        div_announcements = self.api.get_dividend_announcements(limit=100)
        
        if len(div_stocks) == 0:
            return None, None
        
        # Enrich dividend stocks
        for idx, row in div_stocks.iterrows():
            symbol = row.get('symbol')
            info = self.api.get_stock_info(symbol)
            
            if info:
                div_stocks.loc[idx, 'price'] = info.get('currentPrice', 0)
                div_stocks.loc[idx, 'market_cap'] = info.get('marketCap', 0)
                div_stocks.loc[idx, 'pe_ratio'] = info.get('peRatio', 0)
        
        # Dividend quality score
        div_stocks['dividend_quality'] = (
            div_stocks.get('dividendYield', 0) * 0.4 +
            (div_stocks.get('market_cap', 0) / div_stocks.get('market_cap', 1).max() * 10) * 0.3 +
            ((20 - div_stocks.get('pe_ratio', 20).clip(0, 40)) / 20 * 10) * 0.3
        )
        
        div_stocks = div_stocks.sort_values('dividend_quality', ascending=False)
        
        return div_stocks, div_announcements
    
    def find_blue_chip_opportunities(self):
        """Find opportunities in large cap stocks (blue chips)"""
        
        large_caps = self.api.get_large_cap_stocks(limit=50)
        
        if len(large_caps) == 0:
            return None
        
        # Enrich with metrics
        for idx, row in large_caps.iterrows():
            symbol = row.get('symbol')
            info = self.api.get_stock_info(symbol)
            
            if info:
                large_caps.loc[idx, 'dividend_yield'] = info.get('dividendYield', 0)
                large_caps.loc[idx, 'pe_ratio'] = info.get('peRatio', 0)
                large_caps.loc[idx, 'change_pct'] = info.get('changePct', 0)
                large_caps.loc[idx, 'volume'] = info.get('volume', 0)
        
        # Blue chip quality score
        large_caps['quality_score'] = (
            large_caps.get('dividend_yield', 0) / 10 * 0.3 +
            ((20 - large_caps.get('pe_ratio', 20).clip(0, 40)) / 20 * 10) * 0.4 +
            (large_caps.get('market_cap', 0) / large_caps.get('market_cap', 1).max() * 10) * 0.3
        )
        
        large_caps = large_caps.sort_values('quality_score', ascending=False)
        
        return large_caps
    
    def create_custom_screen(self, strategy='balanced'):
        """
        Create custom stock screens based on strategy
        
        Strategies:
        - 'growth': High momentum, near 52-week highs
        - 'value': Low P/E, low P/B, near 52-week lows
        - 'income': High dividend yield, stable
        - 'balanced': Mix of all factors
        - 'quality': Large cap, good fundamentals
        """
        
        if strategy == 'growth':
            criteria = {
                'near_52week_high': True,
                'min_market_cap': 5_000_000_000,  # 5B minimum
            }
        
        elif strategy == 'value':
            criteria = {
                'max_pe': 12,
                'max_pb': 1.5,
                'min_dividend_yield': 3.0
            }
        
        elif strategy == 'income':
            criteria = {
                'min_dividend_yield': 6.0,
                'min_market_cap': 10_000_000_000,  # Large cap for stability
            }
        
        elif strategy == 'quality':
            criteria = {
                'min_market_cap': 20_000_000_000,  # 20B+
                'max_pe': 20,
                'min_dividend_yield': 2.0
            }
        
        else:  # balanced
            criteria = {
                'max_pe': 15,
                'min_dividend_yield': 3.0,
                'min_market_cap': 10_000_000_000
            }
        
        screened = self.api.screen_stocks(criteria)
        
        return screened


def generate_valuation_report():
    """Generate comprehensive valuation report"""
    
    analyzer = ValuationAnalyzer()
    
    print("\n" + "="*70)
    print("VALUATION ANALYSIS REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Momentum Stocks (52-week highs)
    print("\n🚀 MOMENTUM LEADERS (At 52-Week Highs)")
    print("-"*70)
    
    momentum = analyzer.analyze_momentum_stocks()
    
    if momentum is not None and len(momentum) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'Change %':<12} {'Momentum Score'}")
        print("-"*70)
        
        for _, row in momentum.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('currentPrice', 0):>7.2f}  "
                  f"{row.get('change_pct', 0):>+7.2f}%    "
                  f"{row.get('momentum_score', 0):>7.2f}")
    else:
        print("No data available")
    
    # 2. Value Stocks (52-week lows)
    print("\n💎 VALUE OPPORTUNITIES (At 52-Week Lows)")
    print("-"*70)
    
    value = analyzer.analyze_value_stocks()
    
    if value is not None and len(value) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'P/E':<8} {'Div Yield':<12} {'Value Score'}")
        print("-"*70)
        
        for _, row in value.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('currentPrice', 0):>7.2f}  "
                  f"{row.get('pe_ratio', 0):>5.1f}  "
                  f"{row.get('dividend_yield', 0):>6.2f}%     "
                  f"{row.get('value_score', 0):>7.2f}")
    else:
        print("No data available")
    
    # 3. Dividend Opportunities
    print("\n💰 TOP DIVIDEND STOCKS")
    print("-"*70)
    
    div_stocks, div_announcements = analyzer.analyze_dividend_opportunities()
    
    if div_stocks is not None and len(div_stocks) > 0:
        print(f"{'Symbol':<8} {'Price':<12} {'Div Yield':<12} {'Quality Score'}")
        print("-"*70)
        
        for _, row in div_stocks.head(10).iterrows():
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"PKR {row.get('price', 0):>7.2f}  "
                  f"{row.get('dividendYield', 0):>6.2f}%     "
                  f"{row.get('dividend_quality', 0):>7.2f}")
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
            market_cap_b = row.get('marketCap', 0) / 1e9
            print(f"{row.get('symbol', 'N/A'):<8} "
                  f"{market_cap_b:>7.2f}B       "
                  f"{row.get('pe_ratio', 0):>5.1f}  "
                  f"{row.get('dividend_yield', 0):>6.2f}%     "
                  f"{row.get('quality_score', 0):>7.2f}")
    else:
        print("No data available")
    
    # 5. Custom Screens
    print("\n" + "="*70)
    print("STRATEGY-BASED STOCK SCREENS")
    print("="*70)
    
    strategies = ['growth', 'value', 'income', 'quality']
    
    for strategy in strategies:
        print(f"\n📊 {strategy.upper()} STRATEGY")
        print("-"*70)
        
        screened = analyzer.create_custom_screen(strategy)
        
        if screened is not None and len(screened) > 0:
            print(f"Found {len(screened)} stocks matching criteria")
            print(f"\nTop 5 {strategy} picks:")
            print(f"{'Symbol':<8} {'Price':<12} {'P/E':<8} {'Div Yield'}")
            print("-"*50)
            
            for _, row in screened.head(5).iterrows():
                print(f"{row.get('symbol', 'N/A'):<8} "
                      f"PKR {row.get('price', 0):>7.2f}  "
                      f"{row.get('pe_ratio', 0):>5.1f}  "
                      f"{row.get('dividend_yield', 0):>6.2f}%")
        else:
            print("No stocks match criteria")
    
    print("\n" + "="*70)
    
    # Save reports
    Path("reports").mkdir(exist_ok=True)
    
    if momentum is not None:
        momentum.to_csv('reports/momentum_stocks.csv', index=False)
    if value is not None:
        value.to_csv('reports/value_stocks.csv', index=False)
    if div_stocks is not None:
        div_stocks.to_csv('reports/dividend_stocks.csv', index=False)
    if blue_chips is not None:
        blue_chips.to_csv('reports/blue_chip_stocks.csv', index=False)
    
    print("✓ Valuation reports saved to reports/ directory")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_valuation_report()