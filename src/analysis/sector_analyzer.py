"""
Sector-based analysis and predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


class SectorAnalyzer:
    """Analyze sectors and make sector-rotation predictions"""
    
    def __init__(self):
        self.api = SarmayaAPI()
    
    def get_sector_momentum(self):
        """Calculate sector momentum scores"""
        
        sectors_df = self.api.get_sector_performance()
        
        if sectors_df is None:
            return None
        
        # Calculate momentum score
        sectors_df['momentum_score'] = (
            sectors_df['change_pct'] * 0.4 +  # Recent performance
            (sectors_df['volume'] / sectors_df['volume'].max() * 100) * 0.3 +  # Volume strength
            (sectors_df['market_cap'] / sectors_df['market_cap'].max() * 100) * 0.3  # Size factor
        )
        
        # Rank sectors
        sectors_df = sectors_df.sort_values('momentum_score', ascending=False)
        sectors_df['rank'] = range(1, len(sectors_df) + 1)
        
        return sectors_df
    
    def get_sector_rotation_signals(self):
        """Identify sector rotation opportunities"""
        
        sectors_df = self.get_sector_momentum()
        
        if sectors_df is None:
            return None
        
        # Strong sectors (top 25%)
        top_quartile = int(len(sectors_df) * 0.25)
        strong_sectors = sectors_df.head(top_quartile)
        
        # Weak sectors (bottom 25%)
        weak_sectors = sectors_df.tail(top_quartile)
        
        # Turnaround candidates (weak but showing signs of life)
        turnaround = weak_sectors[weak_sectors['change_pct'] > 0]
        
        return {
            'strong_sectors': strong_sectors,
            'weak_sectors': weak_sectors,
            'turnaround_candidates': turnaround
        }
    
    def get_sector_leaders(self, sector_name, top_n=5):
        """Get top performing stocks in a sector"""
        
        # Get stocks in sector
        kse100 = self.api.get_kse100_tickers()
        
        sector_stocks = []
        for symbol in kse100:
            stock_sector = self.api.get_stock_sector(symbol)
            if stock_sector and sector_name.lower() in stock_sector.lower():
                info = self.api.get_stock_info(symbol)
                if info:
                    sector_stocks.append({
                        'symbol': symbol,
                        'price': info.get('currentPrice', 0),
                        'change_pct': info.get('changePct', 0),
                        'volume': info.get('volume', 0)
                    })
        
        if not sector_stocks:
            return None
        
        df = pd.DataFrame(sector_stocks)
        df = df.sort_values('change_pct', ascending=False)
        
        return df.head(top_n)


def generate_sector_report():
    """Generate comprehensive sector analysis report"""
    
    analyzer = SectorAnalyzer()
    
    print("\n" + "="*70)
    print("SECTOR ANALYSIS REPORT")
    print("="*70)
    
    # Get sector momentum
    sectors = analyzer.get_sector_momentum()
    
    if sectors is None:
        print("✗ Unable to fetch sector data")
        return
    
    # Top performing sectors
    print("\n🟢 TOP PERFORMING SECTORS")
    print("-"*70)
    print(f"{'Rank':<6} {'Sector':<35} {'Change %':<12} {'Momentum'}")
    print("-"*70)
    
    for _, row in sectors.head(10).iterrows():
        print(f"{row['rank']:<6} {row['sector'][:34]:<35} {row['change_pct']:>+7.2f}%    {row['momentum_score']:>6.2f}")
    
    # Bottom performing sectors
    print("\n🔴 UNDERPERFORMING SECTORS")
    print("-"*70)
    print(f"{'Rank':<6} {'Sector':<35} {'Change %':<12} {'Momentum'}")
    print("-"*70)
    
    for _, row in sectors.tail(10).iterrows():
        print(f"{row['rank']:<6} {row['sector'][:34]:<35} {row['change_pct']:>+7.2f}%    {row['momentum_score']:>6.2f}")
    
    # Rotation signals
    rotation = analyzer.get_sector_rotation_signals()
    
    if rotation and len(rotation['turnaround_candidates']) > 0:
        print("\n🔄 SECTOR ROTATION CANDIDATES (Oversold with recent bounce)")
        print("-"*70)
        print(f"{'Sector':<35} {'Change %':<12} {'Market Cap'}")
        print("-"*70)
        
        for _, row in rotation['turnaround_candidates'].iterrows():
            market_cap_b = row['market_cap'] / 1e9
            print(f"{row['sector'][:34]:<35} {row['change_pct']:>+7.2f}%    {market_cap_b:>7.2f}B")
    
    print("\n" + "="*70)
    
    # Save report
    sectors.to_csv('reports/sector_analysis.csv', index=False)
    print("✓ Sector report saved to: reports/sector_analysis.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    generate_sector_report()