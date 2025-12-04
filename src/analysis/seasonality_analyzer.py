"""
Enhanced Seasonality Analysis Module
Analyzes historical stock data to identify recurring seasonal patterns with probability scoring.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
from scipy.stats import binomtest

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class SeasonalityAnalyzer:
    def __init__(self, min_year=2015):
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = project_root / "data" / "raw" / "historical"
        self.reports_dir = project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        self.min_year = min_year  # Only analyze data from this year onwards
        
    def load_stock_data(self, symbol):
        """Load historical data for a specific stock"""
        file_path = self.data_dir / f"{symbol}.csv"
        if not file_path.exists():
            return None
            
        try:
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Handle date column
            if 'time' in df.columns:
                df['date'] = pd.to_datetime(df['time'], errors='coerce')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                return None
            
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
                
            df = df.sort_values('date')
            
            # Filter data from min_year onwards
            df = df[df['date'].dt.year >= self.min_year]
            
            if len(df) == 0:
                return None
            
            return df
        except Exception as e:
            # Silently skip problematic files
            return None

    def calculate_probability_score(self, win_rate, avg_return, years_analyzed, return_std):
        """
        Calculate probability score (0-100) based on multiple factors
        
        Formula:
        - win_rate: 40% weight
        - normalized_return: 30% weight
        - years_weight: 20% weight
        - consistency: 10% weight
        """
        # Win rate component (0-100)
        win_rate_score = win_rate
        
        # Return magnitude component (normalize to 0-100)
        # Assume returns between -20% to +20% are typical
        return_score = min(100, max(0, (avg_return + 20) / 40 * 100))
        
        # Years component (more years = higher confidence)
        # Cap at 10 years for 100% score
        years_score = min(100, (years_analyzed / 10) * 100)
        
        # Consistency component (lower std = higher score)
        # Assume std of 0-15% is typical range
        if return_std > 0:
            consistency_score = max(0, 100 - (return_std / 15 * 100))
        else:
            consistency_score = 100
            
        # Weighted average
        probability_score = (
            win_rate_score * 0.4 +
            return_score * 0.3 +
            years_score * 0.2 +
            consistency_score * 0.1
        )
        
        return round(probability_score, 1)
    
    def calculate_statistical_significance(self, win_count, total_count):
        """
        Use binomial test to determine if win rate is statistically significant
        H0: win_rate = 50% (random)
        """
        if total_count < 3:
            return False, 1.0
            
        # Two-tailed binomial test
        result = binomtest(win_count, total_count, 0.5, alternative='two-sided')
        p_value = result.pvalue
        is_significant = p_value < 0.05
        
        return is_significant, p_value

    def analyze_stock_seasonality(self, symbol, df):
        """Analyze seasonality for a single stock with enhanced metrics"""
        try:
            if df is None or len(df) < 180:  # Lowered to 6 months minimum
                return None, None
                
            # Add month and year columns
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['month_name'] = df['date'].dt.strftime('%B')
            
            # Calculate monthly returns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            monthly_df = df.set_index('date')[numeric_cols].resample('ME').last()
            
            # Calculate return based on close price
            monthly_df.loc[:, 'return'] = monthly_df['close'].pct_change(fill_method=None) * 100
            
            # Drop the first row (NaN return)
            monthly_df = monthly_df.dropna(subset=['return'])
            
            # Add back month/year after resampling
            monthly_df['month'] = monthly_df.index.month
            monthly_df['month_name'] = monthly_df.index.strftime('%B')
            monthly_df['year'] = monthly_df.index.year
            
            # Calculate stats per month
            monthly_stats = []
            
            for month in range(1, 13):
                month_data = monthly_df[monthly_df['month'] == month]
                
                if len(month_data) < 1:
                    continue
                    
                win_count = len(month_data[month_data['return'] > 0])
                total_count = len(month_data)
                win_rate = (win_count / total_count) * 100
                avg_return = month_data['return'].mean()
                return_std = month_data['return'].std()
                
                # Calculate probability score
                probability_score = self.calculate_probability_score(
                    win_rate, avg_return, total_count, return_std
                )
                
                # Statistical significance
                is_significant, p_value = self.calculate_statistical_significance(
                    win_count, total_count
                )
                
                # Confidence level
                if is_significant and total_count >= 5 and probability_score >= 75:
                    confidence = 'HIGH'
                elif total_count >= 3 and probability_score >= 60:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'
                
                monthly_stats.append({
                    'month': month,
                    'month_name': datetime(2000, month, 1).strftime('%B'),
                    'win_rate': round(win_rate, 1),
                    'avg_return': round(avg_return, 2),
                    'return_std': round(return_std, 2),
                    'years_analyzed': total_count,
                    'positive_years': win_count,
                    'negative_years': total_count - win_count,
                    'probability_score': probability_score,
                    'confidence': confidence,
                    'is_significant': bool(is_significant),  # Convert to Python bool
                    'p_value': round(p_value, 4),
                    'historical_returns': [round(r, 2) for r in month_data['return'].tolist()]
                })
                
            return monthly_stats, monthly_df
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None, None

    def generate_monthly_top_picks(self, detailed_stats):
        """Generate top picks for each month based on probability scores"""
        monthly_picks = {}
        
        for month_num in range(1, 13):
            month_name = datetime(2000, month_num, 1).strftime('%B')
            month_stocks = []
            
            for symbol, stats_list in detailed_stats.items():
                for stat in stats_list:
                    if stat['month'] == month_num:
                        month_stocks.append({
                            'symbol': symbol,
                            'win_rate': stat['win_rate'],
                            'avg_return': stat['avg_return'],
                            'probability_score': stat['probability_score'],
                            'years_analyzed': stat['years_analyzed'],
                            'confidence': stat['confidence'],
                            'historical_returns': stat['historical_returns'],
                            'return_std': stat['return_std']
                        })
            
            # Sort by probability score and take top 20
            month_stocks.sort(key=lambda x: x['probability_score'], reverse=True)
            monthly_picks[month_name] = month_stocks[:20]
        
        return monthly_picks

    def run_analysis(self):
        """Run enhanced seasonality analysis on all stocks"""
        print("\n" + "="*70)
        print("🔍 ENHANCED SEASONALITY ANALYSIS")
        print("="*70)
        print(f"📅 Analyzing data from {self.min_year} onwards")
        
        all_files = list(self.data_dir.glob("*.csv"))
        print(f"📊 Found {len(all_files)} stock files")
        
        summary_patterns = []
        detailed_stats = {}
        stocks_processed = 0
        stocks_with_data = 0
        
        for file_path in all_files:
            symbol = file_path.stem.upper()
            
            # Skip temporary files
            if symbol.startswith('TEMP_'):
                continue
            
            stocks_processed += 1
            if stocks_processed % 50 == 0:
                print(f"  Processed {stocks_processed}/{len(all_files)} stocks...")
                
            df = self.load_stock_data(symbol)
            if df is None or len(df) < 180:
                continue
                
            stats, monthly_df = self.analyze_stock_seasonality(symbol, df)
            
            if stats:
                stocks_with_data += 1
                detailed_stats[symbol] = stats
                
                # Identify strong patterns for summary
                for stat in stats:
                    # Strong Bullish Pattern: Win Rate >= 70% OR High Probability Score
                    if (stat['win_rate'] >= 70 or stat['probability_score'] >= 75) and stat['years_analyzed'] >= 2:
                        summary_patterns.append({
                            'symbol': symbol,
                            'pattern_type': 'Bullish',
                            'month': stat['month_name'],
                            'win_rate': stat['win_rate'],
                            'avg_return': stat['avg_return'],
                            'probability_score': stat['probability_score'],
                            'years': stat['years_analyzed'],
                            'confidence': stat['confidence']
                        })
                    
                    # Strong Bearish Pattern
                    elif (stat['win_rate'] <= 30 or stat['probability_score'] <= 25) and stat['years_analyzed'] >= 2:
                        summary_patterns.append({
                            'symbol': symbol,
                            'pattern_type': 'Bearish',
                            'month': stat['month_name'],
                            'win_rate': stat['win_rate'],
                            'avg_return': stat['avg_return'],
                            'probability_score': stat['probability_score'],
                            'years': stat['years_analyzed'],
                            'confidence': stat['confidence']
                        })

        print(f"\n✅ Analysis Complete!")
        print(f"   Stocks processed: {stocks_processed}")
        print(f"   Stocks with sufficient data: {stocks_with_data}")

        # Save Summary Report
        if summary_patterns:
            summary_df = pd.DataFrame(summary_patterns)
            summary_df = summary_df.sort_values(['probability_score', 'win_rate'], ascending=False)
            summary_path = self.reports_dir / "seasonality_analysis.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n📄 Saved summary: {summary_path}")
            print(f"   Found {len(summary_patterns)} significant seasonal patterns")
        else:
            print("\n⚠️  No significant seasonal patterns found")

        # Save Detailed Stats (JSON)
        json_path = self.reports_dir / "monthly_stats.json"
        with open(json_path, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        print(f"📄 Saved detailed stats: {json_path}")
        print(f"   {len(detailed_stats)} stocks included")
        
        # Generate and save monthly top picks
        print(f"\n🎯 Generating monthly top picks...")
        monthly_picks = self.generate_monthly_top_picks(detailed_stats)
        picks_path = self.reports_dir / "monthly_top_picks.json"
        with open(picks_path, 'w') as f:
            json.dump(monthly_picks, f, indent=2)
        print(f"📄 Saved monthly top picks: {picks_path}")
        
        # Print summary of top picks
        print(f"\n📊 Monthly Top Picks Summary:")
        for month, picks in monthly_picks.items():
            if picks:
                top_pick = picks[0]
                print(f"   {month:12s}: {top_pick['symbol']:8s} (Score: {top_pick['probability_score']:.1f}, Win Rate: {top_pick['win_rate']:.1f}%)")
        
        print("\n" + "="*70)

def main():
    analyzer = SeasonalityAnalyzer(min_year=2015)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
