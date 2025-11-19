"""
Portfolio optimization using Modern Portfolio Theory
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.sarmaaya_api import SarmayaAPI


class PortfolioOptimizer:
    """Optimize portfolio allocation using MPT"""
    
    def __init__(self):
        self.api = SarmayaAPI()
    
    def get_returns_data(self, symbols, days=252):
        """Get historical returns for symbols"""
        
        returns_dict = {}
        
        for symbol in symbols:
            df = self.api.get_price_history(symbol, days=days)
            
            if df is not None and len(df) > 1:
                df = df.sort_values('date')
                df['returns'] = df['close'].pct_change()
                returns_dict[symbol] = df['returns'].dropna()
        
        if not returns_dict:
            return None
        
        # Combine into single DataFrame
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def calculate_portfolio_metrics(self, weights, returns_df):
        """Calculate portfolio return and risk"""
        
        # Expected annual return
        expected_return = np.sum(returns_df.mean() * weights) * 252
        
        # Portfolio variance
        cov_matrix = returns_df.cov() * 252
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = expected_return / portfolio_std if portfolio_std > 0 else 0
        
        return expected_return, portfolio_std, sharpe_ratio
    
    def optimize_max_sharpe(self, returns_df):
        """Optimize for maximum Sharpe ratio"""
        
        n_assets = len(returns_df.columns)
        
        # Objective function (negative Sharpe to minimize)
        def neg_sharpe(weights):
            _, _, sharpe = self.calculate_portfolio_metrics(weights, returns_df)
            return -sharpe
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        )
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def optimize_min_risk(self, returns_df, target_return=None):
        """Optimize for minimum risk (volatility)"""
        
        n_assets = len(returns_df.columns)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            cov_matrix = returns_df.cov() * 252
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Add target return constraint if specified
        if target_return:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.sum(returns_df.mean() * x) * 252 - target_return
            })
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def create_efficient_frontier(self, returns_df, n_portfolios=50):
        """Generate efficient frontier"""
        
        returns_range = np.linspace(
            returns_df.mean().min() * 252,
            returns_df.mean().max() * 252,
            n_portfolios
        )
        
        efficient_portfolios = []
        
        for target_return in returns_range:
            try:
                weights = self.optimize_min_risk(returns_df, target_return)
                ret, risk, sharpe = self.calculate_portfolio_metrics(weights, returns_df)
                
                efficient_portfolios.append({
                    'return': ret,
                    'risk': risk,
                    'sharpe': sharpe,
                    'weights': weights
                })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def recommend_portfolio(self, symbols, investment_amount=100000, strategy='max_sharpe'):
        """
        Recommend optimal portfolio allocation
        
        strategy: 'max_sharpe', 'min_risk', 'balanced'
        """
        
        print(f"\n{'='*70}")
        print(f"PORTFOLIO OPTIMIZATION - {strategy.upper()}")
        print(f"{'='*70}")
        print(f"Investment Amount: PKR {investment_amount:,.0f}")
        print(f"Number of Stocks: {len(symbols)}\n")
        
        # Get returns data
        print("Fetching historical data...")
        returns_df = self.get_returns_data(symbols, days=252)
        
        if returns_df is None or len(returns_df) < 30:
            print("✗ Insufficient data for optimization")
            return None
        
        print(f"✓ Got {len(returns_df)} days of data\n")
        
        # Optimize based on strategy
        if strategy == 'max_sharpe':
            weights = self.optimize_max_sharpe(returns_df)
        elif strategy == 'min_risk':
            weights = self.optimize_min_risk(returns_df)
        elif strategy == 'balanced':
            # Combination of max sharpe and min risk
            weights_sharpe = self.optimize_max_sharpe(returns_df)
            weights_risk = self.optimize_min_risk(returns_df)
            weights = (weights_sharpe + weights_risk) / 2
            weights = weights / weights.sum()  # Renormalize
        else:
            weights = self.optimize_max_sharpe(returns_df)
        
        # Calculate metrics
        exp_return, risk, sharpe = self.calculate_portfolio_metrics(weights, returns_df)
        
        # Create allocation DataFrame
        allocation = pd.DataFrame({
            'symbol': returns_df.columns,
            'weight': weights,
            'allocation_pct': weights * 100,
            'investment': weights * investment_amount
        })
        
        allocation = allocation[allocation['weight'] > 0.01].sort_values('weight', ascending=False)
        
        # Print results
        print("="*70)
        print("PORTFOLIO METRICS")
        print("="*70)
        print(f"Expected Annual Return:  {exp_return:>6.2%}")
        print(f"Expected Annual Risk:    {risk:>6.2%}")
        print(f"Sharpe Ratio:            {sharpe:>6.3f}")
        
        print("\n" + "="*70)
        print("RECOMMENDED ALLOCATION")
        print("="*70)
        print(f"{'Symbol':<8} {'Weight':<10} {'%':<10} {'Investment (PKR)'}")
        print("-"*70)
        
        for _, row in allocation.iterrows():
            print(f"{row['symbol']:<8} {row['weight']:<10.4f} {row['allocation_pct']:<10.2f} {row['investment']:>15,.0f}")
        
        print("-"*70)
        print(f"{'TOTAL':<8} {'1.0000':<10} {'100.00':<10} {investment_amount:>15,.0f}")
        print("="*70 + "\n")
        
        # Save to CSV
        allocation.to_csv('reports/portfolio_allocation.csv', index=False)
        print("✓ Portfolio saved to: reports/portfolio_allocation.csv\n")
        
        return allocation


def generate_portfolio_recommendations():
    """Generate portfolio recommendations for different strategies"""
    
    optimizer = PortfolioOptimizer()
    
    # Top liquid stocks
    symbols = ['HBL', 'OGDC', 'PPL', 'ENGRO', 'LUCK', 'MCB', 'UBL', 'PSO', 'HUBC', 'FFC',
               'MEBL', 'BAFL', 'NRL', 'SSGC', 'MARI', 'POL']
    
    investment_amount = 500000  # PKR 500,000
    
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION - MULTIPLE STRATEGIES")
    print("="*70 + "\n")
    
    # Strategy 1: Maximum Sharpe Ratio (Aggressive)
    print("\n🎯 STRATEGY 1: MAXIMUM SHARPE RATIO (Aggressive Growth)")
    optimizer.recommend_portfolio(symbols, investment_amount, strategy='max_sharpe')
    
    # Strategy 2: Minimum Risk (Conservative)
    print("\n🛡️  STRATEGY 2: MINIMUM RISK (Conservative)")
    optimizer.recommend_portfolio(symbols, investment_amount, strategy='min_risk')
    
    # Strategy 3: Balanced
    print("\n⚖️  STRATEGY 3: BALANCED")
    optimizer.recommend_portfolio(symbols, investment_amount, strategy='balanced')


if __name__ == "__main__":
    # Install scipy if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run(['pip', 'install', 'scipy', '--break-system-packages'])
    
    generate_portfolio_recommendations()