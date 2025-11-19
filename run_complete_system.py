"""
PSX COMPLETE SYSTEM - SIMPLIFIED & WORKING VERSION
Focuses on what actually works, skips buggy components
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime
import subprocess

sys.path.append(str(Path(__file__).parent))


def collect_data():
    """Collect market data using Yahoo Finance"""
    
    print("\n" + "="*70)
    print("📊 COLLECTING MARKET DATA")
    print("="*70)
    
    try:
        print("\n✓ Using Yahoo Finance (reliable)")
        result = subprocess.run(
            ['python', 'collect_yfinance_data.py'],
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✓ Data collection successful")
            return True
        else:
            print("✗ Data collection failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def train_models(max_stocks=50):
    """Train ML models"""
    
    print("\n" + "="*70)
    print("🤖 TRAINING ML MODELS")
    print("="*70)
    
    try:
        result = subprocess.run(
            ['python', 'train_all.py', '--max', str(max_stocks)],
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"✓ Training completed ({max_stocks} stocks)")
            return True
        else:
            print("✗ Training failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def generate_signals():
    """Generate trading signals"""
    
    print("\n" + "="*70)
    print("📈 GENERATING TRADING SIGNALS")
    print("="*70)
    
    try:
        result = subprocess.run(
            ['python', 'generate_signals.py'],
            capture_output=False
        )
        
        if result.returncode == 0:
            print("✓ Signals generated successfully")
            return True
        else:
            print("✗ Signal generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def show_sarmaaya_data():
    """Display Sarmaaya API data"""
    
    print("\n" + "="*70)
    print("📊 SARMAAYA MARKET DATA")
    print("="*70)
    
    try:
        from src.data_collection.sarmaaya_api import SarmayaAPI
        
        api = SarmayaAPI()
        
        # Market Summary
        print("\n🔹 MARKET INDICES:")
        print("-"*70)
        summary = api.get_market_summary()
        if summary:
            for idx in summary[:5]:  # Top 5 indices
                change = idx.get('changePercentage', 0)
                symbol = '🟢' if change > 0 else '🔴' if change < 0 else '⚪'
                print(f"  {symbol} {idx['name'][:40]:<40} {idx['close']:>12,.2f} ({change:>+6.2f}%)")
        
        # 52-Week Highs
        print("\n🔹 52-WEEK HIGHS:")
        print("-"*70)
        highs = api.get_52_week_highs(5)
        if highs:
            for stock in highs:
                print(f"  📈 {stock['symbol']:<8} {stock['name'][:30]:<30} {stock['close']:>10,.2f}")
        
        # 52-Week Lows
        print("\n🔹 52-WEEK LOWS:")
        print("-"*70)
        lows = api.get_52_week_lows(5)
        if lows:
            for stock in lows:
                print(f"  📉 {stock['symbol']:<8} {stock['name'][:30]:<30} {stock['close']:>10,.2f}")
        
        # Most Active
        print("\n🔹 MOST ACTIVE (By Volume):")
        print("-"*70)
        active = api.get_active_stocks(5)
        if active:
            for stock in active:
                volume = stock.get('volume', 0)
                print(f"  💹 {stock['symbol']:<8} {stock['name'][:30]:<30} Vol: {volume:>12,}")
        
        # Top Gainers
        print("\n🔹 TOP GAINERS:")
        print("-"*70)
        gainers = api.get_top_gainers(5)
        if gainers:
            for stock in gainers:
                change = stock.get('change_percent', 0)
                print(f"  🚀 {stock['symbol']:<8} {stock['name'][:30]:<30} {change:>+6.2f}%")
        
        # Dividends
        print("\n🔹 TOP DIVIDEND STOCKS:")
        print("-"*70)
        divs = api.get_dividend_stocks(limit=5)
        if divs:
            for stock in divs:
                div_pct = stock.get('percentage', stock.get('dividend_yield', 0))
                print(f"  💰 {stock['symbol']:<8} {stock['name'][:30]:<30} {div_pct}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Sarmaaya data failed: {e}")
        return False


def show_summary():
    """Show system summary"""
    
    print("\n" + "="*70)
    print("📊 SYSTEM SUMMARY")
    print("="*70)
    
    # Count models
    models_dir = Path("models/saved")
    if models_dir.exists():
        lstm_models = len(list(models_dir.glob("lstm_*.h5")))
        xgb_models = len(list(models_dir.glob("xgboost_*.json")))
        print(f"\n✓ Trained Models:")
        print(f"  - LSTM: {lstm_models}")
        print(f"  - XGBoost: {xgb_models}")
    
    # Count data files
    data_dir = Path("data/raw/historical")
    if data_dir.exists():
        data_files = len(list(data_dir.glob("*.csv")))
        print(f"\n✓ Historical Data:")
        print(f"  - Stocks: {data_files}")
    
    # Check reports
    reports_dir = Path("reports")
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.csv"))
        if reports:
            print(f"\n✓ Reports Generated:")
            for report in reports[-5:]:  # Last 5 reports
                print(f"  - {report.name}")
    
    print("\n" + "="*70)


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='PSX Complete Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_system.py --all              # Run everything
  python run_complete_system.py --collect          # Just collect data
  python run_complete_system.py --train 100        # Train 100 stocks
  python run_complete_system.py --signals          # Generate signals
  python run_complete_system.py --market           # Show market data
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run complete workflow (collect + train + signals)')
    parser.add_argument('--collect', action='store_true', 
                       help='Collect market data')
    parser.add_argument('--train', type=int, metavar='N', 
                       help='Train N stocks (default: 50)')
    parser.add_argument('--signals', action='store_true', 
                       help='Generate trading signals')
    parser.add_argument('--market', action='store_true', 
                       help='Show Sarmaaya market data')
    parser.add_argument('--summary', action='store_true', 
                       help='Show system summary')
    
    args = parser.parse_args()
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("🚀 PSX AI TRADING SYSTEM")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    # Run requested operations
    if args.all:
        # Complete workflow
        success &= collect_data()
        success &= train_models(max_stocks=args.train or 50)
        success &= generate_signals()
        show_sarmaaya_data()
        show_summary()
        
    else:
        if args.collect:
            success &= collect_data()
        
        if args.train is not None:
            success &= train_models(max_stocks=args.train)
        
        if args.signals:
            success &= generate_signals()
        
        if args.market:
            show_sarmaaya_data()
        
        if args.summary:
            show_summary()
    
    # Final status
    print("\n" + "="*70)
    if success:
        print("✅ OPERATION COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("  1. View dashboard: streamlit run dashboard/app.py")
        print("  2. Check reports: dir reports\\")
    else:
        print("⚠️  OPERATION COMPLETED WITH ERRORS")
        print("\nCheck output above for details")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()