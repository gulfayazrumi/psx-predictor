"""
COMPLETE UPDATE WORKFLOW
1. Checks data freshness
2. Collects new data if needed
3. Retrains models
4. Generates signals
"""
import subprocess
from pathlib import Path
from datetime import datetime


def main():
    print("\n" + "="*70)
    print("COMPLETE PSX SYSTEM UPDATE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Check data freshness
    print("STEP 1: Checking data freshness...")
    print("-"*70)
    subprocess.run(["python", "check_data_freshness.py"])
    
    print("\nDoes your data need updating?")
    print("If data is VERY OLD or OLD, you should collect fresh data.")
    
    choice = input("\nCollect fresh data now? (yes/no): ").strip().lower()
    
    if choice == 'yes':
        print("\nSTEP 2: Collecting fresh data...")
        print("-"*70)
        subprocess.run(["python", "collect_fresh_data.py"])
    else:
        print("\nSkipping data collection...")
    
    print("\nSTEP 3: Retraining models...")
    print("-"*70)
    retrain = input("Retrain models with current data? (yes/no): ").strip().lower()
    
    if retrain == 'yes':
        stocks_num = input("How many stocks to retrain? (default 50): ").strip()
        stocks_num = stocks_num if stocks_num else "50"
        
        subprocess.run(["python", "train_all.py", "--max", stocks_num])
    
    print("\nSTEP 4: Generating signals...")
    print("-"*70)
    subprocess.run(["python", "generate_signals.py"])
    
    print("\n" + "="*70)
    print("✅ UPDATE COMPLETE!")
    print("="*70)
    print("\nNext: streamlit run dashboard/app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpdate cancelled.")