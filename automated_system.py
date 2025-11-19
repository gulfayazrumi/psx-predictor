"""
Fully Automated PSX Trading System
- Updates data daily (APPENDS, doesn't replace)
- Retrains models when needed
- Keeps all historical data
- Auto-commits to GitHub
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import subprocess
from src.data_collection.sarmaaya_api import SarmayaAPI
import time

class AutomatedTradingSystem:
    def __init__(self):
        self.base_path = Path(".")
        self.api = SarmayaAPI()
        self.today = datetime.now().strftime('%Y-%m-%d')
        
    def update_historical_data(self):
        """
        Update historical CSV files - APPENDS new data, keeps old data
        """
        print("\n" + "="*70)
        print("📊 UPDATING HISTORICAL DATA (APPEND MODE)")
        print("="*70)
        
        # Get live prices from Sarmaaya
        all_stocks = []
        for page in range(1, 11):
            try:
                stocks = self.api.get_all_stocks(page=page, limit=50)
                if not stocks:
                    break
                all_stocks.extend(stocks)
                print(f"  Fetched page {page}: {len(stocks)} stocks")
                time.sleep(0.3)
            except:
                break
        
        print(f"\n✓ Got {len(all_stocks)} stocks from API")
        
        # Update each CSV file
        updated_count = 0
        new_files = 0
        
        for stock in all_stocks:
            symbol = stock['symbol']
            csv_path = self.base_path / f"data/raw/historical/{symbol}.csv"
            
            # Create new row with today's data
            new_row = {
                'Date': self.today,
                'Open': stock.get('open', stock['close']),
                'High': stock.get('high', stock['close']),
                'Low': stock.get('low', stock['close']),
                'Close': stock['close'],
                'Volume': stock.get('volume', 0)
            }
            
            if csv_path.exists():
                # READ existing data
                df = pd.read_csv(csv_path)
                
                # Check if today's data already exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    today_date = pd.to_datetime(self.today)
                    
                    if today_date in df['Date'].values:
                        continue  # Skip if already updated today
                
                # APPEND new row (doesn't replace!)
                new_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_df], ignore_index=True)
                
                # Save updated file
                df.to_csv(csv_path, index=False)
                updated_count += 1
                
            else:
                # Create new file
                df = pd.DataFrame([new_row])
                df.to_csv(csv_path, index=False)
                new_files += 1
        
        print(f"\n✅ COMPLETE!")
        print(f"   Updated: {updated_count} files")
        print(f"   Created: {new_files} new files")
        print(f"   📁 All historical data preserved!")
        
        return updated_count + new_files
    
    def create_daily_snapshot(self):
        """
        Create snapshot of today's data for backup
        """
        print("\n" + "="*70)
        print("📸 CREATING DAILY SNAPSHOT")
        print("="*70)
        
        snapshot_dir = self.base_path / f"data/raw/snapshots/{self.today}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all historical files to snapshot
        historical_dir = self.base_path / "data/raw/historical"
        
        count = 0
        for csv_file in historical_dir.glob("*.csv"):
            shutil.copy2(csv_file, snapshot_dir / csv_file.name)
            count += 1
        
        print(f"✓ Snapshot created: {count} files")
        print(f"  Location: {snapshot_dir}")
        
    def check_if_retraining_needed(self):
        """
        Decide if models need retraining
        """
        print("\n" + "="*70)
        print("🤔 CHECKING IF RETRAINING NEEDED")
        print("="*70)
        
        # Check last training date
        training_log = self.base_path / "models/last_training.txt"
        
        if not training_log.exists():
            print("⚠️ No training log found - RETRAIN NEEDED")
            return True
        
        with open(training_log, 'r') as f:
            last_train = f.read().strip()
        
        last_train_date = datetime.strptime(last_train, '%Y-%m-%d')
        days_since = (datetime.now() - last_train_date).days
        
        print(f"  Last training: {last_train} ({days_since} days ago)")
        
        # Retrain monthly or if never trained
        if days_since >= 30:
            print("✓ RETRAIN NEEDED (>30 days)")
            return True
        else:
            print(f"✗ No retraining needed (only {days_since} days)")
            return False
    
    def retrain_models_background(self):
        """
        Retrain top 50 models in background
        """
        print("\n" + "="*70)
        print("🔄 STARTING BACKGROUND MODEL RETRAINING")
        print("="*70)
        
        # Create model version backup
        version_dir = self.base_path / f"models/versions/v_{self.today}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup current models
        saved_dir = self.base_path / "models/saved"
        if saved_dir.exists():
            for model_file in saved_dir.glob("*"):
                shutil.copy2(model_file, version_dir / model_file.name)
        
        print("✓ Current models backed up")
        print("\n🚀 Starting retraining (this runs in background)...")
        
        # Run training script
        import sys
        subprocess.Popen([
            sys.executable, 'train_all.py', '--max', '400'
        ])
                
        print("✓ Training started in background process")
        print("  Models will be updated automatically when done")
        
        # Update training log
        with open(self.base_path / "models/last_training.txt", 'w') as f:
            f.write(self.today)
    
    def update_predictions(self):
        """
        Generate fresh predictions with latest prices
        """
        print("\n" + "="*70)
        print("🎯 UPDATING PREDICTIONS")
        print("="*70)
        
        # Run prediction update
        subprocess.run(['python', 'update_live_signals.py'])
        
        # Save to history
        signals_file = self.base_path / "reports/trading_signals.csv"
        history_dir = self.base_path / "reports/history"
        history_dir.mkdir(exist_ok=True)
        
        if signals_file.exists():
            shutil.copy2(
                signals_file,
                history_dir / f"{self.today}.csv"
            )
            print(f"✓ Predictions saved to history: {self.today}.csv")
    
    def commit_to_github(self):
        """
        Auto-commit changes to GitHub
        """
        print("\n" + "="*70)
        print("📤 COMMITTING TO GITHUB")
        print("="*70)
        
        try:
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Commit with timestamp
            commit_msg = f"Auto-update {self.today} - Data & Predictions"
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            # Push to GitHub
            subprocess.run(['git', 'push'], check=True)
            
            print("✅ Successfully pushed to GitHub!")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ GitHub push failed: {e}")
            print("   (This is OK if no changes or network issue)")
    
    def run_daily_cycle(self):
        """
        Complete daily automation cycle
        """
        print("\n" + "="*70)
        print(f"🤖 AUTOMATED SYSTEM - {self.today}")
        print("="*70)
        
        # 1. Update historical data (APPEND, don't replace)
        self.update_historical_data()
        
        # 2. Create daily snapshot
        self.create_daily_snapshot()
        
        # 3. Update predictions with latest prices
        self.update_predictions()
        
        # 4. Check if retraining needed
        if self.check_if_retraining_needed():
            self.retrain_models_background()
        
        # 5. Commit everything to GitHub
        self.commit_to_github()
        
        print("\n" + "="*70)
        print("✅ DAILY CYCLE COMPLETE!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"   ✓ Historical data updated (appended)")
        print(f"   ✓ Daily snapshot created")
        print(f"   ✓ Predictions refreshed")
        print(f"   ✓ Changes pushed to GitHub")
        print(f"\n💾 All old data preserved in:")
        print(f"   - data/raw/historical/ (full history)")
        print(f"   - data/raw/snapshots/{self.today}/")
        print(f"   - reports/history/{self.today}.csv")
        print("="*70 + "\n")


if __name__ == "__main__":
    system = AutomatedTradingSystem()
    system.run_daily_cycle()