"""
Quick Start - Main control panel for PSX Predictor
"""
import subprocess
import sys
from pathlib import Path


def print_menu():
    print("\n" + "="*70)
    print("PSX STOCK PREDICTOR - QUICK START")
    print("="*70)
    print("\n1. View your training results")
    print("2. Test prediction on HBL")
    print("3. Start dashboard")
    print("4. Retrain 5 stocks (test)")
    print("5. Exit")
    print("\n" + "="*70)


def run_command(cmd):
    """Run a command and show output"""
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")


def main():
    while True:
        print_menu()
        choice = input("\nYour choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "="*70)
            run_command("python view_results.py")
        
        elif choice == '2':
            print("\n" + "="*70)
            run_command("python test_single_stock.py HBL")
        
        elif choice == '3':
            print("\n" + "="*70)
            print("Starting dashboard...")
            print("Open browser: http://localhost:8501")
            print("Press Ctrl+C to stop")
            print("="*70 + "\n")
            run_command("streamlit run dashboard/app.py")
        
        elif choice == '4':
            print("\n" + "="*70)
            print("Retraining 5 stocks (this will take ~10 minutes)...")
            print("="*70 + "\n")
            run_command("python train_all.py --max 5")
        
        elif choice == '5':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")