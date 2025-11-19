"""
Remove duplicate dates and standardize format
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

def clean_csv(csv_file):
    """Remove duplicates and standardize dates"""
    
    try:
        df = pd.read_csv(csv_file)
        
        # Find date column
        date_col = None
        for col in ['TIME', 'Date', 'time', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return False
        
        # Convert all dates to pandas datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=[date_col], keep='first')
        
        # Sort by date (newest first)
        df = df.sort_values(date_col, ascending=False)
        
        # Format dates as DD-MMM-YY
        df[date_col] = df[date_col].dt.strftime('%d-%b-%y')
        
        # Save
        df.to_csv(csv_file, index=False)
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Clean all CSV files"""
    
    print("\n" + "="*70)
    print("🧹 CLEANING DUPLICATE DATES")
    print("="*70)
    
    data_dir = Path("data/raw/historical")
    csv_files = list(data_dir.glob("*.csv"))
    
    cleaned = 0
    errors = 0
    
    for csv_file in csv_files:
        if clean_csv(csv_file):
            cleaned += 1
            if cleaned % 50 == 0:
                print(f"  ✓ Cleaned {cleaned} files...")
        else:
            errors += 1
    
    print("\n" + "="*70)
    print(f"✅ COMPLETE!")
    print(f"   Cleaned: {cleaned}")
    print(f"   Errors: {errors}")
    print("="*70)
    
    # Verify
    print("\n📋 Verification:")
    for symbol in ['786', 'HBL', 'OGDC']:
        csv_file = data_dir / f"{symbol}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            date_col = 'TIME' if 'TIME' in df.columns else 'Date'
            print(f"   {symbol}: {len(df)} rows, latest = {df[date_col].iloc[0]}")


if __name__ == "__main__":
    main()