"""
Fix duplicate columns in all CSV files
"""
import pandas as pd
from pathlib import Path

def fix_csv(file_path):
    """Remove duplicate columns and empty columns"""
    try:
        df = pd.read_csv(file_path)
        
        # Get original column count
        orig_cols = len(df.columns)
        
        # Keep only columns with data (remove all-NaN columns)
        df = df.dropna(axis=1, how='all')
        
        # Remove duplicate column names (keep first)
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Standardize column names to uppercase
        df.columns = df.columns.str.upper()
        
        # Rename TIME to DATE if exists
        if 'TIME' in df.columns:
            df = df.rename(columns={'TIME': 'DATE'})
        
        # Parse dates
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y', errors='coerce')
            df = df[df['DATE'].notna()]
        
        # Convert OHLCV to numeric
        for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with no OHLC data
        if all(c in df.columns for c in ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
            df = df[df[['OPEN', 'HIGH', 'LOW', 'CLOSE']].notna().all(axis=1)]
        
        # Sort by date
        if 'DATE' in df.columns:
            df = df.sort_values('DATE')
        
        # Save if we have data
        if len(df) > 0:
            # Keep only essential columns
            cols_to_keep = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[cols_to_keep]
            
            df.to_csv(file_path, index=False)
            return True, f"{len(df)} rows, {orig_cols}→{len(df.columns)} cols"
        
        return False, "No data after cleaning"
        
    except Exception as e:
        return False, str(e)

def main():
    historical_dir = Path("data/raw/historical")
    
    print("\n" + "="*70)
    print("🧹 FIXING DUPLICATE COLUMNS IN ALL CSV FILES")
    print("="*70 + "\n")
    
    success = 0
    failed = 0
    
    for csv_file in sorted(historical_dir.glob("*.csv")):
        result, msg = fix_csv(csv_file)
        
        if result:
            print(f"✓ {csv_file.stem:<10} {msg}")
            success += 1
        else:
            print(f"✗ {csv_file.stem:<10} {msg}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Fixed: {success} files")
    print(f"❌ Failed: {failed} files")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()