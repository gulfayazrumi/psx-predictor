"""
Diagnose CSV file issues
"""
import pandas as pd
from pathlib import Path

def diagnose_csv(symbol):
    """Diagnose a specific CSV file"""
    csv_path = Path(f"data/raw/historical/{symbol}.csv")
    
    print(f"\n{'='*70}")
    print(f"🔍 DIAGNOSING: {symbol}")
    print(f"{'='*70}")
    
    if not csv_path.exists():
        print(f"❌ File does not exist: {csv_path}")
        return
    
    try:
        # Read raw
        df = pd.read_csv(csv_path)
        print(f"✓ File loaded")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Shape: {df.shape}")
        
        # Check for duplicates
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].tolist()
            print(f"⚠️  Duplicate columns: {dups}")
        
        # Check for empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            print(f"⚠️  Empty columns: {empty_cols}")
        
        # Check date column
        date_col = None
        if 'TIME' in df.columns:
            date_col = 'TIME'
        elif 'Date' in df.columns:
            date_col = 'Date'
        elif 'DATE' in df.columns:
            date_col = 'DATE'
        
        if date_col:
            print(f"✓ Date column: {date_col}")
            print(f"  Sample values: {df[date_col].head(3).tolist()}")
            print(f"  Data type: {df[date_col].dtype}")
        else:
            print(f"❌ No date column found")
        
        # Check OHLCV
        for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
            if col in df.columns:
                print(f"✓ {col}: {df[col].dtype}")
            elif col.lower() in df.columns:
                print(f"⚠️  {col.lower()} (lowercase)")
            elif col.capitalize() in df.columns:
                print(f"⚠️  {col.capitalize()} (capitalized)")
            else:
                print(f"❌ {col} missing")
        
        # Show first row
        print(f"\n📋 First row:")
        print(df.iloc[0])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # Diagnose problematic stocks
    for symbol in ['AABS', 'ADAMS', 'FFC', 'SYS', '786']:
        diagnose_csv(symbol)