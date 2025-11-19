import pandas as pd
from pathlib import Path

historical_dir = Path('data/raw/historical')

print("Fixing CSV files...")

for csv_file in historical_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)
    
    # Keep only first 6 columns
    df = df.iloc[:, :6]
    
    # Set proper column names
    df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
    
    # Fix dates
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y', errors='coerce')
    df = df.dropna(subset=['DATE'])
    
    # Fix numbers
    for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE'])
    
    # Save
    if len(df) > 0:
        df.to_csv(csv_file, index=False)
        print(f"✓ {csv_file.stem}")

print("\nDone! All CSV files fixed.")