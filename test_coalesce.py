import pandas as pd
import numpy as np

# Simulate the OGDC.csv structure
data = {
    'DATE': [np.nan, np.nan, np.nan],
    'OPEN': [np.nan, np.nan, np.nan],
    'Date': ['2025-11-25', '2025-11-26', '2025-11-27'],
    'Open': [240.0, 245.0, 250.0]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\nColumns:", df.columns.tolist())

# Normalize
df.columns = df.columns.str.strip().str.lower()
print("\nAfter lowercase:")
print(df)
print("Columns:", df.columns.tolist())

# Try coalesce
def coalesce_rows(x):
    return x.bfill().iloc[0]

df_coalesced = df.T.groupby(level=0).apply(coalesce_rows).T
print("\nAfter coalesce:")
print(df_coalesced)
print("Columns:", df_coalesced.columns.tolist())
print("\ndate column:")
print(df_coalesced['date'])
