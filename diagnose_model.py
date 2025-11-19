"""
Diagnose what features the model expects
"""
from pathlib import Path
import pickle

symbol = "SYS"

# Check scaler
scaler_path = Path(f"models/saved/scaler_{symbol.lower()}.pkl")

if scaler_path.exists():
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler found")
    print(f"  Expected features: {scaler.n_features_in_}")
else:
    print(f"❌ No scaler found at: {scaler_path}")
    print(f"\nChecking all model files for {symbol}:")
    models_dir = Path("models/saved")
    for file in models_dir.glob(f"*{symbol.lower()}*"):
        print(f"  - {file.name}")