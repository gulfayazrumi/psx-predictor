from pathlib import Path
import os

models_dir = Path("models/v12")
if not models_dir.exists():
    print("Directory not found")
else:
    files = list(models_dir.glob("lstm_*.h5"))
    print(f"Glob count: {len(files)}")
    
    # Check all files
    all_files = list(models_dir.glob("*"))
    print(f"Total files: {len(all_files)}")
    
    # Print first 5 and last 5 from glob
    print("First 5 glob matches:")
    for f in sorted(files)[:5]:
        print(f"  {f.name}")
        
    print("Last 5 glob matches:")
    for f in sorted(files)[-5:]:
        print(f"  {f.name}")
