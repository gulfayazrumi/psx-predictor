"""
Fix the feature mismatch bug in train_all.py
"""
from pathlib import Path

# Read the file
train_all_path = Path("train_all.py")
content = train_all_path.read_text(encoding='utf-8')

# Find and fix the bug
# The issue is that we need to use the same feature columns for both training and prediction

# Look for the ensemble prediction section
old_code_1 = "prediction = ensemble.predict_next_day(df_features, feature_cols)"
new_code_1 = "prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)"

if old_code_1 in content:
    content = content.replace(old_code_1, new_code_1)
    print("✓ Fixed ensemble prediction call")
else:
    print("⚠️  Prediction call not found or already fixed")

# Also ensure feature_cols_xgb is defined correctly
# Look for where it's defined and make sure it excludes the right columns
old_definition = """feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]"""

new_definition = """feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]
    
    # For XGBoost, exclude OHLCV columns as well
    feature_cols_xgb = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 'target_direction',
                                     'close', 'open', 'high', 'low', 'volume']]"""

if old_definition in content and 'feature_cols_xgb' not in content:
    content = content.replace(old_definition, new_definition)
    print("✓ Fixed feature columns definition")
else:
    print("⚠️  Feature definition not found or already fixed")

# Save the fixed file
train_all_path.write_text(content, encoding='utf-8')

print("\n✅ Bug fix complete!")
print("Now run: python train_all.py --symbol HBL")
print("This will test the fix on a single stock")