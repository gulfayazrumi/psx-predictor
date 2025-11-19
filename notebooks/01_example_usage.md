# PSX Stock Predictor - Example Usage

This notebook demonstrates how to use the PSX Stock Predictor system.

## Setup

```python
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils import load_config
from src.data_collection.sarmaaya_api import SarmayaAPIClient
from src.preprocessing.feature_engineer import FeatureEngineer
```

## 1. Fetch Stock Data

```python
# Initialize API client
api_client = SarmayaAPIClient()

# Fetch data for HBL (Habib Bank Limited)
symbol = 'HBL'
df = api_client.get_price_history(symbol, days=365)

print(f"Fetched {len(df)} days of data for {symbol}")
df.head()
```

## 2. Feature Engineering

```python
# Initialize feature engineer
engineer = FeatureEngineer()

# Create all features
df_features = engineer.create_all_features(df)

print(f"Created {len(df_features.columns)} features")
print("\nFeature columns:")
for i, col in enumerate(df_features.columns, 1):
    print(f"{i:3d}. {col}")
```

## 3. Visualize Data

```python
import plotly.graph_objects as go

# Candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

fig.update_layout(
    title=f'{symbol} Stock Price',
    yaxis_title='Price (PKR)',
    xaxis_title='Date',
    template='plotly_white'
)

fig.show()
```

```python
# Technical indicators
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['close'], 
                         mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['SMA_20'], 
                         mode='lines', name='SMA 20'))
fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['SMA_50'], 
                         mode='lines', name='SMA 50'))

fig.update_layout(
    title='Price with Moving Averages',
    yaxis_title='Price (PKR)',
    xaxis_title='Date',
    template='plotly_white'
)

fig.show()
```

## 4. Train LSTM Model

```python
from src.models.lstm_model import LSTMPredictor, create_train_test_split

# Get feature columns
feature_cols = [col for col in df_features.columns 
                if col not in ['date', 'time', 'target_next_close', 'target_direction']]

print(f"Using {len(feature_cols)} features for training")

# Create train/test split
data_split = create_train_test_split(
    df_features,
    feature_cols,
    target_col='close',
    test_size=0.2,
    lookback=60
)

print(f"Train shape: {data_split['X_train'].shape}")
print(f"Test shape: {data_split['X_test'].shape}")
```

```python
# Initialize and train LSTM
lstm = LSTMPredictor(lookback=60)
lstm.scaler = data_split['scaler']
lstm.feature_scaler = data_split['feature_scaler']

# Train model (this will take a few minutes)
history = lstm.train(
    data_split['X_train'],
    data_split['y_train'],
    data_split['X_test'],
    data_split['y_test']
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['history']['loss'], label='Train Loss')
plt.plot(history['history']['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['history']['mae'], label='Train MAE')
plt.plot(history['history']['val_mae'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Training MAE')

plt.tight_layout()
plt.show()
```

## 5. Train XGBoost Model

```python
from src.models.xgboost_model import XGBoostDirectionPredictor, prepare_classification_data

# Prepare classification data
feature_cols_clf = [col for col in df_features.columns 
                    if col not in ['date', 'time', 'target_next_close', 
                                  'target_direction', 'close', 'open', 'high', 'low', 'volume']]

data_split_clf = prepare_classification_data(
    df_features,
    feature_cols_clf,
    target_col='target_direction',
    test_size=0.2
)

# Train XGBoost
xgb = XGBoostDirectionPredictor()
results = xgb.train(
    data_split_clf['X_train'],
    data_split_clf['y_train'],
    data_split_clf['X_test'],
    data_split_clf['y_test']
)

# Evaluate
eval_results = xgb.evaluate(
    data_split_clf['X_test'],
    data_split_clf['y_test'],
    feature_names=feature_cols_clf
)
```

## 6. Create Ensemble & Make Predictions

```python
from src.models.ensemble import EnsemblePredictor

# Create ensemble
ensemble = EnsemblePredictor()
ensemble.set_models(lstm, xgb)

# Make prediction
prediction = ensemble.predict_next_day(df_features, feature_cols)

print("="*60)
print("PREDICTION RESULTS")
print("="*60)
print(f"Symbol: {symbol}")
print(f"Current Price: PKR {prediction['current_price']:.2f}")
print(f"Predicted Price: PKR {prediction['predicted_price']:.2f}")
print(f"Change: {prediction['predicted_change_pct']:+.2f}%")
print(f"Direction: {prediction['direction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Signal Strength: {prediction['signal_strength']}")
print(f"Recommendation: {prediction['recommendation']}")
print(f"Models Agree: {'Yes' if prediction['models_agree'] else 'No'}")
print("="*60)
```

## 7. Save Models

```python
# Save LSTM
lstm.save_model(f'../models/saved/lstm_{symbol.lower()}')

# Save XGBoost
xgb.save_model(f'../models/saved/xgboost_{symbol.lower()}')

print(f"✓ Models saved for {symbol}")
```

## Next Steps

1. Try different stocks
2. Experiment with hyperparameters
3. Add more features
4. Implement backtesting
5. Deploy models via API

## References

- [Project README](../README.md)
- [Configuration](../config/config.yaml)
- [API Documentation](../src/api/api_server.py)
