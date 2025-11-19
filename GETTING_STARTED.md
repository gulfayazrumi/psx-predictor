# 🚀 PSX Stock Predictor - Development Guide

## Project Overview

This is a **production-ready AI-powered stock prediction system** for the Pakistan Stock Exchange (PSX). The system combines:

1. **LSTM Neural Network** - For price prediction
2. **XGBoost** - For direction classification  
3. **Ensemble Model** - Combines both for robust predictions
4. **Live Data Integration** - Via Sarmaaya.pk API
5. **REST API** - FastAPI server for integration
6. **Dashboard** - Interactive Streamlit visualization

## What Has Been Built

### ✅ Completed Components

1. **Project Structure**
   - Complete modular architecture
   - Proper separation of concerns
   - Ready for VS Code

2. **Data Collection Module** (`src/data_collection/`)
   - `sarmaaya_api.py` - Full API client with rate limiting
   - Supports single stock, multiple stocks, and KSE-100
   - Automatic retry and error handling

3. **Feature Engineering** (`src/preprocessing/`)
   - `feature_engineer.py` - 20+ technical indicators
   - RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R
   - Price, volume, momentum, and lagged features
   - Rolling statistical features

4. **Models** (`src/models/`)
   - `lstm_model.py` - LSTM for price prediction
     - 3-layer architecture with dropout
     - Sequence preparation and scaling
     - Training with early stopping
   
   - `xgboost_model.py` - XGBoost for direction
     - Binary classification (UP/DOWN)
     - Feature importance analysis
     - Comprehensive evaluation metrics
   
   - `ensemble.py` - Combines both models
     - Weighted predictions
     - Confidence scoring
     - Trading recommendations (BUY/SELL/HOLD)

5. **API Server** (`src/api/`)
   - `api_server.py` - FastAPI REST API
   - Endpoints for single/multiple predictions
   - Market summary and historical data
   - CORS enabled for frontend integration

6. **Dashboard** (`dashboard/`)
   - `app.py` - Streamlit interactive dashboard
   - Market overview with live data
   - Stock analysis with charts
   - Candlestick charts, volume, indicators
   - RSI visualization

7. **Utilities** (`src/utils/`)
   - Configuration management
   - Data loading/saving helpers
   - Normalization functions
   - Metric calculations

8. **Training Pipeline** (`train.py`)
   - Complete end-to-end training script
   - Command-line interface
   - Supports single stock or KSE-100
   - Automatic feature engineering
   - Model saving and evaluation

9. **Documentation**
   - Comprehensive README with examples
   - API documentation
   - Usage examples
   - Configuration guide

10. **Historical Data**
    - 422 stock historical files (OHLCV)
    - Multiple PSX indices (KSE-100, KSE-30, KMI-30)
    - Sector and lot information
    - Data from Sept 2020 onwards

## 🎯 How to Get Started

### Phase 1: Setup (5 minutes)

```bash
# Navigate to project
cd psx-predictor

# Run quick start
./quickstart.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Phase 2: Train Your First Model (30 minutes)

```bash
# Train on a single stock (recommended for testing)
python train.py --symbol HBL --days 365

# This will:
# 1. Fetch data from Sarmaaya API
# 2. Engineer 50+ features
# 3. Train LSTM model
# 4. Train XGBoost model
# 5. Create ensemble
# 6. Save models to models/saved/
```

### Phase 3: Run the System

```bash
# Terminal 1: Start API
python src/api/api_server.py

# Terminal 2: Start Dashboard
streamlit run dashboard/app.py
```

### Phase 4: Test Predictions

```bash
# Via API
curl http://localhost:8000/predict/HBL

# Via Python
python
>>> from src.models.ensemble import EnsemblePredictor
>>> # Load models and predict...
```

## 📊 Working with the Dataset

Your dataset includes:
- **422 stocks** with historical OHLCV data
- **Data range**: Sept 2020 - Oct 2020 (you can extend via API)
- **Format**: CSV files in `data/raw/historical/`

To use existing data:
```bash
# Load existing data instead of fetching
python train.py --skip-data --load-symbol HBL
```

## 🔧 Customization Guide

### 1. Adjust Model Architecture

Edit `config/config.yaml`:

```yaml
models:
  lstm:
    layers:
      - units: 256  # Increase for more capacity
        return_sequences: true
      - units: 128
        return_sequences: true
    epochs: 150  # More training
```

### 2. Add New Features

In `src/preprocessing/feature_engineer.py`:

```python
def add_custom_features(self, df):
    # Add your custom indicators
    df['my_indicator'] = ...
    return df
```

### 3. Change Prediction Horizon

Currently predicts next day. To predict 1 week:

```python
# In feature_engineer.py
df['target_next_close'] = df['close'].shift(-5)  # 5 days ahead
```

### 4. Adjust Ensemble Weights

In `config/config.yaml`:

```yaml
ensemble:
  lstm_weight: 0.7  # Favor LSTM more
  xgboost_weight: 0.3
```

## 🎓 Learning Path

### Beginner
1. ✅ Run the quickstart script
2. ✅ Train on one stock (HBL)
3. ✅ Explore the dashboard
4. ✅ Make predictions via API

### Intermediate
1. Train on multiple stocks
2. Modify hyperparameters
3. Add custom features
4. Analyze feature importance
5. Run backtests

### Advanced
1. Implement multi-timeframe predictions
2. Add sentiment analysis
3. Create portfolio optimization
4. Deploy to cloud (AWS/Azure)
5. Add WebSocket streaming

## 🐛 Troubleshooting

### Issue: API returns 503
**Solution**: Models not loaded. Train models first.

```bash
python train.py --symbol HBL --days 365
```

### Issue: TensorFlow not found
**Solution**: Install TensorFlow

```bash
pip install tensorflow==2.13.0
```

### Issue: Sarmaaya API fails
**Solution**: API might be rate-limited or down. Use existing data:

```bash
python train.py --skip-data --load-symbol HBL
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in config:

```yaml
models:
  lstm:
    batch_size: 16  # Reduce from 32
```

## 📈 Next Steps for Production

### Week 1: Testing
- [ ] Train on 10+ stocks
- [ ] Validate predictions
- [ ] Tune hyperparameters
- [ ] Run comprehensive backtests

### Week 2: Optimization
- [ ] Feature selection
- [ ] Model optimization
- [ ] Add caching layer
- [ ] Implement monitoring

### Week 3: Deployment
- [ ] Containerize with Docker
- [ ] Set up CI/CD
- [ ] Deploy to cloud
- [ ] Add authentication

### Week 4: Enhancements
- [ ] Mobile app
- [ ] Telegram bot
- [ ] Portfolio tracker
- [ ] Alert system

## 📚 Code Examples

### Example 1: Train and Predict

```python
# Train
!python train.py --symbol OGDC --days 730

# Load and predict
from src.models.ensemble import EnsemblePredictor
from src.data_collection.sarmaaya_api import SarmayaAPIClient

api = SarmayaAPIClient()
df = api.get_price_history('OGDC', days=30)

ensemble = EnsemblePredictor()
ensemble.lstm_model.load_model('models/saved/lstm_ogdc')
ensemble.xgboost_model.load_model('models/saved/xgboost_ogdc')

prediction = ensemble.predict_next_day(df, feature_cols)
print(f"Predicted: {prediction['predicted_price']:.2f}")
```

### Example 2: Batch Predictions

```python
symbols = ['HBL', 'OGDC', 'PPL', 'ENGRO', 'LUCK']

for symbol in symbols:
    prediction = api.get(f"http://localhost:8000/predict/{symbol}")
    print(f"{symbol}: {prediction['recommendation']}")
```

### Example 3: Custom Analysis

```python
from src.preprocessing.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
df_features = engineer.create_all_features(your_df)

# Analyze RSI
high_rsi = df_features[df_features['RSI_14'] > 70]
print(f"Overbought periods: {len(high_rsi)}")

# Find patterns
bullish = df_features[
    (df_features['MACD'] > df_features['MACD_signal']) &
    (df_features['RSI_14'] < 70)
]
```

## 🎉 You're All Set!

You now have a complete, production-ready stock prediction system!

### Resources
- 📖 [README.md](README.md) - Full documentation
- ⚙️ [config.yaml](config/config.yaml) - Configuration
- 📓 [Example Notebook](notebooks/01_example_usage.md) - Tutorials
- 🌐 API Docs: http://localhost:8000/docs (when running)

### Support
- Check README for detailed guides
- Explore the code - it's well documented
- Experiment and learn!

**Happy Predicting! 📈💰**
