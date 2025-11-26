"""
Update trading signals with LIVE market prices and v12 predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.data_collection.sarmaaya_api import SarmayaAPI
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostDirectionPredictor
from src.models.ensemble import EnsemblePredictor
from src.preprocessing.feature_engineer import FeatureEngineer

def get_live_prices():
    """Fetch live prices from Sarmaaya API"""
    print("\n📡 Fetching live market prices...")
    api = SarmayaAPI()
    
    all_stocks = []
    for page in range(1, 11):
        try:
            stocks = api.get_all_stocks(page=page, limit=50)
            if not stocks:
                break
            all_stocks.extend(stocks)
            time.sleep(0.3)
        except:
            break
    
    # Convert to dict for easy lookup
    live_prices = {}
    for stock in all_stocks:
        live_prices[stock['symbol']] = {
            'current_price': stock['close'],
            'open': stock.get('open', stock['close']),
            'high': stock.get('high', stock['close']),
            'low': stock.get('low', stock['close']),
            'volume': stock.get('volume', 0),
            'change': stock.get('change', 0),
            'change_pct': stock.get('change_pct', 0)
        }
    
    print(f"✓ Got live prices for {len(live_prices)} stocks")
    return live_prices

def predict_with_models(symbol, models_dir):
    """Generate prediction using v12 models"""
    try:
        # Load historical data for features
        csv_path = Path(f"data/raw/historical/{symbol}.csv")
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.loc[:, ~df.columns.duplicated()]
        
        if 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        df = df.sort_values('date')
        
        if len(df) < 60:
            return None
            
        # Feature Engineering
        engineer = FeatureEngineer()
        df_features = engineer.create_all_features(df)
        
        # Check if models exist
        lstm_path = models_dir / f"lstm_{symbol.lower()}.h5"
        xgb_path = models_dir / f"xgboost_{symbol.lower()}.json"
        
        if not lstm_path.exists() or not xgb_path.exists():
            return None
            
        # Load models
        lstm = LSTMPredictor(lookback=60, model_type='v12')
        if not lstm.load_model(lstm_path.parent / f"lstm_{symbol.lower()}"):
            return None
            
        xgb = XGBoostDirectionPredictor()
        xgb.load_model(str(models_dir / f"xgboost_{symbol.lower()}"))
        
        # Ensemble
        ensemble = EnsemblePredictor()
        ensemble.set_models(lstm, xgb)
        
        # Prepare features
        feature_cols_xgb = [col for col in df_features.columns 
                           if col not in ['date', 'time', 'target_next_close', 
                                         'target_direction', 'close', 'open', 'high', 'low', 'volume']]
        
        # Predict
        prediction = ensemble.predict_next_day(df_features, feature_cols_xgb)
        
        return prediction
        
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("🔄 UPDATING LIVE TRADING SIGNALS")
    print("="*70)
    
    # Step 1: Get live prices from market
    live_prices = get_live_prices()
    
    # Step 2: Get predictions from v12 models
    models_dir = Path("models/v12")
    if not models_dir.exists():
        print("❌ No v12 models found. Please run training first.")
        return
        
    model_files = list(models_dir.glob("lstm_*.h5"))
    stocks_with_models = [f.stem.replace("lstm_", "").upper() for f in model_files]
    
    print(f"\n🤖 Generating predictions for {len(stocks_with_models)} stocks...")
    
    signals = []
    
    for i, symbol in enumerate(stocks_with_models, 1):
        print(f"[{i}/{len(stocks_with_models)}] {symbol}...", end=' ', flush=True)
        
        # Get live price
        if symbol not in live_prices:
            print("✗ (no live price)")
            continue
            
        live_data = live_prices[symbol]
        
        # Get prediction
        prediction = predict_with_models(symbol, models_dir)
        
        if prediction:
            # Recalculate metrics using LIVE price
            live_price = live_data['current_price']
            predicted_price = prediction['predicted_price']
            
            # Calculate change based on LIVE price vs Predicted price
            if live_price > 0:
                percent_change = ((predicted_price - live_price) / live_price) * 100
            else:
                percent_change = 0.0
                
            direction = 'UP' if percent_change > 0 else 'DOWN'
            
            # Update recommendation based on new metrics
            confidence = prediction['confidence']
            
            if confidence > 0.7:
                signal_strength = 'STRONG'
            elif confidence > 0.6:
                signal_strength = 'MODERATE'
            else:
                signal_strength = 'WEAK'
                
            if signal_strength == 'STRONG' and direction == 'UP':
                recommendation = 'BUY'
            elif signal_strength == 'STRONG' and direction == 'DOWN':
                recommendation = 'SELL'
            elif signal_strength == 'MODERATE':
                recommendation = 'HOLD' if percent_change > -1 else 'SELL'
            else:
                recommendation = 'HOLD'

            # Combine live price with prediction
            signal = {
                'symbol': symbol,
                'current_price': live_price,  # LIVE from API
                'predicted_price': predicted_price,  # From model
                'percent_change': percent_change,  # Recalculated
                'direction': direction,  # Recalculated
                'confidence': confidence,
                'signal_strength': signal_strength, # Recalculated
                'recommendation': recommendation, # Recalculated
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'market_change': live_data['change'],
                'market_change_pct': live_data['change_pct']
            }
            signals.append(signal)
            print(f"✓ (Live: {live_price}, Pred: {predicted_price:.2f}, Change: {percent_change:.2f}%)")
        else:
            print("✗ (prediction failed)")
    
    if signals:
        # Save to CSV
        df = pd.DataFrame(signals)
        output_path = Path("reports/trading_signals.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved {len(signals)} live signals to {output_path}")
        print(f"📊 Signals include:")
        print(f"   - LIVE current prices from market")
        print(f"   - AI predictions for tomorrow")
        print(f"   - Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n⚠️ No signals generated.")

if __name__ == "__main__":
    main()