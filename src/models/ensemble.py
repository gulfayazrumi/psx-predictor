"""
Ensemble Model - Combines LSTM (price prediction) and XGBoost (direction prediction)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostDirectionPredictor
from src.utils import load_config


class EnsemblePredictor:
    """
    Ensemble model combining:
    - LSTM: Predicts next day's price
    - XGBoost: Predicts direction (UP/DOWN) with confidence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.config = config
        else:
            self.config = load_config()
        
        self.lstm_weight = self.config['models']['ensemble']['lstm_weight']
        self.xgboost_weight = self.config['models']['ensemble']['xgboost_weight']
        
        self.lstm_model = None
        self.xgboost_model = None
        
        print(f"Ensemble Predictor initialized")
        print(f"  LSTM weight: {self.lstm_weight}")
        print(f"  XGBoost weight: {self.xgboost_weight}")
    
    def set_models(self, lstm_model: LSTMPredictor, xgboost_model: XGBoostDirectionPredictor):
        """Set the trained models"""
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        print("✓ Models loaded into ensemble")
    
    def predict_next_day(self, recent_data: pd.DataFrame,
                        feature_columns: List[str],
                        lstm_lookback: int = 60) -> Dict:
        """
        Make ensemble prediction for next day
        
        Args:
            recent_data: Recent historical data with features
            feature_columns: List of feature columns
            lstm_lookback: Lookback period for LSTM
        
        Returns:
            Comprehensive prediction dictionary
        """
        if self.lstm_model is None or self.xgboost_model is None:
            raise ValueError("Models not loaded. Use set_models() first.")
        
        # Get current price
        current_price = recent_data['close'].iloc[-1]
        
        # LSTM Prediction (price)
        try:
            lstm_pred = self.lstm_model.predict_next_day(recent_data, feature_columns)
            lstm_price = lstm_pred['predicted_price']
            lstm_direction = lstm_pred['direction']
            lstm_change_pct = lstm_pred['predicted_change_pct']
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            lstm_price = current_price
            lstm_direction = 'NEUTRAL'
            lstm_change_pct = 0.0
        
        # XGBoost Prediction (direction)
        try:
            xgb_pred = self.xgboost_model.predict_next_day(recent_data, feature_columns)
            xgb_direction = xgb_pred['direction']
            xgb_confidence = xgb_pred['confidence']
            xgb_prob_up = xgb_pred['probability_up']
            xgb_prob_down = xgb_pred['probability_down']
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            xgb_direction = 'NEUTRAL'
            xgb_confidence = 0.5
            xgb_prob_up = 0.5
            xgb_prob_down = 0.5
        
        # Ensemble Logic
        # If LSTM and XGBoost agree on direction, high confidence
        # If they disagree, moderate confidence and average prediction
        
        agreement = (lstm_direction == xgb_direction)
        
        if agreement:
            # Models agree - use LSTM price with XGBoost confidence
            final_direction = lstm_direction
            final_price = lstm_price
            final_confidence = xgb_confidence
        else:
            # Models disagree - weighted average
            # Convert XGBoost probabilities to price adjustment
            if xgb_direction == 'UP':
                xgb_price_adjustment = current_price * (1 + 0.02 * xgb_confidence)  # Assume 2% move
            else:
                xgb_price_adjustment = current_price * (1 - 0.02 * xgb_confidence)
            
            # Weighted average of predictions
            final_price = (self.lstm_weight * lstm_price + 
                          self.xgboost_weight * xgb_price_adjustment)
            
            # Reduce confidence due to disagreement
            final_confidence = xgb_confidence * 0.7
            
            # Final direction based on ensemble price
            final_direction = 'UP' if final_price > current_price else 'DOWN'
        
        # Calculate final metrics
        final_change_pct = ((final_price - current_price) / current_price) * 100
        
        # Generate signal strength
        if final_confidence > 0.7:
            signal_strength = 'STRONG'
        elif final_confidence > 0.6:
            signal_strength = 'MODERATE'
        else:
            signal_strength = 'WEAK'
        
        # Trading recommendation
        if signal_strength == 'STRONG' and final_direction == 'UP':
            recommendation = 'BUY'
        elif signal_strength == 'STRONG' and final_direction == 'DOWN':
            recommendation = 'SELL'
        elif signal_strength == 'MODERATE':
            recommendation = 'HOLD' if final_change_pct > -1 else 'SELL'
        else:
            recommendation = 'HOLD'
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(final_price),
            'predicted_change_pct': float(final_change_pct),
            'direction': final_direction,
            'confidence': float(final_confidence),
            'signal_strength': signal_strength,
            'recommendation': recommendation,
            'models_agree': agreement,
            'lstm': {
                'price': float(lstm_price),
                'direction': lstm_direction,
                'change_pct': float(lstm_change_pct)
            },
            'xgboost': {
                'direction': xgb_direction,
                'confidence': float(xgb_confidence),
                'prob_up': float(xgb_prob_up),
                'prob_down': float(xgb_prob_down)
            }
        }
    
    def predict_multiple_stocks(self, stocks_data: Dict[str, pd.DataFrame],
                               feature_columns: List[str]) -> Dict[str, Dict]:
        """
        Predict for multiple stocks
        
        Args:
            stocks_data: Dictionary {symbol: dataframe}
            feature_columns: List of feature columns
        
        Returns:
            Dictionary {symbol: prediction}
        """
        predictions = {}
        
        for symbol, data in stocks_data.items():
            try:
                pred = self.predict_next_day(data, feature_columns)
                predictions[symbol] = pred
            except Exception as e:
                print(f"Failed to predict {symbol}: {e}")
                predictions[symbol] = {'error': str(e)}
        
        return predictions
    
    def get_top_opportunities(self, predictions: Dict[str, Dict],
                            top_n: int = 10,
                            min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Get top trading opportunities from predictions
        
        Args:
            predictions: Dictionary of predictions from predict_multiple_stocks
            top_n: Number of top opportunities to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            DataFrame with top opportunities
        """
        opportunities = []
        
        for symbol, pred in predictions.items():
            if 'error' in pred:
                continue
            
            if pred['confidence'] >= min_confidence:
                opportunities.append({
                    'symbol': symbol,
                    'current_price': pred['current_price'],
                    'predicted_price': pred['predicted_price'],
                    'change_pct': pred['predicted_change_pct'],
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'signal_strength': pred['signal_strength'],
                    'recommendation': pred['recommendation']
                })
        
        if not opportunities:
            return pd.DataFrame()
        
        df = pd.DataFrame(opportunities)
        
        # Sort by absolute change percentage and confidence
        df['score'] = abs(df['change_pct']) * df['confidence']
        df = df.sort_values('score', ascending=False)
        
        return df.head(top_n)


def backtest_ensemble(ensemble: EnsemblePredictor,
                     test_data: pd.DataFrame,
                     feature_columns: List[str],
                     lookback: int = 60) -> Dict:
    """
    Simple backtest of ensemble predictions
    
    Args:
        ensemble: Trained ensemble model
        test_data: Test dataset with features
        feature_columns: List of feature columns
        lookback: Lookback period
    
    Returns:
        Backtest results
    """
    predictions = []
    actuals = []
    
    print(f"Running backtest on {len(test_data) - lookback} days...")
    
    for i in range(lookback, len(test_data) - 1):
        # Get historical data up to this point
        historical = test_data.iloc[:i+1]
        
        # Make prediction
        try:
            pred = ensemble.predict_next_day(historical, feature_columns, lookback)
            
            # Get actual next day price
            actual_price = test_data.iloc[i+1]['close']
            
            predictions.append(pred)
            actuals.append(actual_price)
        
        except Exception as e:
            continue
    
    # Calculate metrics
    pred_prices = [p['predicted_price'] for p in predictions]
    pred_directions = [1 if p['direction'] == 'UP' else 0 for p in predictions]
    
    actual_changes = [actuals[i] - test_data.iloc[lookback+i]['close'] 
                     for i in range(len(actuals))]
    actual_directions = [1 if c > 0 else 0 for c in actual_changes]
    
    # Direction accuracy
    direction_accuracy = np.mean([p == a for p, a in zip(pred_directions, actual_directions)])
    
    # Price prediction error
    mae = np.mean(np.abs(np.array(pred_prices) - np.array(actuals)))
    mape = np.mean(np.abs((np.array(actuals) - np.array(pred_prices)) / np.array(actuals))) * 100
    
    print(f"\n{'='*50}")
    print("BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Total Predictions: {len(predictions)}")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print(f"MAE (Price): PKR {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*50}")
    
    return {
        'direction_accuracy': direction_accuracy,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'actuals': actuals
    }


if __name__ == "__main__":
    print("Ensemble Predictor module loaded successfully!")
    print("This module combines LSTM and XGBoost for robust predictions")
