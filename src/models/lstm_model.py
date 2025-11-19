"""
LSTM Model for Stock Price Prediction
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import pickle
from pathlib import Path

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will not work.")


class LSTMPredictor:
    """LSTM model for stock price prediction"""
    
    def __init__(self, config: Optional[Dict] = None, lookback: int = 60):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        if config:
            self.config = config
        else:
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from src.utils import load_config
            self.config = load_config()
        
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
        # Model architecture from config
        self.lstm_config = self.config['models']['lstm']
        self.training_config = self.config['training']
    
    def prepare_sequences(self, data: np.ndarray, 
                         target: np.ndarray, 
                         lookback: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM
        
        Args:
            data: Feature array (n_samples, n_features)
            target: Target array (n_samples,)
            lookback: Number of time steps to look back
        
        Returns:
            X: (n_sequences, lookback, n_features)
            y: (n_sequences,)
        """
        if lookback is None:
            lookback = self.lookback
        
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (lookback, n_features)
        """
        model = Sequential(name='LSTM_Stock_Predictor')
        
        layers_config = self.lstm_config['layers']
        dropout = self.lstm_config['dropout']
        
        # Add LSTM layers
        for i, layer_conf in enumerate(layers_config):
            if i == 0:
                # First layer
                model.add(LSTM(
                    units=layer_conf['units'],
                    return_sequences=layer_conf['return_sequences'],
                    input_shape=input_shape
                ))
            else:
                # Subsequent layers
                model.add(LSTM(
                    units=layer_conf['units'],
                    return_sequences=layer_conf['return_sequences']
                ))
            
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.lstm_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              model_save_path: Optional[str] = None) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training features (n_samples, lookback, n_features)
            y_train: Training target (n_samples,)
            X_val: Validation features
            y_val: Validation target
            model_save_path: Path to save best model
        """
        print(f"Training LSTM model...")
        print(f"  Train shape: {X_train.shape}")
        
        if X_val is not None:
            print(f"  Validation shape: {X_val.shape}")
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        print(f"\n{self.model.summary()}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.training_config['early_stopping']['patience'],
                min_delta=self.training_config['early_stopping']['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if model_save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Training completed!")
        
        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1] if validation_data else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_next_day(self, recent_data: pd.DataFrame, 
                        feature_columns: List[str]) -> Dict:
        """
        Predict next day's price
        
        Args:
            recent_data: Recent data with features (must have at least lookback rows)
            feature_columns: List of feature column names
        
        Returns:
            Dictionary with prediction and confidence
        """
        if len(recent_data) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} rows of data")
        
        # Get last lookback periods
        data = recent_data[feature_columns].values[-self.lookback:]
        
        # Scale data
        data_scaled = self.feature_scaler.transform(data)
        
        # Reshape for LSTM
        X = data_scaled.reshape(1, self.lookback, len(feature_columns))
        
        # Predict
        prediction_scaled = self.predict(X)[0]
        
        # Inverse transform
        # Create dummy array for inverse transform
        dummy = np.zeros((1, 1))
        dummy[0, 0] = prediction_scaled
        prediction = self.scaler.inverse_transform(dummy)[0, 0]
        
        current_price = recent_data['close'].iloc[-1]
        predicted_change = ((prediction - current_price) / current_price) * 100
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(prediction),
            'predicted_change_pct': float(predicted_change),
            'direction': 'UP' if prediction > current_price else 'DOWN'
        }
    
    def save_model(self, filepath: str):
        """Save model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        
        # Save model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)
        
        # Save scalers
        scaler_path = filepath.with_suffix('.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'lookback': self.lookback
            }, f)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scalers saved to {scaler_path}")
    
    def load_model(self, filepath: str):
        """Load model and scalers"""
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix('.h5')
        self.model = load_model(model_path)
        
        # Load scalers
        scaler_path = filepath.with_suffix('.pkl')
        with open(scaler_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_scaler = data['feature_scaler']
            self.lookback = data['lookback']
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scalers loaded from {scaler_path}")


def create_train_test_split(df: pd.DataFrame, 
                            feature_cols: List[str],
                            target_col: str = 'close',
                            test_size: float = 0.2,
                            lookback: int = 60) -> Dict:
    """
    Create train/test split for time series
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        test_size: Proportion of data for testing
        lookback: Lookback period for LSTM
    
    Returns:
        Dictionary with train/test splits
    """
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx - lookback:].copy()  # Include lookback for sequences
    
    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Scale data
    scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create sequences
    lstm = LSTMPredictor(lookback=lookback)
    X_train_seq, y_train_seq = lstm.prepare_sequences(X_train_scaled, y_train_scaled)
    X_test_seq, y_test_seq = lstm.prepare_sequences(X_test_scaled, y_test_scaled)
    
    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_test': X_test_seq,
        'y_test': y_test_seq,
        'scaler': scaler,
        'feature_scaler': feature_scaler,
        'train_dates': train_df['date'].values,
        'test_dates': test_df['date'].values[lookback:]
    }


if __name__ == "__main__":
    print("LSTM Predictor module loaded successfully!")
    
    if TENSORFLOW_AVAILABLE:
        print("✓ TensorFlow is available")
        print(f"  Keras version: {keras.__version__}")
    else:
        print("✗ TensorFlow not available")
