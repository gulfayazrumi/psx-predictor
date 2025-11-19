"""
XGBoost Model for Stock Price Direction Classification
Predicts whether price will go UP or DOWN
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")


class XGBoostDirectionPredictor:
    """XGBoost classifier for predicting price direction"""
    
    def __init__(self, config: Optional[Dict] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required")
        
        if config:
            self.config = config
        else:
            import sys
            sys.path.append(str(Path(__file__).parent.parent.parent))
            from src.utils import load_config
            self.config = load_config()
        
        self.xgb_config = self.config['models']['xgboost']
        self.model = None
        self.feature_importance = None
    
    def build_model(self) -> xgb.XGBClassifier:
        """Build XGBoost classifier"""
        model = xgb.XGBClassifier(
            n_estimators=self.xgb_config['n_estimators'],
            max_depth=self.xgb_config['max_depth'],
            learning_rate=self.xgb_config['learning_rate'],
            subsample=self.xgb_config['subsample'],
            colsample_bytree=self.xgb_config['colsample_bytree'],
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - binary (0=DOWN, 1=UP)
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Dictionary with training results
        """
        print(f"Training XGBoost model...")
        print(f"  Train shape: {X_train.shape}")
        print(f"  Class distribution: DOWN={np.sum(y_train == 0)}, UP={np.sum(y_train == 1)}")
        
        self.model = self.build_model()
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"  Validation shape: {X_val.shape}")
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        results = {
            'train_accuracy': train_acc,
            'feature_importance': self.feature_importance
        }
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred)
            val_recall = recall_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred)
            
            results.update({
                'val_accuracy': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1
            })
            
            print(f"\n✓ Training completed!")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val Precision: {val_precision:.4f}")
            print(f"  Val Recall: {val_recall:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
        else:
            print(f"\n✓ Training completed!")
            print(f"  Train Accuracy: {train_acc:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict price direction
        
        Returns:
            Array of predictions (0=DOWN, 1=UP)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of each class
        
        Returns:
            Array of shape (n_samples, 2) with probabilities [DOWN, UP]
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)
    
    def predict_next_day(self, recent_data: pd.DataFrame,
                        feature_columns: List[str]) -> Dict:
        """
        Predict next day's direction
        
        Args:
            recent_data: Recent data with features (last row will be used)
            feature_columns: List of feature column names
        
        Returns:
            Dictionary with prediction and confidence
        """
        # Get last row features
        X = recent_data[feature_columns].iloc[-1:].values
        
        # Predict
        prediction = self.predict(X)[0]
        probabilities = self.predict_proba(X)[0]
        
        direction = 'UP' if prediction == 1 else 'DOWN'
        confidence = probabilities[prediction]
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability_down': float(probabilities[0]),
            'probability_up': float(probabilities[1])
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                feature_names: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features for importance plot
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                      target_names=['DOWN', 'UP'],
                                      output_dict=True)
        
        print("\n" + "=" * 50)
        print("XGBoost Model Evaluation")
        print("=" * 50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              DOWN    UP")
        print(f"Actual DOWN   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       UP     {cm[1,0]:4d}  {cm[1,1]:4d}")
        print("=" * 50)
        
        # Feature importance
        if feature_names is not None and self.feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        
        # Save model
        model_path = filepath.with_suffix('.json')
        self.model.save_model(model_path)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = filepath.with_suffix('.pkl')
            with open(importance_path, 'wb') as f:
                pickle.dump(self.feature_importance, f)
        
        print(f"✓ Model saved to {model_path}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        filepath = Path(filepath)
        
        # Load model
        model_path = filepath.with_suffix('.json')
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        # Load feature importance
        importance_path = filepath.with_suffix('.pkl')
        if importance_path.exists():
            with open(importance_path, 'rb') as f:
                self.feature_importance = pickle.load(f)
        
        print(f"✓ Model loaded from {model_path}")


def prepare_classification_data(df: pd.DataFrame,
                                feature_cols: List[str],
                                target_col: str = 'target_direction',
                                test_size: float = 0.2) -> Dict:
    """
    Prepare data for XGBoost classification
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name (binary: 0=DOWN, 1=UP)
        test_size: Proportion for test set
    
    Returns:
        Dictionary with train/test splits
    """
    # Remove rows with NaN
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Split point (time series split)
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"Data prepared for classification:")
    print(f"  Train: {X_train.shape}, DOWN={np.sum(y_train==0)}, UP={np.sum(y_train==1)}")
    print(f"  Test:  {X_test.shape}, DOWN={np.sum(y_test==0)}, UP={np.sum(y_test==1)}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_dates': train_df['date'].values if 'date' in train_df.columns else None,
        'test_dates': test_df['date'].values if 'date' in test_df.columns else None
    }


if __name__ == "__main__":
    print("XGBoost Direction Predictor module loaded successfully!")
    
    if XGBOOST_AVAILABLE:
        print("✓ XGBoost is available")
        print(f"  Version: {xgb.__version__}")
    else:
        print("✗ XGBoost not available")
