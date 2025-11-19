"""
Configuration and utility functions
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Setup project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = CONFIG_PATH
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_project_paths() -> Dict[str, Path]:
    """Get all project paths"""
    config = load_config()
    paths = config['paths']
    
    return {
        'root': PROJECT_ROOT,
        'data_raw': PROJECT_ROOT / paths['data_raw'],
        'data_processed': PROJECT_ROOT / paths['data_processed'],
        'data_features': PROJECT_ROOT / paths['data_features'],
        'predictions': PROJECT_ROOT / paths['predictions'],
        'models': PROJECT_ROOT / paths['models'],
        'logs': PROJECT_ROOT / paths['logs']
    }


def create_directories():
    """Create all necessary directories"""
    paths = get_project_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    print("✓ All directories created successfully!")


def save_dataframe(df: pd.DataFrame, filename: str, data_type: str = 'processed'):
    """Save dataframe to appropriate directory"""
    paths = get_project_paths()
    
    if data_type == 'raw':
        filepath = paths['data_raw'] / filename
    elif data_type == 'processed':
        filepath = paths['data_processed'] / filename
    elif data_type == 'features':
        filepath = paths['data_features'] / filename
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath}")
    return filepath


def load_dataframe(filename: str, data_type: str = 'processed') -> pd.DataFrame:
    """Load dataframe from appropriate directory"""
    paths = get_project_paths()
    
    if data_type == 'raw':
        filepath = paths['data_raw'] / filename
    elif data_type == 'processed':
        filepath = paths['data_processed'] / filename
    elif data_type == 'features':
        filepath = paths['data_features'] / filename
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    return pd.read_csv(filepath)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns"""
    return prices.pct_change()


def normalize_data(data: np.ndarray, method: str = 'minmax') -> tuple:
    """
    Normalize data
    Returns: (normalized_data, scaler_params)
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        scaler_params = {'min': min_val, 'max': max_val}
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        normalized = (data - mean) / (std + 1e-8)
        scaler_params = {'mean': mean, 'std': std}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, scaler_params


def denormalize_data(data: np.ndarray, scaler_params: Dict, method: str = 'minmax') -> np.ndarray:
    """Denormalize data back to original scale"""
    if method == 'minmax':
        return data * (scaler_params['max'] - scaler_params['min']) + scaler_params['min']
    elif method == 'zscore':
        return data * scaler_params['std'] + scaler_params['mean']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Get trading days between start and end date (excluding weekends)"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # B = business days
    return dates


def is_market_open() -> bool:
    """Check if PSX market is currently open"""
    now = datetime.now()
    
    # PSX trading hours: 9:30 AM - 3:30 PM PKT (Monday-Friday)
    if now.weekday() >= 5:  # Weekend
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    return market_open <= now <= market_close


def format_currency(amount: float, currency: str = "PKR") -> str:
    """Format amount as currency"""
    return f"{currency} {amount:,.2f}"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    config = load_config()
    print(f"✓ Config loaded: {len(config)} sections")
    
    create_directories()
    print(f"✓ Project root: {PROJECT_ROOT}")
