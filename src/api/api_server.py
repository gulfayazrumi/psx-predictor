"""
FastAPI REST API for Stock Predictions
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config
from src.data_collection.sarmaaya_api import SarmayaAPIClient
from src.preprocessing.feature_engineer import FeatureEngineer
from src.models.ensemble import EnsemblePredictor

# Initialize FastAPI
app = FastAPI(
    title="PSX Stock Predictor API",
    description="AI-powered stock prediction API for Pakistan Stock Exchange",
    version="1.0.0"
)

# Load configuration
config = load_config()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded on startup)
ensemble_model = None
api_client = None
feature_engineer = None


# Pydantic models for request/response
class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    direction: str
    confidence: float
    signal_strength: str
    recommendation: str
    models_agree: bool
    lstm: Dict
    xgboost: Dict


class MultiStockPrediction(BaseModel):
    predictions: Dict[str, PredictionResponse]
    top_opportunities: List[Dict]


class MarketSummaryResponse(BaseModel):
    timestamp: str
    market_view: Optional[Dict]
    top_gainers: Optional[List[Dict]]
    top_losers: Optional[List[Dict]]
    most_active: Optional[List[Dict]]


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global ensemble_model, api_client, feature_engineer
    
    print("Loading models...")
    
    # Initialize components
    api_client = SarmayaAPIClient()
    feature_engineer = FeatureEngineer()
    
    # TODO: Load trained models
    # ensemble_model = EnsemblePredictor()
    # ensemble_model.lstm_model.load_model("models/saved/lstm_model")
    # ensemble_model.xgboost_model.load_model("models/saved/xgboost_model")
    
    print("✓ API ready!")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "PSX Stock Predictor API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/{symbol}",
            "predict_multiple": "/predict/multiple",
            "market_summary": "/market/summary",
            "stock_history": "/stock/{symbol}/history"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ensemble_model is not None
    }


@app.get("/predict/{symbol}")
async def predict_stock(
    symbol: str,
    days: int = Query(default=365, ge=30, le=1000, description="Days of historical data")
):
    """
    Predict next day price for a specific stock
    
    Args:
        symbol: Stock symbol (e.g., HBL, OGDC, PPL)
        days: Number of days of historical data to use
    
    Returns:
        Prediction with confidence and recommendation
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Fetch historical data
        df = api_client.get_price_history(symbol.upper(), days=days)
        
        if df is None or len(df) < 100:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {symbol}")
        
        # Engineer features
        df_features = feature_engineer.create_all_features(df)
        
        if len(df_features) < 60:
            raise HTTPException(status_code=400, detail="Not enough data after feature engineering")
        
        # Get feature columns (exclude date and target columns)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'time', 'target_next_close', 'target_direction']]
        
        # Make prediction
        prediction = ensemble_model.predict_next_day(df_features, feature_cols)
        
        return {
            "symbol": symbol.upper(),
            "timestamp": datetime.now().isoformat(),
            **prediction
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/multiple")
async def predict_multiple_stocks(
    symbols: str = Query(..., description="Comma-separated stock symbols"),
    days: int = Query(default=365, ge=30, le=1000),
    top_n: int = Query(default=10, ge=1, le=50)
):
    """
    Predict for multiple stocks and get top opportunities
    
    Args:
        symbols: Comma-separated symbols (e.g., HBL,OGDC,PPL)
        days: Days of historical data
        top_n: Number of top opportunities to return
    
    Returns:
        Predictions for all stocks and top opportunities
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    
    predictions = {}
    stocks_data = {}
    
    # Fetch and process each stock
    for symbol in symbol_list:
        try:
            df = api_client.get_price_history(symbol, days=days)
            if df is not None and len(df) >= 100:
                df_features = feature_engineer.create_all_features(df)
                if len(df_features) >= 60:
                    stocks_data[symbol] = df_features
        except Exception as e:
            print(f"Failed to process {symbol}: {e}")
            continue
    
    if not stocks_data:
        raise HTTPException(status_code=404, detail="No valid stock data found")
    
    # Get feature columns
    feature_cols = [col for col in list(stocks_data.values())[0].columns 
                   if col not in ['date', 'time', 'target_next_close', 'target_direction']]
    
    # Make predictions
    predictions = ensemble_model.predict_multiple_stocks(stocks_data, feature_cols)
    
    # Get top opportunities
    top_opps = ensemble_model.get_top_opportunities(predictions, top_n=top_n)
    
    return {
        "predictions": predictions,
        "top_opportunities": top_opps.to_dict('records') if not top_opps.empty else []
    }


@app.get("/predict/kse100")
async def predict_kse100(
    days: int = Query(default=365, ge=30, le=1000),
    top_n: int = Query(default=10, ge=1, le=50)
):
    """
    Predict for all KSE-100 stocks and get top opportunities
    
    Args:
        days: Days of historical data
        top_n: Number of top opportunities
    
    Returns:
        Top opportunities from KSE-100
    """
    if ensemble_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Get KSE-100 stocks
    kse100_stocks = api_client.get_kse100_stocks()
    
    if not kse100_stocks:
        raise HTTPException(status_code=404, detail="Failed to fetch KSE-100 list")
    
    # Use predict_multiple_stocks
    symbols_str = ','.join(kse100_stocks[:50])  # Limit to first 50 for speed
    
    return await predict_multiple_stocks(symbols=symbols_str, days=days, top_n=top_n)


@app.get("/market/summary")
async def get_market_summary():
    """
    Get comprehensive market summary
    
    Returns:
        Market view, top gainers, losers, and most active stocks
    """
    try:
        summary = api_client.get_market_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market summary: {str(e)}")


@app.get("/stock/{symbol}/history")
async def get_stock_history(
    symbol: str,
    days: int = Query(default=30, ge=1, le=1000)
):
    """
    Get historical price data for a stock
    
    Args:
        symbol: Stock symbol
        days: Number of days
    
    Returns:
        Historical OHLCV data
    """
    try:
        df = api_client.get_price_history(symbol.upper(), days=days)
        
        if df is None:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        return {
            "symbol": symbol.upper(),
            "data": df.to_dict('records'),
            "count": len(df)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stocks/list")
async def list_stocks(index: str = Query(default="KSE100", description="Index name")):
    """
    Get list of stocks in an index
    
    Args:
        index: Index name (KSE100, KSE30, etc.)
    
    Returns:
        List of stock symbols
    """
    try:
        if index.upper() == "KSE100":
            stocks = api_client.get_kse100_stocks()
            return {"index": index, "stocks": stocks, "count": len(stocks)}
        else:
            raise HTTPException(status_code=400, detail="Only KSE100 supported currently")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = config['api']['port']
    host = config['api']['host']
    
    print(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=config['api']['reload'],
        log_level=config['api']['log_level']
    )
