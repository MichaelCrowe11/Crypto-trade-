"""
API routes for the crypto trading application
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from app.core.config import settings
from app.core.trading_engine import TradingEngine
from app.exchange.kraken_client import KrakenClient
from app.ml.models import ModelManager
from app.database.models import get_db, Trade, Position, RiskEvent, PortfolioSnapshot


# Create API router
api_router = APIRouter()

# Initialize components (will be done properly in main.py)
trading_engine = None
model_manager = None


# Pydantic models for API
class TradingStatus(BaseModel):
    enabled: bool
    mode: str
    trading_pairs: List[str]
    last_cycle: Optional[datetime] = None


class PortfolioInfo(BaseModel):
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    daily_return: Optional[float] = None


class PositionInfo(BaseModel):
    symbol: str
    side: str
    amount: float
    entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float
    opened_at: datetime


class TradeInfo(BaseModel):
    id: int
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: datetime
    pnl: Optional[float] = None


class PredictionInfo(BaseModel):
    symbol: str
    predictions: Dict[str, float]
    confidence: float
    current_price: float
    timestamp: datetime


class RiskEventInfo(BaseModel):
    event_type: str
    symbol: Optional[str]
    description: str
    severity: str
    occurred_at: datetime


@api_router.get("/status", response_model=TradingStatus)
async def get_trading_status():
    """Get current trading system status"""
    return TradingStatus(
        enabled=settings.trading.enabled,
        mode=settings.trading.mode,
        trading_pairs=settings.trading.trading_pairs
    )


@api_router.get("/portfolio", response_model=PortfolioInfo)
async def get_portfolio_info():
    """Get current portfolio information"""
    try:
        async with KrakenClient() as client:
            trade_balance = await client.get_trade_balance()
            
            return PortfolioInfo(
                total_value=trade_balance.get('equity', 0.0),
                cash_balance=trade_balance.get('free_margin', 0.0),
                positions_value=trade_balance.get('cost_basis', 0.0),
                unrealized_pnl=trade_balance.get('unrealized_pnl', 0.0)
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio info: {str(e)}")


@api_router.get("/positions", response_model=List[PositionInfo])
async def get_positions(db=Depends(get_db)):
    """Get current positions"""
    try:
        positions = db.query(Position).all()
        
        position_infos = []
        for pos in positions:
            position_infos.append(PositionInfo(
                symbol=pos.symbol,
                side=pos.side,
                amount=pos.amount,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                opened_at=pos.opened_at
            ))
        
        return position_infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")


@api_router.get("/trades", response_model=List[TradeInfo])
async def get_trades(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = None,
    db=Depends(get_db)
):
    """Get recent trades"""
    try:
        query = db.query(Trade).order_by(Trade.timestamp.desc())
        
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        
        trades = query.limit(limit).all()
        
        trade_infos = []
        for trade in trades:
            trade_infos.append(TradeInfo(
                id=trade.id,
                order_id=trade.order_id,
                symbol=trade.symbol,
                side=trade.side,
                amount=trade.amount,
                price=trade.price,
                status=trade.status,
                timestamp=trade.timestamp,
                pnl=trade.pnl
            ))
        
        return trade_infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trades: {str(e)}")


@api_router.get("/predictions/{symbol}", response_model=PredictionInfo)
async def get_predictions(symbol: str):
    """Get ML predictions for a symbol"""
    try:
        global model_manager, trading_engine
        
        if not model_manager:
            model_manager = ModelManager()
            model_manager.initialize_models()
        
        if not trading_engine:
            trading_engine = TradingEngine()
        
        # Get market data
        df = await trading_engine.get_market_data(symbol)
        if df.empty:
            raise HTTPException(status_code=404, detail="No market data available")
        
        # Get predictions
        predictions = model_manager.get_predictions(df)
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        # Get current price
        async with KrakenClient() as client:
            ticker_data = await client.get_ticker([symbol])
            current_price = float(ticker_data[symbol]['c'][0])
        
        # Calculate confidence
        pred_values = list(predictions.values())
        prediction_std = pd.Series(pred_values).std()
        confidence = max(0.0, min(1.0, 1.0 - (prediction_std / current_price)))
        
        return PredictionInfo(
            symbol=symbol,
            predictions=predictions,
            confidence=confidence,
            current_price=current_price,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@api_router.get("/market-data/{symbol}")
async def get_market_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=5000),
    interval: int = Query(1, ge=1, le=1440)
):
    """Get historical market data"""
    try:
        async with KrakenClient() as client:
            ohlc_data = await client.get_ohlc(symbol, interval=interval)
            
            if symbol not in ohlc_data:
                raise HTTPException(status_code=404, detail="Symbol not found")
            
            data = ohlc_data[symbol][-limit:]  # Get last N data points
            
            formatted_data = []
            for point in data:
                formatted_data.append({
                    "timestamp": int(point[0]),
                    "open": float(point[1]),
                    "high": float(point[2]),
                    "low": float(point[3]),
                    "close": float(point[4]),
                    "volume": float(point[6])
                })
            
            return {
                "symbol": symbol,
                "interval": interval,
                "data": formatted_data
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")


@api_router.get("/risk-events", response_model=List[RiskEventInfo])
async def get_risk_events(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = None,
    db=Depends(get_db)
):
    """Get recent risk events"""
    try:
        query = db.query(RiskEvent).order_by(RiskEvent.occurred_at.desc())
        
        if severity:
            query = query.filter(RiskEvent.severity == severity)
        
        events = query.limit(limit).all()
        
        event_infos = []
        for event in events:
            event_infos.append(RiskEventInfo(
                event_type=event.event_type,
                symbol=event.symbol,
                description=event.description,
                severity=event.severity,
                occurred_at=event.occurred_at
            ))
        
        return event_infos
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk events: {str(e)}")


@api_router.get("/performance")
async def get_performance_metrics(
    days: int = Query(30, ge=1, le=365),
    db=Depends(get_db)
):
    """Get performance metrics"""
    try:
        # Get portfolio snapshots from the last N days
        start_date = datetime.now() - timedelta(days=days)
        snapshots = db.query(PortfolioSnapshot).filter(
            PortfolioSnapshot.snapshot_at >= start_date
        ).order_by(PortfolioSnapshot.snapshot_at).all()
        
        if not snapshots:
            return {"error": "No performance data available"}
        
        # Calculate metrics
        values = [s.total_value for s in snapshots]
        returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            returns.append(daily_return)
        
        if returns:
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            avg_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            "period_days": days,
            "total_return": (values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
            "average_daily_return": avg_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "current_value": values[-1],
            "peak_value": max(values),
            "data_points": len(snapshots)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@api_router.post("/trading/enable")
async def enable_trading():
    """Enable trading (requires manual override)"""
    # This would typically require authentication and authorization
    # For safety, we don't actually enable trading here
    return {
        "message": "Trading enable request received",
        "note": "Trading must be enabled in configuration file for safety"
    }


@api_router.post("/trading/disable")
async def disable_trading():
    """Disable trading"""
    global trading_engine
    if trading_engine:
        trading_engine.stop()
    
    return {"message": "Trading disabled"}


@api_router.post("/models/retrain")
async def retrain_models():
    """Trigger model retraining"""
    try:
        global model_manager, trading_engine
        
        if not model_manager:
            model_manager = ModelManager()
            model_manager.initialize_models()
        
        if not trading_engine:
            trading_engine = TradingEngine()
        
        results = {}
        
        # Retrain models for each trading pair
        for symbol in settings.trading.trading_pairs:
            df = await trading_engine.get_market_data(symbol, limit=5000)
            if not df.empty:
                training_results = model_manager.train_models(df)
                results[symbol] = training_results
        
        return {
            "message": "Model retraining completed",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain models: {str(e)}")


@api_router.get("/models/status")
async def get_model_status():
    """Get ML model status"""
    try:
        global model_manager
        
        if not model_manager:
            model_manager = ModelManager()
            model_manager.initialize_models()
        
        status = model_manager.get_model_status()
        
        return {
            "models": status,
            "last_training": model_manager.last_training_time.isoformat() if model_manager.last_training_time else None,
            "should_retrain": model_manager.should_retrain()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@api_router.get("/config")
async def get_configuration():
    """Get current configuration (sanitized)"""
    return {
        "trading": {
            "enabled": settings.trading.enabled,
            "mode": settings.trading.mode,
            "trading_pairs": settings.trading.trading_pairs,
            "risk": {
                "max_position_size": settings.trading.risk.max_position_size,
                "max_daily_loss": settings.trading.risk.max_daily_loss,
                "stop_loss": settings.trading.risk.stop_loss,
                "take_profit": settings.trading.risk.take_profit,
                "max_open_positions": settings.trading.risk.max_open_positions
            }
        },
        "ml": {
            "lstm_enabled": settings.ml.lstm_enabled,
            "ensemble_enabled": settings.ml.ensemble_enabled,
            "retraining_frequency_hours": settings.ml.retraining_frequency_hours,
            "min_data_points": settings.ml.min_data_points
        },
        "exchange": {
            "name": settings.exchange.name,
            "sandbox": settings.exchange.sandbox,
            "rate_limit": settings.exchange.rate_limit
        }
    }