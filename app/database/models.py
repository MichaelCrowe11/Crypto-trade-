"""
Database models for the crypto trading application
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import asyncio

from app.core.config import settings


Base = declarative_base()


class Trade(Base):
    """Trade execution record"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(100), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy/sell
    order_type = Column(String(20), nullable=False)  # market/limit
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    filled_amount = Column(Float, default=0.0)
    status = Column(String(20), nullable=False)  # pending/filled/cancelled
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # ML prediction info
    predicted_price = Column(Float)
    confidence = Column(Float)
    model_predictions = Column(Text)  # JSON string of all model predictions
    
    # P&L tracking
    pnl = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Position(Base):
    """Current position tracking"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    side = Column(String(10), nullable=False)  # long/short
    amount = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    
    # Risk management
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MarketData(Base):
    """Historical market data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Technical indicators (can be calculated and stored)
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )


class MLPrediction(Base):
    """ML model predictions"""
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    prediction_type = Column(String(20), nullable=False)  # price/direction
    predicted_value = Column(Float, nullable=False)
    confidence = Column(Float)
    actual_value = Column(Float)  # For performance tracking
    
    # Prediction context
    prediction_horizon = Column(Integer, default=1)  # Hours ahead
    features_used = Column(Text)  # JSON string of features
    
    # Timestamps
    predicted_at = Column(DateTime, default=datetime.utcnow)
    target_time = Column(DateTime, nullable=False, index=True)
    actual_time = Column(DateTime)  # When actual value was observed


class RiskEvent(Base):
    """Risk management events"""
    __tablename__ = "risk_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), index=True)
    description = Column(Text)
    severity = Column(String(20), nullable=False)  # low/medium/high/critical
    
    # Event details
    trigger_value = Column(Float)
    threshold_value = Column(Float)
    action_taken = Column(String(100))
    
    # Timestamps
    occurred_at = Column(DateTime, default=datetime.utcnow, index=True)


class PortfolioSnapshot(Base):
    """Portfolio value snapshots"""
    __tablename__ = "portfolio_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    
    # Performance metrics
    daily_return = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Timestamp
    snapshot_at = Column(DateTime, default=datetime.utcnow, index=True)


class AuditLog(Base):
    """Audit log for compliance"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100))
    user_id = Column(String(100), default="system")
    
    # Change details
    old_values = Column(Text)  # JSON string
    new_values = Column(Text)  # JSON string
    
    # Metadata
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Timestamp
    logged_at = Column(DateTime, default=datetime.utcnow, index=True)


# Database setup
engine = create_engine(
    settings.database.url,
    echo=settings.database.echo,
    pool_size=settings.database.pool_size
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database utility functions
class DatabaseManager:
    """Database operations manager"""
    
    @staticmethod
    def create_trade_record(
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float,
        predicted_price: Optional[float] = None,
        confidence: Optional[float] = None,
        model_predictions: Optional[str] = None
    ) -> Trade:
        """Create a new trade record"""
        db = SessionLocal()
        try:
            trade = Trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                predicted_price=predicted_price,
                confidence=confidence,
                model_predictions=model_predictions,
                status="pending"
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            return trade
        finally:
            db.close()
    
    @staticmethod
    def update_trade_status(order_id: str, status: str, filled_amount: float = None, pnl: float = None):
        """Update trade status"""
        db = SessionLocal()
        try:
            trade = db.query(Trade).filter(Trade.order_id == order_id).first()
            if trade:
                trade.status = status
                if filled_amount is not None:
                    trade.filled_amount = filled_amount
                if pnl is not None:
                    trade.pnl = pnl
                trade.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    @staticmethod
    def create_position(
        symbol: str,
        side: str,
        amount: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Position:
        """Create or update position"""
        db = SessionLocal()
        try:
            # Check if position already exists
            existing_position = db.query(Position).filter(Position.symbol == symbol).first()
            
            if existing_position:
                # Update existing position
                existing_position.amount += amount if existing_position.side == side else -amount
                existing_position.entry_price = (
                    existing_position.entry_price * existing_position.amount + entry_price * amount
                ) / (existing_position.amount + amount)
                existing_position.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing_position)
                return existing_position
            else:
                # Create new position
                position = Position(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                db.add(position)
                db.commit()
                db.refresh(position)
                return position
        finally:
            db.close()
    
    @staticmethod
    def store_market_data(symbol: str, ohlcv_data: list):
        """Store market data batch"""
        db = SessionLocal()
        try:
            market_records = []
            for data_point in ohlcv_data:
                timestamp, open_price, high_price, low_price, close_price, volume = data_point[:6]
                
                # Check if record already exists
                existing = db.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp == datetime.fromtimestamp(timestamp)
                ).first()
                
                if not existing:
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(timestamp),
                        open_price=float(open_price),
                        high_price=float(high_price),
                        low_price=float(low_price),
                        close_price=float(close_price),
                        volume=float(volume)
                    )
                    market_records.append(market_data)
            
            if market_records:
                db.add_all(market_records)
                db.commit()
                
        finally:
            db.close()
    
    @staticmethod
    def log_risk_event(
        event_type: str,
        description: str,
        severity: str = "medium",
        symbol: Optional[str] = None,
        trigger_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        action_taken: Optional[str] = None
    ):
        """Log a risk management event"""
        db = SessionLocal()
        try:
            risk_event = RiskEvent(
                event_type=event_type,
                symbol=symbol,
                description=description,
                severity=severity,
                trigger_value=trigger_value,
                threshold_value=threshold_value,
                action_taken=action_taken
            )
            db.add(risk_event)
            db.commit()
        finally:
            db.close()
    
    @staticmethod
    def create_portfolio_snapshot(
        total_value: float,
        cash_balance: float,
        positions_value: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0
    ):
        """Create portfolio snapshot"""
        db = SessionLocal()
        try:
            snapshot = PortfolioSnapshot(
                total_value=total_value,
                cash_balance=cash_balance,
                positions_value=positions_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl
            )
            db.add(snapshot)
            db.commit()
        finally:
            db.close()
    
    @staticmethod
    def log_audit_event(
        action: str,
        entity_type: str,
        entity_id: str = None,
        old_values: str = None,
        new_values: str = None,
        user_id: str = "system"
    ):
        """Log audit event for compliance"""
        db = SessionLocal()
        try:
            audit_log = AuditLog(
                action=action,
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=user_id,
                old_values=old_values,
                new_values=new_values
            )
            db.add(audit_log)
            db.commit()
        finally:
            db.close()