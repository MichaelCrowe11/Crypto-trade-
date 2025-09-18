"""
Trading scheduler for automated execution
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.trading_engine import TradingEngine
from app.database.models import DatabaseManager


logger = setup_logging(__name__)


class TradingScheduler:
    """Scheduler for automated trading operations"""
    
    def __init__(self):
        """Initialize trading scheduler"""
        self.trading_engine: Optional[TradingEngine] = None
        self.is_running = False
        self.tasks = []
    
    async def start(self) -> None:
        """Start the trading scheduler"""
        logger.info("Starting trading scheduler")
        
        self.trading_engine = TradingEngine()
        await self.trading_engine.initialize()
        
        self.is_running = True
        
        # Start scheduled tasks
        if settings.trading.enabled:
            # Main trading loop
            self.tasks.append(asyncio.create_task(self._trading_loop()))
        
        # Portfolio monitoring (always runs)
        self.tasks.append(asyncio.create_task(self._portfolio_monitoring_loop()))
        
        # Model retraining check
        self.tasks.append(asyncio.create_task(self._model_retraining_loop()))
        
        # Data collection
        self.tasks.append(asyncio.create_task(self._data_collection_loop()))
        
        logger.info("Trading scheduler started with {} tasks", len(self.tasks))
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("Trading scheduler tasks cancelled")
    
    def stop(self) -> None:
        """Stop the trading scheduler"""
        logger.info("Stopping trading scheduler")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Stop trading engine
        if self.trading_engine:
            self.trading_engine.stop()
    
    async def _trading_loop(self) -> None:
        """Main trading execution loop"""
        logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                await self.trading_engine.run_trading_cycle()
                
                # Wait 5 minutes between cycles
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Error in trading loop", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _portfolio_monitoring_loop(self) -> None:
        """Portfolio monitoring and snapshot creation"""
        logger.info("Starting portfolio monitoring loop")
        
        while self.is_running:
            try:
                await self._create_portfolio_snapshot()
                
                # Create snapshot every 15 minutes
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error("Error in portfolio monitoring", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _model_retraining_loop(self) -> None:
        """Model retraining check loop"""
        logger.info("Starting model retraining loop")
        
        while self.is_running:
            try:
                if self.trading_engine.model_manager.should_retrain():
                    logger.info("Triggering scheduled model retraining")
                    
                    for symbol in settings.trading.trading_pairs:
                        df = await self.trading_engine.get_market_data(symbol, limit=5000)
                        if not df.empty:
                            results = self.trading_engine.model_manager.train_models(df)
                            logger.info("Scheduled retraining completed", symbol=symbol, results=list(results.keys()))
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error("Error in model retraining loop", error=str(e))
                await asyncio.sleep(1800)  # Wait 30 minutes before retry
    
    async def _data_collection_loop(self) -> None:
        """Data collection and storage loop"""
        logger.info("Starting data collection loop")
        
        while self.is_running:
            try:
                await self._collect_market_data()
                
                # Collect data every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Error in data collection", error=str(e))
                await asyncio.sleep(180)  # Wait 3 minutes before retry
    
    async def _create_portfolio_snapshot(self) -> None:
        """Create portfolio snapshot for performance tracking"""
        try:
            from app.exchange.kraken_client import KrakenClient
            
            async with KrakenClient() as client:
                # Get account balances
                balances = await client.get_account_balance()
                trade_balance = await client.get_trade_balance()
                
                total_value = trade_balance.get('equity', 0.0)
                cash_balance = sum(float(b.available) for b in balances if b.currency in ['USD', 'EUR'])
                positions_value = trade_balance.get('cost_basis', 0.0)
                unrealized_pnl = trade_balance.get('unrealized_pnl', 0.0)
                
                # Calculate realized P&L from recent trades
                from app.database.models import SessionLocal, Trade
                db = SessionLocal()
                try:
                    today = datetime.now().date()
                    today_start = datetime.combine(today, datetime.min.time())
                    
                    daily_trades = db.query(Trade).filter(
                        Trade.timestamp >= today_start,
                        Trade.status == 'filled'
                    ).all()
                    
                    realized_pnl = sum(trade.pnl or 0.0 for trade in daily_trades)
                finally:
                    db.close()
                
                # Create snapshot
                DatabaseManager.create_portfolio_snapshot(
                    total_value=total_value,
                    cash_balance=cash_balance,
                    positions_value=positions_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl
                )
                
                logger.debug("Portfolio snapshot created", 
                           total_value=total_value, 
                           unrealized_pnl=unrealized_pnl)
                
        except Exception as e:
            logger.error("Failed to create portfolio snapshot", error=str(e))
    
    async def _collect_market_data(self) -> None:
        """Collect and store market data"""
        try:
            from app.exchange.kraken_client import KrakenClient
            
            async with KrakenClient() as client:
                for symbol in settings.trading.trading_pairs:
                    try:
                        # Get recent OHLC data
                        ohlc_data = await client.get_ohlc(symbol, interval=1)
                        
                        if symbol in ohlc_data:
                            # Store in database
                            DatabaseManager.store_market_data(symbol, ohlc_data[symbol][-100:])
                            
                            logger.debug("Market data collected", symbol=symbol, points=len(ohlc_data[symbol]))
                    
                    except Exception as e:
                        logger.error("Failed to collect market data", symbol=symbol, error=str(e))
                        
        except Exception as e:
            logger.error("Market data collection failed", error=str(e))