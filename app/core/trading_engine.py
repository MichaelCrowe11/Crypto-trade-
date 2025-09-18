"""
Trading engine with ML predictions, risk management, and safety controls
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from enum import Enum
import pandas as pd

from app.core.config import settings
from app.core.logging import setup_logging, log_trade_event, log_error_with_context
from app.exchange.kraken_client import KrakenClient, Balance, Order
from app.ml.models import ModelManager
from app.database.models import Trade, Position, RiskEvent


logger = setup_logging(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradingSignal:
    """Trading signal with confidence and reasoning"""
    
    def __init__(
        self,
        symbol: str,
        signal: SignalType,
        confidence: float,
        price_prediction: float,
        current_price: float,
        reasoning: Dict[str, any] = None
    ):
        self.symbol = symbol
        self.signal = signal
        self.confidence = confidence
        self.price_prediction = price_prediction
        self.current_price = current_price
        self.reasoning = reasoning or {}
        self.timestamp = datetime.now()


class RiskManager:
    """Risk management system with multiple safety checks"""
    
    def __init__(self):
        """Initialize risk manager"""
        self.daily_pnl = 0.0
        self.open_positions = {}
        self.daily_trade_count = 0
        self.last_reset_date = datetime.now().date()
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trade_count = 0
            self.last_reset_date = current_date
            logger.info("Daily risk metrics reset")
    
    def check_position_size(self, symbol: str, amount: float, portfolio_value: float) -> bool:
        """
        Check if position size is within limits
        
        Args:
            symbol: Trading pair symbol
            amount: Position amount in base currency
            portfolio_value: Total portfolio value
            
        Returns:
            True if position size is acceptable
        """
        position_value = amount
        position_ratio = position_value / portfolio_value if portfolio_value > 0 else 1.0
        
        if position_ratio > settings.trading.risk.max_position_size:
            logger.warning(
                "Position size limit exceeded",
                symbol=symbol,
                ratio=position_ratio,
                limit=settings.trading.risk.max_position_size
            )
            return False
        
        return True
    
    def check_daily_loss_limit(self, portfolio_value: float) -> bool:
        """
        Check if daily loss limit is exceeded
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            True if within daily loss limit
        """
        loss_ratio = abs(self.daily_pnl) / portfolio_value if portfolio_value > 0 else 0.0
        
        if self.daily_pnl < 0 and loss_ratio > settings.trading.risk.max_daily_loss:
            logger.warning(
                "Daily loss limit exceeded",
                daily_pnl=self.daily_pnl,
                loss_ratio=loss_ratio,
                limit=settings.trading.risk.max_daily_loss
            )
            return False
        
        return True
    
    def check_max_positions(self) -> bool:
        """Check if maximum number of open positions is exceeded"""
        if len(self.open_positions) >= settings.trading.risk.max_open_positions:
            logger.warning(
                "Maximum open positions reached",
                current=len(self.open_positions),
                limit=settings.trading.risk.max_open_positions
            )
            return False
        
        return True
    
    def check_trade_frequency(self) -> bool:
        """Check if trade frequency limit is exceeded"""
        if self.daily_trade_count >= settings.compliance.max_trade_frequency:
            logger.warning(
                "Daily trade frequency limit exceeded",
                count=self.daily_trade_count,
                limit=settings.compliance.max_trade_frequency
            )
            return False
        
        return True
    
    def calculate_position_size(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_price: float
    ) -> float:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_price: Current asset price
            
        Returns:
            Position size in base currency units
        """
        # Base position size as percentage of portfolio
        base_position_value = portfolio_value * settings.trading.risk.max_position_size
        
        # Adjust based on signal confidence
        confidence_adjustment = signal.confidence * 0.5 + 0.5  # Scale to 0.5-1.0
        adjusted_position_value = base_position_value * confidence_adjustment
        
        # Convert to asset units
        position_size = adjusted_position_value / current_price
        
        return position_size
    
    def should_execute_trade(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        position_size: float
    ) -> Tuple[bool, str]:
        """
        Comprehensive trade execution check
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            position_size: Proposed position size
            
        Returns:
            Tuple of (should_execute, reason)
        """
        self.reset_daily_metrics()
        
        # Check if trading is enabled
        if not settings.trading.enabled:
            return False, "Trading disabled in configuration"
        
        # Check signal confidence threshold
        if signal.confidence < 0.6:  # Minimum 60% confidence
            return False, f"Signal confidence too low: {signal.confidence:.2f}"
        
        # Check position size limits
        if not self.check_position_size(signal.symbol, position_size * signal.current_price, portfolio_value):
            return False, "Position size limit exceeded"
        
        # Check daily loss limit
        if not self.check_daily_loss_limit(portfolio_value):
            return False, "Daily loss limit exceeded"
        
        # Check maximum positions
        if signal.signal != SignalType.SELL and not self.check_max_positions():
            return False, "Maximum open positions reached"
        
        # Check trade frequency
        if not self.check_trade_frequency():
            return False, "Daily trade frequency limit exceeded"
        
        return True, "All risk checks passed"


class TradingEngine:
    """Main trading engine with ML integration"""
    
    def __init__(self):
        """Initialize trading engine"""
        self.kraken_client = KrakenClient()
        self.model_manager = ModelManager()
        self.risk_manager = RiskManager()
        self.is_running = False
        self.market_data_cache = {}
    
    async def initialize(self) -> None:
        """Initialize trading engine components"""
        logger.info("Initializing trading engine")
        
        # Initialize ML models
        self.model_manager.initialize_models()
        
        # Load existing models if available
        for model in self.model_manager.models.values():
            model.load_model()
        
        logger.info("Trading engine initialized", mode=settings.trading.mode)
    
    async def get_market_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Get historical market data for analysis
        
        Args:
            symbol: Trading pair symbol
            limit: Number of data points to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            async with self.kraken_client:
                # Get OHLC data (1-minute intervals)
                ohlc_data = await self.kraken_client.get_ohlc(symbol, interval=1)
                
                # Convert to DataFrame
                if symbol in ohlc_data:
                    data = ohlc_data[symbol]
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                    ])
                    
                    # Convert timestamp and numeric columns
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Cache the data
                    self.market_data_cache[symbol] = df
                    
                    return df
                else:
                    logger.error("No OHLC data received", symbol=symbol)
                    return pd.DataFrame()
                    
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "get_market_data", "symbol": symbol})
            return pd.DataFrame()
    
    async def generate_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal using ML models
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Trading signal or None if insufficient data
        """
        try:
            # Get market data
            df = await self.get_market_data(symbol)
            if df.empty or len(df) < 100:
                logger.warning("Insufficient market data for signal generation", symbol=symbol)
                return None
            
            # Get current price
            async with self.kraken_client:
                ticker_data = await self.kraken_client.get_ticker([symbol])
                if symbol not in ticker_data:
                    logger.error("Failed to get current price", symbol=symbol)
                    return None
                
                current_price = float(ticker_data[symbol]['c'][0])  # Last trade price
            
            # Get ML predictions
            predictions = self.model_manager.get_predictions(df)
            if not predictions:
                logger.warning("No ML predictions available", symbol=symbol)
                return None
            
            # Calculate average prediction and confidence
            pred_values = list(predictions.values())
            avg_prediction = sum(pred_values) / len(pred_values)
            
            # Calculate confidence based on prediction agreement
            prediction_std = pd.Series(pred_values).std()
            confidence = max(0.0, min(1.0, 1.0 - (prediction_std / current_price)))
            
            # Determine signal
            price_change_threshold = 0.02  # 2% threshold
            predicted_change = (avg_prediction - current_price) / current_price
            
            if predicted_change > price_change_threshold:
                signal_type = SignalType.BUY
            elif predicted_change < -price_change_threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                confidence=confidence,
                price_prediction=avg_prediction,
                current_price=current_price,
                reasoning={
                    "predictions": predictions,
                    "predicted_change": predicted_change,
                    "confidence_calculation": f"1.0 - ({prediction_std:.4f} / {current_price:.2f})"
                }
            )
            
            logger.info(
                "Trading signal generated",
                symbol=symbol,
                signal=signal_type.value,
                confidence=confidence,
                prediction=avg_prediction,
                current_price=current_price
            )
            
            return signal
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "generate_trading_signal", "symbol": symbol})
            return None
    
    async def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            True if trade executed successfully
        """
        try:
            # Get portfolio information
            async with self.kraken_client:
                balances = await self.kraken_client.get_account_balance()
                trade_balance = await self.kraken_client.get_trade_balance()
                
                portfolio_value = trade_balance.get('equity', 0.0)
                
                if portfolio_value <= 0:
                    logger.error("Invalid portfolio value", value=portfolio_value)
                    return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal, portfolio_value, signal.current_price
            )
            
            # Risk management checks
            should_execute, reason = self.risk_manager.should_execute_trade(
                signal, portfolio_value, position_size
            )
            
            if not should_execute:
                logger.info("Trade rejected by risk management", symbol=signal.symbol, reason=reason)
                return False
            
            # Execute trade
            async with self.kraken_client:
                if signal.signal == SignalType.BUY:
                    order_result = await self.kraken_client.place_order(
                        pair=signal.symbol,
                        side='buy',
                        order_type='market',
                        volume=position_size
                    )
                elif signal.signal == SignalType.SELL:
                    # For sell signals, we sell existing position
                    order_result = await self.kraken_client.place_order(
                        pair=signal.symbol,
                        side='sell',
                        order_type='market',
                        volume=position_size
                    )
                else:
                    return False  # HOLD signal
            
            # Update risk manager
            self.risk_manager.daily_trade_count += 1
            
            # Log trade execution
            log_trade_event(
                logger,
                "order_placed",
                signal.symbol,
                side=signal.signal.value,
                size=position_size,
                price=signal.current_price,
                confidence=signal.confidence,
                order_id=order_result.get('txid', ['unknown'])[0]
            )
            
            return True
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "execute_trade", "symbol": signal.symbol})
            return False
    
    async def monitor_positions(self) -> None:
        """Monitor open positions for stop-loss and take-profit"""
        try:
            async with self.kraken_client:
                open_orders = await self.kraken_client.get_open_orders()
                
                for order in open_orders:
                    # Check if we need to set stop-loss or take-profit orders
                    await self.manage_position_risk(order)
                    
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "monitor_positions"})
    
    async def manage_position_risk(self, order: Order) -> None:
        """
        Manage risk for individual position
        
        Args:
            order: Open order/position to manage
        """
        try:
            # Get current price
            async with self.kraken_client:
                ticker_data = await self.kraken_client.get_ticker([order.symbol])
                if order.symbol not in ticker_data:
                    return
                
                current_price = float(ticker_data[order.symbol]['c'][0])
            
            # Calculate stop-loss and take-profit levels
            if order.side == 'buy':
                stop_loss_price = order.price * (1 - settings.trading.risk.stop_loss)
                take_profit_price = order.price * (1 + settings.trading.risk.take_profit)
                
                if current_price <= stop_loss_price:
                    # Trigger stop-loss
                    await self.trigger_stop_loss(order, current_price)
                elif current_price >= take_profit_price:
                    # Trigger take-profit
                    await self.trigger_take_profit(order, current_price)
            
            elif order.side == 'sell':
                stop_loss_price = order.price * (1 + settings.trading.risk.stop_loss)
                take_profit_price = order.price * (1 - settings.trading.risk.take_profit)
                
                if current_price >= stop_loss_price:
                    # Trigger stop-loss for short position
                    await self.trigger_stop_loss(order, current_price)
                elif current_price <= take_profit_price:
                    # Trigger take-profit for short position
                    await self.trigger_take_profit(order, current_price)
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "manage_position_risk", "order_id": order.order_id})
    
    async def trigger_stop_loss(self, order: Order, current_price: float) -> None:
        """Trigger stop-loss for position"""
        try:
            async with self.kraken_client:
                # Close position with market order
                close_side = 'sell' if order.side == 'buy' else 'buy'
                close_result = await self.kraken_client.place_order(
                    pair=order.symbol,
                    side=close_side,
                    order_type='market',
                    volume=order.remaining
                )
            
            # Update daily PnL
            pnl = (current_price - order.price) * order.amount if order.side == 'buy' else (order.price - current_price) * order.amount
            self.risk_manager.daily_pnl += pnl
            
            log_trade_event(
                logger,
                "stop_loss_triggered",
                order.symbol,
                order_id=order.order_id,
                entry_price=order.price,
                exit_price=current_price,
                pnl=pnl
            )
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "trigger_stop_loss", "order_id": order.order_id})
    
    async def trigger_take_profit(self, order: Order, current_price: float) -> None:
        """Trigger take-profit for position"""
        try:
            async with self.kraken_client:
                # Close position with market order
                close_side = 'sell' if order.side == 'buy' else 'buy'
                close_result = await self.kraken_client.place_order(
                    pair=order.symbol,
                    side=close_side,
                    order_type='market',
                    volume=order.remaining
                )
            
            # Update daily PnL
            pnl = (current_price - order.price) * order.amount if order.side == 'buy' else (order.price - current_price) * order.amount
            self.risk_manager.daily_pnl += pnl
            
            log_trade_event(
                logger,
                "take_profit_triggered",
                order.symbol,
                order_id=order.order_id,
                entry_price=order.price,
                exit_price=current_price,
                pnl=pnl
            )
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "trigger_take_profit", "order_id": order.order_id})
    
    async def run_trading_cycle(self) -> None:
        """Run one complete trading cycle"""
        logger.info("Starting trading cycle")
        
        try:
            # Check if models need retraining
            if self.model_manager.should_retrain():
                logger.info("Retraining ML models")
                
                # Get training data for each trading pair
                for symbol in settings.trading.trading_pairs:
                    df = await self.get_market_data(symbol, limit=5000)  # More data for training
                    if not df.empty:
                        training_results = self.model_manager.train_models(df)
                        logger.info("Model training completed", symbol=symbol, results=list(training_results.keys()))
            
            # Generate and execute trading signals
            for symbol in settings.trading.trading_pairs:
                signal = await self.generate_trading_signal(symbol)
                
                if signal and signal.signal != SignalType.HOLD:
                    success = await self.execute_trade(signal)
                    if success:
                        logger.info("Trade executed successfully", symbol=symbol, signal=signal.signal.value)
                    else:
                        logger.warning("Trade execution failed", symbol=symbol)
            
            # Monitor existing positions
            await self.monitor_positions()
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "run_trading_cycle"})
        
        logger.info("Trading cycle completed")
    
    async def start(self) -> None:
        """Start the trading engine"""
        await self.initialize()
        self.is_running = True
        
        logger.info("Trading engine started", 
                   enabled=settings.trading.enabled, 
                   mode=settings.trading.mode)
        
        while self.is_running:
            try:
                await self.run_trading_cycle()
                
                # Wait before next cycle (e.g., 5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                log_error_with_context(logger, e, {"operation": "trading_engine_main_loop"})
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop(self) -> None:
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Trading engine stopped")