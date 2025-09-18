"""
Kraken exchange client with WebSocket support and comprehensive API coverage
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import aiohttp
import websockets
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import setup_logging, log_error_with_context


logger = setup_logging(__name__)


class OrderBook(BaseModel):
    """Order book data model"""
    symbol: str
    bids: List[Tuple[float, float]]  # price, volume
    asks: List[Tuple[float, float]]  # price, volume
    timestamp: datetime


class Trade(BaseModel):
    """Trade data model"""
    symbol: str
    side: str  # buy/sell
    amount: float
    price: float
    timestamp: datetime
    trade_id: str


class Balance(BaseModel):
    """Account balance model"""
    currency: str
    balance: float
    available: float
    reserved: float


class Order(BaseModel):
    """Order model"""
    order_id: str
    symbol: str
    side: str
    type: str
    amount: float
    price: Optional[float]
    status: str
    timestamp: datetime
    filled: float = 0.0
    remaining: float = 0.0


class KrakenClient:
    """
    Kraken exchange client with REST API and WebSocket support
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Kraken client
        
        Args:
            api_key: Kraken API key
            secret_key: Kraken secret key
        """
        self.api_key = api_key or settings.kraken_api_key
        self.secret_key = secret_key or settings.kraken_secret_key
        self.base_url = settings.exchange.api_url
        self.ws_url = settings.exchange.websocket_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self._rate_limiter = asyncio.Semaphore(1)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.ws_connection:
            await self.ws_connection.close()
    
    def _generate_signature(self, endpoint: str, data: Dict[str, Any]) -> str:
        """
        Generate API signature for authenticated requests
        
        Args:
            endpoint: API endpoint path
            data: Request data
            
        Returns:
            HMAC signature
        """
        if not self.secret_key:
            raise ValueError("Secret key required for authenticated requests")
        
        # Create nonce
        data['nonce'] = str(int(time.time() * 1000))
        
        # Encode data
        encoded_data = urllib.parse.urlencode(data)
        
        # Create message
        message = endpoint + hashlib.sha256(
            (data['nonce'] + encoded_data).encode()
        ).digest()
        
        # Generate signature
        signature = hmac.new(
            base64.b64decode(self.secret_key),
            message,
            hashlib.sha512
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict[str, Any] = None,
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Kraken API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            authenticated: Whether request requires authentication
            
        Returns:
            API response data
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if authenticated:
            if not self.api_key or not self.secret_key:
                raise ValueError("API credentials required for authenticated requests")
            
            data = data or {}
            signature = self._generate_signature(endpoint, data.copy())
            
            headers.update({
                'API-Key': self.api_key,
                'API-Sign': signature,
                'Content-Type': 'application/x-www-form-urlencoded'
            })
        
        # Rate limiting
        async with self._rate_limiter:
            await asyncio.sleep(1.0 / settings.exchange.rate_limit)
            
            if method.upper() == 'GET':
                async with self.session.get(url, params=data, headers=headers) as response:
                    result = await response.json()
            else:
                async with self.session.post(url, data=data, headers=headers) as response:
                    result = await response.json()
        
        # Check for API errors
        if 'error' in result and result['error']:
            error_msg = ', '.join(result['error'])
            logger.error("Kraken API error", error=error_msg, endpoint=endpoint)
            raise Exception(f"Kraken API error: {error_msg}")
        
        return result.get('result', {})
    
    async def get_server_time(self) -> datetime:
        """Get server time"""
        result = await self._make_request('GET', '/0/public/Time')
        return datetime.fromtimestamp(result['unixtime'])
    
    async def get_asset_info(self, assets: List[str] = None) -> Dict[str, Any]:
        """Get asset information"""
        data = {}
        if assets:
            data['asset'] = ','.join(assets)
        
        return await self._make_request('GET', '/0/public/Assets', data)
    
    async def get_tradable_pairs(self, pairs: List[str] = None) -> Dict[str, Any]:
        """Get tradable asset pairs"""
        data = {}
        if pairs:
            data['pair'] = ','.join(pairs)
        
        return await self._make_request('GET', '/0/public/AssetPairs', data)
    
    async def get_ticker(self, pairs: List[str]) -> Dict[str, Any]:
        """Get ticker information for trading pairs"""
        data = {'pair': ','.join(pairs)}
        return await self._make_request('GET', '/0/public/Ticker', data)
    
    async def get_ohlc(
        self,
        pair: str,
        interval: int = 1,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get OHLC data
        
        Args:
            pair: Trading pair
            interval: Time frame interval in minutes
            since: Return data since given timestamp
        """
        data = {'pair': pair, 'interval': interval}
        if since:
            data['since'] = since
        
        return await self._make_request('GET', '/0/public/OHLC', data)
    
    async def get_order_book(self, pair: str, count: int = 100) -> OrderBook:
        """Get order book for trading pair"""
        data = {'pair': pair, 'count': count}
        result = await self._make_request('GET', '/0/public/Depth', data)
        
        # Parse order book data
        pair_data = list(result.values())[0]
        
        return OrderBook(
            symbol=pair,
            bids=[(float(price), float(volume)) for price, volume, _ in pair_data['bids']],
            asks=[(float(price), float(volume)) for price, volume, _ in pair_data['asks']],
            timestamp=datetime.now()
        )
    
    async def get_recent_trades(self, pair: str, since: Optional[int] = None) -> List[Trade]:
        """Get recent trades for trading pair"""
        data = {'pair': pair}
        if since:
            data['since'] = since
        
        result = await self._make_request('GET', '/0/public/Trades', data)
        
        # Parse trades data
        pair_data = list(result.values())[0]
        trades = []
        
        for trade_data in pair_data:
            price, volume, timestamp, side, order_type, _ = trade_data
            trades.append(Trade(
                symbol=pair,
                side='buy' if side == 'b' else 'sell',
                amount=float(volume),
                price=float(price),
                timestamp=datetime.fromtimestamp(float(timestamp)),
                trade_id=f"{pair}_{timestamp}_{price}_{volume}"
            ))
        
        return trades
    
    async def get_account_balance(self) -> List[Balance]:
        """Get account balance"""
        result = await self._make_request('POST', '/0/private/Balance', authenticated=True)
        
        balances = []
        for currency, balance in result.items():
            balances.append(Balance(
                currency=currency,
                balance=float(balance),
                available=float(balance),  # Kraken doesn't separate available/reserved in balance
                reserved=0.0
            ))
        
        return balances
    
    async def get_trade_balance(self, asset: str = 'USD') -> Dict[str, float]:
        """Get trade balance information"""
        data = {'asset': asset}
        result = await self._make_request('POST', '/0/private/TradeBalance', data, authenticated=True)
        
        return {
            'equity': float(result.get('e', 0)),
            'free_margin': float(result.get('mf', 0)),
            'margin_level': float(result.get('ml', 0)) if result.get('ml') else None,
            'unrealized_pnl': float(result.get('n', 0)),
            'cost_basis': float(result.get('c', 0)),
            'current_value': float(result.get('v', 0))
        }
    
    async def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        oflags: Optional[str] = None,
        starttm: Optional[int] = None,
        expiretm: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order
        
        Args:
            pair: Trading pair
            side: buy or sell
            order_type: market, limit, stop-loss, etc.
            volume: Order volume
            price: Order price (required for limit orders)
            leverage: Leverage ratio
            oflags: Order flags
            starttm: Start time for scheduled orders
            expiretm: Expiration time
        """
        data = {
            'pair': pair,
            'type': side,
            'ordertype': order_type,
            'volume': str(volume)
        }
        
        if price is not None:
            data['price'] = str(price)
        if leverage:
            data['leverage'] = str(leverage)
        if oflags:
            data['oflags'] = oflags
        if starttm:
            data['starttm'] = str(starttm)
        if expiretm:
            data['expiretm'] = str(expiretm)
        
        # Safety check for paper trading
        if settings.trading.mode == "paper":
            logger.info("Paper trading mode - order not actually placed", order_data=data)
            return {
                'txid': [f"paper_{int(time.time())}"],
                'descr': {'order': f"Paper {side} {volume} {pair} @ {price or 'market'}"}
            }
        
        return await self._make_request('POST', '/0/private/AddOrder', data, authenticated=True)
    
    async def cancel_order(self, txid: str) -> Dict[str, Any]:
        """Cancel an order"""
        data = {'txid': txid}
        
        if settings.trading.mode == "paper":
            logger.info("Paper trading mode - order cancellation simulated", txid=txid)
            return {'count': 1}
        
        return await self._make_request('POST', '/0/private/CancelOrder', data, authenticated=True)
    
    async def get_open_orders(self, trades: bool = False) -> List[Order]:
        """Get open orders"""
        data = {'trades': trades}
        result = await self._make_request('POST', '/0/private/OpenOrders', data, authenticated=True)
        
        orders = []
        for txid, order_data in result.get('open', {}).items():
            descr = order_data.get('descr', {})
            orders.append(Order(
                order_id=txid,
                symbol=descr.get('pair', ''),
                side=descr.get('type', ''),
                type=descr.get('ordertype', ''),
                amount=float(order_data.get('vol', 0)),
                price=float(descr.get('price', 0)) if descr.get('price') else None,
                status=order_data.get('status', ''),
                timestamp=datetime.fromtimestamp(float(order_data.get('opentm', 0))),
                filled=float(order_data.get('vol_exec', 0)),
                remaining=float(order_data.get('vol', 0)) - float(order_data.get('vol_exec', 0))
            ))
        
        return orders
    
    async def get_closed_orders(
        self,
        trades: bool = False,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Order]:
        """Get closed orders"""
        data = {'trades': trades}
        if start:
            data['start'] = start
        if end:
            data['end'] = end
        
        result = await self._make_request('POST', '/0/private/ClosedOrders', data, authenticated=True)
        
        orders = []
        for txid, order_data in result.get('closed', {}).items():
            descr = order_data.get('descr', {})
            orders.append(Order(
                order_id=txid,
                symbol=descr.get('pair', ''),
                side=descr.get('type', ''),
                type=descr.get('ordertype', ''),
                amount=float(order_data.get('vol', 0)),
                price=float(descr.get('price', 0)) if descr.get('price') else None,
                status=order_data.get('status', ''),
                timestamp=datetime.fromtimestamp(float(order_data.get('opentm', 0))),
                filled=float(order_data.get('vol_exec', 0)),
                remaining=0.0  # Closed orders are fully executed
            ))
        
        return orders
    
    async def start_websocket(self, pairs: List[str], channels: List[str]):
        """Start WebSocket connection for real-time data"""
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            
            # Subscribe to channels
            subscription = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": channels[0]} if len(channels) == 1 else {"name": channels}
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            logger.info("WebSocket subscription sent", pairs=pairs, channels=channels)
            
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "websocket_start"})
            raise
    
    async def listen_websocket(self, callback):
        """Listen for WebSocket messages"""
        if not self.ws_connection:
            raise RuntimeError("WebSocket not connected")
        
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                await callback(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            log_error_with_context(logger, e, {"operation": "websocket_listen"})
            raise