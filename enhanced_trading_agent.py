import asyncio
import os
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP, WebSocket
import openai
from typing import Dict, List, Optional, Tuple

class EnhancedTradingAgent:
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.bybit_client = self._init_bybit_client()
        self.openai_client = self._init_openai_client()
        self.market_state = {}
        self.active_positions = {}
        self.trade_signals = {}
        self.ws_public = None
        self.ws_private = None
        self.last_analysis_time = {}
        self.min_profit_multiplier = 3.0  # Minimum profit target (3x fees)
        self.max_loss_percentage = 0.02   # 2% max loss per trade
        self.team_agents = self._init_team_agents()

    def _setup_logging(self) -> logging.Logger:
        """Initialize enhanced logging with detailed formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s - [%(filename)s:%(lineno)d]',
            handlers=[
                logging.FileHandler('enhanced_trading.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('EnhancedTrader')

    def _load_config(self, config_path: str) -> dict:
        """Load configuration with fallback to environment variables"""
        if not config_path:
            # Use environment variables with secure fallbacks
            return {
                'bybit_api_key': os.getenv('BYBIT_API_KEY'),
                'bybit_api_secret': os.getenv('BYBIT_API_SECRET'),
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                'trading_pairs': ['BTCUSDT', 'ETHUSDT'],  # Default pairs
                'timeframes': ['1', '3', '5'],  # Multiple timeframes in minutes
                'risk_percentage': 0.01,  # 1% risk per trade
                'leverage': 10,  # Default leverage
                'testnet': False  # Production mode
            }
        
        with open(config_path, 'r') as f:
            return json.load(f)

    def _init_bybit_client(self):
        """Initialize Bybit client with optimized settings"""
        return HTTP(
            api_key=self.config['bybit_api_key'],
            api_secret=self.config['bybit_api_secret'],
            testnet=self.config.get('testnet', False),
            retry_codes=[10002, 10006],  # Retry on rate limit and temporary errors
            retry_delay=0.1,  # Fast retry for HFT
            max_retries=3
        )

    def _init_openai_client(self):
        """Initialize OpenAI client for advanced market analysis"""
        openai.api_key = self.config['openai_api_key']
        return openai.Client()

    def _init_team_agents(self) -> dict:
        """Initialize specialized trading agents"""
        return {
            'market_analyst': self.MarketAnalyst(self),
            'risk_manager': self.RiskManager(self),
            'execution_engine': self.ExecutionEngine(self),
            'portfolio_manager': self.PortfolioManager(self)
        }

    class MarketAnalyst:
        """Specialized agent for market analysis and prediction"""
        def __init__(self, parent):
            self.parent = parent
            self.logger = parent.logger

        async def analyze_market(self, symbol: str, timeframe: str) -> dict:
            """Perform comprehensive market analysis using AI"""
            try:
                # Get recent market data
                klines = await self.parent._get_klines(symbol, timeframe, limit=100)
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame(klines)
                df['close'] = pd.to_numeric(df['close'])
                df['volume'] = pd.to_numeric(df['volume'])
                
                # Calculate technical indicators
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['volatility'] = df['close'].rolling(window=20).std()
                
                # Prepare market context for AI analysis
                market_context = {
                    'price_change_pct': float(df['close'].pct_change().tail(1).values[0]),
                    'volume_change_pct': float(df['volume'].pct_change().tail(1).values[0]),
                    'volatility': float(df['volatility'].tail(1).values[0]),
                    'trend_direction': 'bullish' if df['sma_20'].tail(1).values[0] < df['close'].tail(1).values[0] else 'bearish'
                }
                
                # Get AI insights
                ai_analysis = await self._get_ai_analysis(market_context)
                
                return {
                    'technical_analysis': market_context,
                    'ai_insights': ai_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.error(f"Market analysis failed: {str(e)}")
                return {}

        async def _get_ai_analysis(self, market_data: dict) -> dict:
            """Get AI-powered market insights"""
            try:
                response = await self.parent.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are an expert crypto market analyst. Analyze the market data and provide trading insights."},
                        {"role": "user", "content": f"Analyze this market data and provide trading recommendations: {json.dumps(market_data)}"}
                    ],
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                return {'sentiment': self._extract_sentiment(analysis), 'recommendation': analysis}
            except Exception as e:
                self.logger.error(f"AI analysis failed: {str(e)}")
                return {}

        def _extract_sentiment(self, analysis: str) -> float:
            """Extract numerical sentiment from AI analysis"""
            if 'bullish' in analysis.lower():
                return 0.8
            elif 'bearish' in analysis.lower():
                return -0.8
            return 0.0

    class RiskManager:
        """Advanced risk management and position sizing"""
        def __init__(self, parent):
            self.parent = parent
            self.logger = parent.logger
            self.position_limits = {}

        async def calculate_position_size(self, symbol: str, entry_price: float) -> Tuple[float, dict]:
            """Calculate optimal position size based on risk parameters"""
            try:
                # Get account balance
                wallet = await self.parent.bybit_client.get_wallet_balance(accountType="UNIFIED")
                available_balance = float(wallet['result']['list'][0]['availableBalance'])
                
                # Calculate position size based on risk percentage
                risk_amount = available_balance * self.parent.config['risk_percentage']
                
                # Apply sophisticated position sizing based on market conditions
                position_size = self._calculate_dynamic_position_size(risk_amount, entry_price)
                
                # Apply risk limits
                position_size = min(position_size, self._get_max_position_size(symbol))
                
                return position_size, {
                    'risk_amount': risk_amount,
                    'leverage_used': self.parent.config['leverage'],
                    'max_loss_usd': risk_amount
                }
            except Exception as e:
                self.logger.error(f"Position size calculation failed: {str(e)}")
                return 0.0, {}

        def _calculate_dynamic_position_size(self, risk_amount: float, entry_price: float) -> float:
            """Calculate position size with dynamic adjustments"""
            # Base position size
            base_size = (risk_amount * self.parent.config['leverage']) / entry_price
            
            # Apply volatility adjustment
            volatility_factor = self._get_volatility_factor()
            adjusted_size = base_size * volatility_factor
            
            return adjusted_size

        def _get_volatility_factor(self) -> float:
            """Calculate volatility adjustment factor"""
            # Implement sophisticated volatility calculation
            return 0.8  # Conservative default

        def _get_max_position_size(self, symbol: str) -> float:
            """Get maximum allowed position size for a symbol"""
            # Implement position limits based on liquidity
            return float('inf')  # Temporary unlimited

    class ExecutionEngine:
        """Handles order execution and management"""
        def __init__(self, parent):
            self.parent = parent
            self.logger = parent.logger
            self.order_queue = asyncio.Queue()

        async def execute_trade(self, symbol: str, side: str, size: float, price: float = None) -> dict:
            """Execute trade with sophisticated order types"""
            try:
                # Determine order type and price
                order_type = "Market" if price is None else "Limit"
                
                # Calculate take profit and stop loss
                tp_price = self._calculate_take_profit(price, side)
                sl_price = self._calculate_stop_loss(price, side)
                
                # Place the order
                order = await self.parent.bybit_client.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType=order_type,
                    qty=str(size),
                    price=str(price) if price else None,
                    takeProfit=str(tp_price),
                    stopLoss=str(sl_price),
                    timeInForce="GTC",
                    reduce_only=False,
                    close_on_trigger=False
                )
                
                self.logger.info(f"Order placed: {order}")
                return order
            except Exception as e:
                self.logger.error(f"Trade execution failed: {str(e)}")
                return {}

        def _calculate_take_profit(self, entry_price: float, side: str) -> float:
            """Calculate take profit level"""
            multiplier = self.parent.min_profit_multiplier
            return entry_price * (1 + multiplier/100) if side == "Buy" else entry_price * (1 - multiplier/100)

        def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
            """Calculate stop loss level"""
            return entry_price * (1 - self.parent.max_loss_percentage) if side == "Buy" else entry_price * (1 + self.parent.max_loss_percentage)

    class PortfolioManager:
        """Manages overall portfolio and strategy coordination"""
        def __init__(self, parent):
            self.parent = parent
            self.logger = parent.logger
            self.portfolio_stats = {}

        async def update_portfolio_stats(self):
            """Update portfolio statistics"""
            try:
                positions = await self.parent.bybit_client.get_positions(category="linear")
                wallet = await self.parent.bybit_client.get_wallet_balance(accountType="UNIFIED")
                
                self.portfolio_stats = {
                    'total_equity': float(wallet['result']['list'][0]['totalEquity']),
                    'available_balance': float(wallet['result']['list'][0]['availableBalance']),
                    'positions': positions['result']['list'],
                    'timestamp': datetime.now().isoformat()
                }
                
                return self.portfolio_stats
            except Exception as e:
                self.logger.error(f"Portfolio update failed: {str(e)}")
                return {}

        async def monitor_portfolio_risk(self) -> bool:
            """Monitor overall portfolio risk"""
            try:
                stats = await self.update_portfolio_stats()
                
                # Calculate key risk metrics
                utilized_margin = stats['total_equity'] - stats['available_balance']
                margin_ratio = utilized_margin / stats['total_equity'] if stats['total_equity'] > 0 else 0
                
                # Check risk thresholds
                if margin_ratio > 0.8:  # 80% margin utilized
                    self.logger.warning("High margin utilization detected")
                    return False
                
                return True
            except Exception as e:
                self.logger.error(f"Risk monitoring failed: {str(e)}")
                return False

    async def start(self):
        """Start the enhanced trading agent"""
        self.logger.info("Starting Enhanced Trading Agent...")
        
        try:
            # Initialize WebSocket connections
            await self._init_websockets()
            
            # Start main trading loop
            while True:
                for symbol in self.config['trading_pairs']:
                    try:
                        # Run team of agents
                        await self._run_trading_cycle(symbol)
                        
                        # Monitor and adjust portfolio
                        if not await self.team_agents['portfolio_manager'].monitor_portfolio_risk():
                            self.logger.warning(f"Risk threshold exceeded for {symbol}, skipping trading cycle")
                            continue
                        
                    except Exception as e:
                        self.logger.error(f"Error in trading cycle for {symbol}: {str(e)}")
                
                # Fast cycle for real-time trading
                await asyncio.sleep(1)  # 1-second cycle time
                
        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {str(e)}")
        finally:
            await self._cleanup()

    async def _run_trading_cycle(self, symbol: str):
        """Execute a complete trading cycle"""
        # Get market analysis from Market Analyst
        analysis = await self.team_agents['market_analyst'].analyze_market(symbol, self.config['timeframes'][0])
        
        if not analysis:
            return
        
        # Check if signal is strong enough
        if abs(analysis.get('ai_insights', {}).get('sentiment', 0)) < 0.5:
            return
        
        # Determine trade direction
        side = "Buy" if analysis['ai_insights']['sentiment'] > 0 else "Sell"
        
        # Get current market price
        ticker = await self.bybit_client.get_tickers(category="linear", symbol=symbol)
        current_price = float(ticker['result']['list'][0]['lastPrice'])
        
        # Calculate position size with Risk Manager
        size, risk_params = await self.team_agents['risk_manager'].calculate_position_size(symbol, current_price)
        
        if size <= 0:
            return
        
        # Execute trade through Execution Engine
        await self.team_agents['execution_engine'].execute_trade(symbol, side, size, current_price)

    async def _init_websockets(self):
        """Initialize WebSocket connections"""
        # Public WebSocket
        self.ws_public = WebSocket(
            testnet=self.config.get('testnet', False),
            channel_type="linear"
        )
        
        # Private WebSocket
        self.ws_private = WebSocket(
            testnet=self.config.get('testnet', False),
            channel_type="private",
            api_key=self.config['bybit_api_key'],
            api_secret=self.config['bybit_api_secret']
        )
        
        # Subscribe to relevant channels
        for symbol in self.config['trading_pairs']:
            self.ws_public.kline_stream(
                symbol=symbol,
                interval="1",
                callback=self._handle_kline_update
            )

    async def _handle_kline_update(self, message):
        """Handle real-time kline updates"""
        try:
            data = message['data']
            symbol = data['symbol']
            
            # Update market state
            self.market_state[symbol] = {
                'last_price': float(data['close']),
                'volume': float(data['volume']),
                'timestamp': data['timestamp']
            }
            
            # Check for trading opportunities
            await self._check_trading_opportunity(symbol)
            
        except Exception as e:
            self.logger.error(f"Error handling kline update: {str(e)}")

    async def _check_trading_opportunity(self, symbol: str):
        """Check for trading opportunities on price updates"""
        # Implement real-time trading logic here
        pass

    async def _cleanup(self):
        """Cleanup resources"""
        if self.ws_public:
            await self.ws_public.close()
        if self.ws_private:
            await self.ws_private.close()
        self.logger.info("Trading agent stopped")

if __name__ == "__main__":
    # Initialize and start the trading agent
    agent = EnhancedTradingAgent()
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        logging.info("Trading agent stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
