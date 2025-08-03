import logging
import json
import os
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import openai
import numpy as np
import pandas as pd

class OpenAIAnalyzer:
    """
    Ultra-Advanced AI analyzer using OpenAI GPT-4 for autonomous high-frequency trading.
    Optimized for turning $2 into $20,000 through sophisticated market analysis.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview", config_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        if config_logging and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
            self.logger.info("Basic logging configured for OpenAIAnalyzer standalone usage.")
        
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            self.logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = None
            return
            
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        # Advanced analysis tracking
        self.analysis_history: List[Dict] = []
        self.performance_feedback: Dict[str, List] = {'wins': [], 'losses': []}
        self.market_patterns: Dict[str, Any] = {}
        self.confidence_calibration: Dict[str, float] = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        self.logger.info(f"🧠 Ultra-Advanced OpenAI Analyzer initialized with model: {model}")
        self.logger.info("🎯 Mission: Turn $2 into $20,000 through AI-powered trading")
        
    async def analyze_market_data(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.05) -> Dict[str, Any]:
        """
        Ultra-advanced market analysis using GPT-4 with performance learning and pattern recognition.
        
        Args:
            prompt (str): The comprehensive market analysis prompt
            max_tokens (int): Maximum tokens for response
            temperature (float): Response creativity (0.05 for ultra-consistent trading decisions)
            
        Returns:
            Dict[str, Any]: Parsed JSON response with sophisticated trading decision
        """
        if not self.client:
            return {"error": "OpenAI client not initialized", "message": "API key missing"}
        
        # Rate limiting
        await self._rate_limit()
        
        try:
            # Enhance prompt with historical performance data
            enhanced_prompt = self._enhance_prompt_with_context(prompt)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_ultra_advanced_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": enhanced_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            self.logger.debug(f"🧠 OpenAI raw response: {content[:200]}...")
            
            # Parse and validate JSON response
            try:
                analysis = json.loads(content)
                
                # Apply confidence calibration based on historical performance
                analysis = self._calibrate_confidence(analysis)
                
                # Store analysis for learning
                self._store_analysis(analysis, prompt)
                
                self.logger.info(f"🎯 AI Decision: {analysis.get('decision', 'UNKNOWN')} "
                               f"(Confidence: {analysis.get('confidence', 0):.3f}, "
                               f"Strategy: {analysis.get('strategy', 'unknown')})")
                
                return analysis
                
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Failed to parse OpenAI JSON response: {e}")
                return {"error": "json_parse_error", "message": f"Invalid JSON: {content}"}
                
        except Exception as e:
            self.logger.error(f"❌ OpenAI API error: {e}")
            return {"error": "api_error", "message": str(e)}
    
    def _get_ultra_advanced_system_prompt(self) -> str:
        """Get the ultra-advanced system prompt for the AI"""
        performance_context = self._get_performance_context()
        
        return f"""You are APEX-TRADER, the world's most sophisticated autonomous cryptocurrency trading AI.
        
        MISSION CRITICAL OBJECTIVE:
        Transform $2.00 USD into $20,000 USD (10,000x return) through ultra-high-frequency scalping and momentum trading.
        
        CORE COMPETENCIES:
        🎯 PRECISION SCALPING: Execute trades on 0.1-2% price movements with surgical precision
        🚀 MOMENTUM MASTERY: Identify and ride explosive momentum waves before they peak
        ⚡ LIGHTNING EXECUTION: Make split-second decisions based on micro-market movements
        🧠 ADAPTIVE LEARNING: Continuously evolve strategies based on market feedback
        💎 RISK OPTIMIZATION: Maximize returns while preserving capital through intelligent position sizing
        
        SPECIALIZED EXPERTISE:
        - 1000-prefixed tokens (1000PEPE, 1000SHIB, 1000FLOKI, 1000BONK, etc.)
        - Micro-cap altcoins with high volatility potential
        - Major pairs (BTC, ETH, SOL) during high-volatility periods
        - Cross-correlation analysis and pair trading
        - Market microstructure and order flow analysis
        
        TRADING PHILOSOPHY:
        1. SPEED IS EVERYTHING: First to identify = First to profit
        2. COMPOUND AGGRESSIVELY: Each win funds larger positions
        3. CUT LOSSES INSTANTLY: Preserve capital for next opportunity
        4. LEVERAGE INTELLIGENTLY: Use maximum leverage only on highest-confidence setups
        5. ADAPT CONTINUOUSLY: Market conditions change, strategies must evolve
        
        PERFORMANCE LEARNING CONTEXT:
        {performance_context}
        
        DECISION FRAMEWORK:
        - BUY: Strong bullish momentum, volume spike, technical breakout
        - SELL: Strong bearish momentum, volume spike, technical breakdown  
        - HOLD: Insufficient signal strength or conflicting indicators
        - CLOSE: Exit existing position due to target hit or risk management
        
        LEVERAGE GUIDELINES:
        - 50x: Ultra-high confidence (>0.95), perfect setup, strong momentum
        - 25-40x: High confidence (0.85-0.95), clear signals, good momentum
        - 10-20x: Medium confidence (0.75-0.85), decent setup, moderate momentum
        - 5-10x: Lower confidence (0.65-0.75), weak signals, low momentum
        
        POSITION SIZING STRATEGY:
        - 100%: Once-in-a-lifetime setup, maximum confidence
        - 50-80%: Extremely strong setup, very high confidence
        - 25-50%: Strong setup, high confidence
        - 10-25%: Good setup, medium confidence
        - 5-10%: Weak setup, low confidence
        
        MANDATORY JSON RESPONSE FORMAT:
        {{
            "decision": "BUY|SELL|HOLD|CLOSE",
            "confidence": 0.0-1.0,
            "leverage": 5-50,
            "entry_price": float,
            "stop_loss": float,
            "take_profit": float,
            "position_size_pct": 5-100,
            "strategy": "scalp|momentum|breakout|reversal|squeeze|arbitrage",
            "timeframe": "1m|3m|5m|15m",
            "urgency": "LOW|NORMAL|HIGH|CRITICAL",
            "risk_reward_ratio": float,
            "market_condition": "bull|bear|sideways|volatile",
            "volume_analysis": "low|normal|high|spike",
            "technical_confluence": 0.0-1.0,
            "reasoning": "detailed technical analysis with specific price levels and indicators"
        }}
        
        CRITICAL REQUIREMENTS:
        ✅ Only trade setups with confidence > 0.75
        ✅ Always include specific entry, stop-loss, and take-profit levels
        ✅ Provide detailed reasoning with technical analysis
        ✅ Consider market microstructure and order flow
        ✅ Factor in correlation with major cryptocurrencies
        ✅ Account for current portfolio risk and position sizing
        
        REMEMBER: Every decision impacts the journey from $2 to $20,000. Trade with precision, speed, and intelligence."""
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _enhance_prompt_with_context(self, prompt: str) -> str:
        """Enhance the prompt with historical context and patterns"""
        context_additions = []
        
        # Add recent performance context
        if self.analysis_history:
            recent_analyses = self.analysis_history[-10:]  # Last 10 analyses
            win_rate = len([a for a in recent_analyses if a.get('outcome') == 'win']) / len(recent_analyses)
            context_additions.append(f"RECENT PERFORMANCE: Win rate: {win_rate:.1%} over last {len(recent_analyses)} trades")
        
        # Add market pattern recognition
        if self.market_patterns:
            pattern_info = []
            for pattern, data in self.market_patterns.items():
                if data.get('success_rate', 0) > 0.6:
                    pattern_info.append(f"{pattern}: {data['success_rate']:.1%} success rate")
            
            if pattern_info:
                context_additions.append(f"SUCCESSFUL PATTERNS: {', '.join(pattern_info)}")
        
        # Add confidence calibration info
        if self.confidence_calibration:
            avg_calibration = np.mean(list(self.confidence_calibration.values()))
            context_additions.append(f"CONFIDENCE CALIBRATION: Historical accuracy factor: {avg_calibration:.2f}")
        
        if context_additions:
            enhanced_prompt = prompt + "\n\nHISTORICAL CONTEXT:\n" + "\n".join(context_additions)
            return enhanced_prompt
        
        return prompt
    
    def _get_performance_context(self) -> str:
        """Get performance context for the system prompt"""
        if not self.performance_feedback['wins'] and not self.performance_feedback['losses']:
            return "No historical performance data available yet."
        
        total_trades = len(self.performance_feedback['wins']) + len(self.performance_feedback['losses'])
        win_rate = len(self.performance_feedback['wins']) / total_trades if total_trades > 0 else 0
        
        context = f"Historical Performance: {total_trades} trades, {win_rate:.1%} win rate"
        
        # Add pattern analysis
        if self.market_patterns:
            successful_patterns = [p for p, data in self.market_patterns.items() if data.get('success_rate', 0) > 0.7]
            if successful_patterns:
                context += f"\nHigh-success patterns: {', '.join(successful_patterns[:3])}"
        
        return context
    
    def _calibrate_confidence(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrate confidence based on historical performance"""
        if not self.confidence_calibration:
            return analysis
        
        strategy = analysis.get('strategy', 'unknown')
        if strategy in self.confidence_calibration:
            calibration_factor = self.confidence_calibration[strategy]
            original_confidence = analysis.get('confidence', 0.5)
            calibrated_confidence = min(1.0, max(0.0, original_confidence * calibration_factor))
            
            analysis['confidence'] = calibrated_confidence
            analysis['confidence_calibrated'] = True
            
            self.logger.debug(f"🎯 Confidence calibrated for {strategy}: {original_confidence:.3f} -> {calibrated_confidence:.3f}")
        
        return analysis
    
    def _store_analysis(self, analysis: Dict[str, Any], prompt: str):
        """Store analysis for learning and pattern recognition"""
        analysis_record = {
            'timestamp': datetime.now(),
            'analysis': analysis.copy(),
            'prompt_hash': hash(prompt),
            'outcome': None  # Will be updated when trade completes
        }
        
        self.analysis_history.append(analysis_record)
        
        # Keep only last 1000 analyses to manage memory
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]
    
    def update_trade_outcome(self, analysis_id: str, outcome: str, pnl: float):
        """Update the outcome of a trade for learning purposes"""
        try:
            # Find the analysis record and update outcome
            for record in reversed(self.analysis_history):
                if record.get('analysis', {}).get('timestamp') == analysis_id:
                    record['outcome'] = outcome
                    record['pnl'] = pnl
                    
                    # Update performance feedback
                    if outcome == 'win':
                        self.performance_feedback['wins'].append({
                            'strategy': record['analysis'].get('strategy'),
                            'confidence': record['analysis'].get('confidence'),
                            'pnl': pnl,
                            'timestamp': record['timestamp']
                        })
                    else:
                        self.performance_feedback['losses'].append({
                            'strategy': record['analysis'].get('strategy'),
                            'confidence': record['analysis'].get('confidence'),
                            'pnl': pnl,
                            'timestamp': record['timestamp']
                        })
                    
                    # Update pattern recognition
                    self._update_pattern_recognition(record)
                    
                    # Update confidence calibration
                    self._update_confidence_calibration(record)
                    
                    break
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating trade outcome: {e}")
    
    def _update_pattern_recognition(self, record: Dict[str, Any]):
        """Update market pattern recognition based on trade outcomes"""
        try:
            strategy = record['analysis'].get('strategy', 'unknown')
            outcome = record.get('outcome')
            
            if strategy not in self.market_patterns:
                self.market_patterns[strategy] = {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'success_rate': 0.0,
                    'avg_pnl': 0.0,
                    'total_pnl': 0.0
                }
            
            pattern = self.market_patterns[strategy]
            pattern['total_trades'] += 1
            pattern['total_pnl'] += record.get('pnl', 0)
            
            if outcome == 'win':
                pattern['successful_trades'] += 1
            
            pattern['success_rate'] = pattern['successful_trades'] / pattern['total_trades']
            pattern['avg_pnl'] = pattern['total_pnl'] / pattern['total_trades']
            
        except Exception as e:
            self.logger.error(f"❌ Error updating pattern recognition: {e}")
    
    def _update_confidence_calibration(self, record: Dict[str, Any]):
        """Update confidence calibration based on actual outcomes"""
        try:
            strategy = record['analysis'].get('strategy', 'unknown')
            confidence = record['analysis'].get('confidence', 0.5)
            outcome = record.get('outcome')
            
            if strategy not in self.confidence_calibration:
                self.confidence_calibration[strategy] = 1.0
            
            # Simple calibration: if high confidence trades fail, reduce calibration
            if confidence > 0.8 and outcome == 'loss':
                self.confidence_calibration[strategy] *= 0.95
            elif confidence > 0.8 and outcome == 'win':
                self.confidence_calibration[strategy] = min(1.1, self.confidence_calibration[strategy] * 1.02)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating confidence calibration: {e}")
    
    async def analyze_portfolio_optimization(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio for optimization opportunities"""
        prompt = f"""
        PORTFOLIO OPTIMIZATION ANALYSIS
        
        Current Portfolio Status:
        {json.dumps(portfolio_data, indent=2)}
        
        OPTIMIZATION OBJECTIVES:
        1. Maximize compound growth rate
        2. Minimize correlation risk
        3. Optimize position sizing
        4. Identify rebalancing opportunities
        5. Detect hedging requirements
        
        Provide portfolio optimization recommendations focusing on:
        - Position size adjustments
        - Correlation hedging
        - Risk rebalancing
        - Profit taking strategies
        - New opportunity identification
        """
        
        return await self.analyze_market_data(prompt)
    
    async def analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market regime for strategy adaptation"""
        prompt = f"""
        MARKET REGIME ANALYSIS
        
        Multi-Asset Market Data:
        {json.dumps(market_data, indent=2)}
        
        REGIME CLASSIFICATION TASK:
        Analyze the current market regime and provide strategic recommendations:
        
        1. Market Phase: Bull/Bear/Sideways/Transition
        2. Volatility Regime: Low/Medium/High/Extreme
        3. Liquidity Conditions: Abundant/Normal/Stressed/Crisis
        4. Correlation Environment: Low/Medium/High/Extreme
        5. Momentum Strength: Weak/Moderate/Strong/Explosive
        
        Provide regime-specific trading recommendations and strategy adjustments.
        """
        
        return await self.analyze_market_data(prompt)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of AI performance and learning"""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {"message": "No analysis history available"}
        
        # Calculate performance metrics
        completed_trades = [a for a in self.analysis_history if a.get('outcome') is not None]
        wins = [a for a in completed_trades if a.get('outcome') == 'win']
        
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0
        total_pnl = sum(a.get('pnl', 0) for a in completed_trades)
        
        # Strategy performance
        strategy_performance = {}
        for strategy, data in self.market_patterns.items():
            strategy_performance[strategy] = {
                'success_rate': data['success_rate'],
                'avg_pnl': data['avg_pnl'],
                'total_trades': data['total_trades']
            }
        
        return {
            'total_analyses': total_analyses,
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'strategy_performance': strategy_performance,
            'confidence_calibration': self.confidence_calibration,
            'learning_active': True
        }
    
    def analyze_multiple_timeframes(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multiple timeframes for comprehensive trading decision.
        
        Args:
            symbol (str): Trading symbol
            market_data (Dict): Market data for different timeframes
            
        Returns:
            Dict[str, Any]: Consolidated analysis across timeframes
        """
        prompt = f"""
        MULTI-TIMEFRAME ANALYSIS FOR {symbol}
        
        Current Market Data:
        1m: {market_data.get('1m', 'No data')}
        3m: {market_data.get('3m', 'No data')}
        5m: {market_data.get('5m', 'No data')}
        15m: {market_data.get('15m', 'No data')}
        
        Volume Analysis: {market_data.get('volume_analysis', 'No data')}
        Order Book: {market_data.get('orderbook', 'No data')}
        Recent Trades: {market_data.get('recent_trades', 'No data')}
        
        TASK: Provide aggressive scalping decision for maximum profit extraction.
        Consider:
        - Volume spikes indicating whale activity
        - Support/resistance breaks
        - Momentum divergences
        - Liquidity zones
        - Market microstructure
        
        Target: Turn 2 USDT into 20,000 USDT through compound gains.
        """
        
        return self.analyze_market_data(prompt)
    
    def emergency_loss_mitigation(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emergency analysis for loss mitigation and position management.
        
        Args:
            position_data (Dict): Current position information
            
        Returns:
            Dict[str, Any]: Emergency action recommendation
        """
        prompt = f"""
        EMERGENCY LOSS MITIGATION ANALYSIS
        
        Current Position:
        Symbol: {position_data.get('symbol')}
        Side: {position_data.get('side')}
        Size: {position_data.get('size')}
        Entry Price: {position_data.get('entry_price')}
        Current Price: {position_data.get('current_price')}
        Unrealized PnL: {position_data.get('unrealized_pnl')}
        Leverage: {position_data.get('leverage')}
        
        Market Conditions:
        Volatility: {position_data.get('volatility', 'Unknown')}
        Volume: {position_data.get('volume', 'Unknown')}
        Trend: {position_data.get('trend', 'Unknown')}
        
        CRITICAL DECISION NEEDED:
        - Should we close immediately?
        - Hedge with opposite position?
        - Add to position (average down)?
        - Adjust stop loss?
        - Wait for reversal?
        
        Priority: Capital preservation while maximizing recovery potential.
        """
        
        return self.analyze_market_data(prompt, temperature=0.05)  # Lower temperature for emergency decisions

if __name__ == '__main__':
    # Test the OpenAI analyzer
    analyzer = OpenAIAnalyzer()
    
    if analyzer.client:
        test_prompt = """
        Analyze 1000PEPEUSDT for scalping opportunity:
        
        Current Price: 0.00001234
        1m Volume: 1,250,000 USDT (300% above average)
        RSI: 45 (neutral)
        MACD: Bullish crossover
        Support: 0.00001200
        Resistance: 0.00001280
        
        Recent whale activity detected. Provide aggressive scalping decision.
        """
        
        result = analyzer.analyze_market_data(test_prompt)
        print(f"Test Analysis Result: {json.dumps(result, indent=2)}")
    else:
        print("OpenAI client not initialized - check API key")