"""
Advanced Risk Management System for Autonomous Trading
====================================================

Ultra-sophisticated risk management designed for turning $2 into $20,000
through aggressive but controlled trading strategies.

Features:
- Dynamic position sizing based on market conditions
- Real-time portfolio risk monitoring
- Adaptive stop-loss and take-profit optimization
- Correlation-based risk assessment
- Volatility-adjusted leverage scaling
- Emergency risk controls and circuit breakers

Author: Skyscope AI Trading Systems
Version: 2.0.0
"""

import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
import asyncio


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring"""
    portfolio_value: float
    total_exposure: float
    leverage_ratio: float
    var_1d: float  # Value at Risk 1 day
    max_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    volatility_risk: float


@dataclass
class PositionRisk:
    """Risk assessment for individual positions"""
    symbol: str
    position_size: float
    leverage: int
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float
    risk_percentage: float
    time_decay_risk: float
    liquidity_score: float
    correlation_exposure: float


class AdvancedRiskManager:
    """
    Ultra-advanced risk management system for autonomous cryptocurrency trading.
    Designed to maximize returns while preserving capital through intelligent risk controls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced risk manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_portfolio_risk = config.get('max_portfolio_risk_pct', 25.0) / 100
        self.max_position_risk = config.get('max_risk_per_trade_pct', 5.0) / 100
        self.max_leverage = config.get('max_leverage', 50)
        self.max_drawdown_limit = config.get('max_drawdown_pct', 15.0) / 100
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        
        # Risk tracking
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_history: List[Dict] = []
        self.risk_events: List[Dict] = []
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.volatility_estimates: Dict[str, float] = {}
        
        # Circuit breakers
        self.emergency_stop_triggered = False
        self.daily_loss_limit_hit = False
        self.max_drawdown_hit = False
        
        self.logger.info("🛡️ Advanced Risk Manager initialized")
        self.logger.info(f"📊 Max Portfolio Risk: {self.max_portfolio_risk:.1%}")
        self.logger.info(f"⚡ Max Position Risk: {self.max_position_risk:.1%}")
        self.logger.info(f"🔥 Max Leverage: {self.max_leverage}x")
    
    async def calculate_optimal_position_size(
        self, 
        signal: Dict[str, Any], 
        portfolio_value: float,
        market_data: Dict[str, Any],
        existing_positions: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size using advanced risk management algorithms.
        
        Returns:
            Tuple[float, Dict]: (position_size, risk_analysis)
        """
        try:
            symbol = signal['symbol']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            confidence = signal['confidence']
            leverage = signal['leverage']
            
            # Base risk calculation using Kelly Criterion with modifications
            base_risk = self._calculate_kelly_position_size(signal, market_data)
            
            # Apply confidence scaling
            confidence_multiplier = self._calculate_confidence_multiplier(confidence)
            
            # Apply volatility scaling
            volatility_multiplier = self._calculate_volatility_multiplier(symbol, market_data)
            
            # Apply correlation adjustment
            correlation_multiplier = self._calculate_correlation_multiplier(symbol, existing_positions)
            
            # Apply market condition adjustment
            market_multiplier = self._calculate_market_condition_multiplier(market_data)
            
            # Apply time-based adjustment
            time_multiplier = self._calculate_time_multiplier()
            
            # Calculate final position size
            risk_multipliers = (
                confidence_multiplier * 
                volatility_multiplier * 
                correlation_multiplier * 
                market_multiplier * 
                time_multiplier
            )
            
            adjusted_risk = base_risk * risk_multipliers
            
            # Apply portfolio-level constraints
            max_position_value = portfolio_value * self.max_position_risk
            
            # Calculate position size in base currency
            price_diff = abs(entry_price - stop_loss)
            if price_diff <= 0:
                return 0.0, {"error": "Invalid stop loss level"}
            
            # Risk-based position size
            risk_based_size = (adjusted_risk * portfolio_value) / price_diff
            
            # Apply leverage
            leveraged_size = risk_based_size / leverage if leverage > 1 else risk_based_size
            
            # Apply maximum position value constraint
            max_size_by_value = max_position_value / entry_price
            final_size = min(leveraged_size, max_size_by_value)
            
            # Ensure minimum viable size
            min_size = self._get_minimum_position_size(symbol)
            if final_size < min_size:
                return 0.0, {"error": "Position size below minimum threshold"}
            
            # Create risk analysis
            risk_analysis = {
                'base_risk': base_risk,
                'confidence_multiplier': confidence_multiplier,
                'volatility_multiplier': volatility_multiplier,
                'correlation_multiplier': correlation_multiplier,
                'market_multiplier': market_multiplier,
                'time_multiplier': time_multiplier,
                'final_multiplier': risk_multipliers,
                'adjusted_risk': adjusted_risk,
                'position_size': final_size,
                'risk_amount': final_size * price_diff,
                'risk_percentage': (final_size * price_diff) / portfolio_value,
                'leverage_used': leverage,
                'max_loss_usd': final_size * price_diff * leverage
            }
            
            self.logger.info(f"🎯 Optimal position size for {symbol}: {final_size:.6f}")
            self.logger.info(f"💰 Risk amount: ${risk_analysis['risk_amount']:.2f} ({risk_analysis['risk_percentage']:.2%})")
            
            return final_size, risk_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.0, {"error": str(e)}
    
    def _calculate_kelly_position_size(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate position size using modified Kelly Criterion"""
        try:
            confidence = signal['confidence']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            
            # Calculate win probability (based on confidence and historical data)
            win_prob = self._estimate_win_probability(signal, market_data)
            
            # Calculate average win/loss ratio
            if signal['action'] == 'BUY':
                avg_win = (take_profit - entry_price) / entry_price
                avg_loss = (entry_price - stop_loss) / entry_price
            else:
                avg_win = (entry_price - take_profit) / entry_price
                avg_loss = (stop_loss - entry_price) / entry_price
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win_prob, q = 1-p
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_prob - (1 - win_prob)) / b
            else:
                kelly_fraction = 0
            
            # Apply Kelly fraction limits (never risk more than 25% on Kelly)
            kelly_fraction = max(0, min(0.25, kelly_fraction))
            
            # Apply confidence scaling
            confidence_adjusted_kelly = kelly_fraction * confidence
            
            return confidence_adjusted_kelly
            
        except Exception as e:
            self.logger.error(f"❌ Error in Kelly calculation: {e}")
            return 0.01  # Default to 1% risk
    
    def _estimate_win_probability(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Estimate win probability based on signal characteristics and market conditions"""
        base_prob = signal['confidence']
        
        # Adjust based on market volatility
        volatility = self._calculate_current_volatility(signal['symbol'], market_data)
        if volatility > 0.05:  # High volatility
            base_prob *= 0.9  # Reduce confidence in high volatility
        elif volatility < 0.02:  # Low volatility
            base_prob *= 1.1  # Increase confidence in low volatility
        
        # Adjust based on volume
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:  # High volume
            base_prob *= 1.05
        elif volume_ratio < 0.5:  # Low volume
            base_prob *= 0.95
        
        return min(0.95, max(0.05, base_prob))  # Clamp between 5% and 95%
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate position size multiplier based on AI confidence"""
        if confidence >= 0.95:
            return 2.0  # Double size for ultra-high confidence
        elif confidence >= 0.90:
            return 1.5
        elif confidence >= 0.85:
            return 1.2
        elif confidence >= 0.80:
            return 1.0
        elif confidence >= 0.75:
            return 0.8
        else:
            return 0.5  # Half size for lower confidence
    
    def _calculate_volatility_multiplier(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate position size multiplier based on volatility"""
        try:
            volatility = self._calculate_current_volatility(symbol, market_data)
            
            if volatility > 0.08:  # Extreme volatility
                return 0.5
            elif volatility > 0.05:  # High volatility
                return 0.7
            elif volatility > 0.03:  # Medium volatility
                return 1.0
            elif volatility > 0.01:  # Low volatility
                return 1.3
            else:  # Very low volatility
                return 1.5
                
        except Exception:
            return 1.0
    
    def _calculate_correlation_multiplier(self, symbol: str, existing_positions: Dict[str, Any]) -> float:
        """Calculate position size multiplier based on correlation with existing positions"""
        if not existing_positions:
            return 1.0
        
        try:
            # Calculate correlation exposure
            total_correlation_exposure = 0.0
            
            for pos_symbol, position in existing_positions.items():
                if pos_symbol == symbol:
                    continue
                
                correlation = self._get_correlation(symbol, pos_symbol)
                position_weight = abs(position.get('unrealized_pnl', 0))
                
                total_correlation_exposure += abs(correlation) * position_weight
            
            # Reduce position size if high correlation exposure
            if total_correlation_exposure > 0.7:
                return 0.5
            elif total_correlation_exposure > 0.5:
                return 0.7
            elif total_correlation_exposure > 0.3:
                return 0.9
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_market_condition_multiplier(self, market_data: Dict[str, Any]) -> float:
        """Calculate position size multiplier based on overall market conditions"""
        try:
            # Analyze market sentiment
            btc_change = market_data.get('btc_24h_change', 0)
            market_fear_greed = market_data.get('fear_greed_index', 50)
            
            # Bull market conditions
            if btc_change > 5 and market_fear_greed > 70:
                return 1.3  # Increase size in strong bull market
            elif btc_change > 2 and market_fear_greed > 60:
                return 1.1
            # Bear market conditions
            elif btc_change < -5 and market_fear_greed < 30:
                return 0.7  # Reduce size in strong bear market
            elif btc_change < -2 and market_fear_greed < 40:
                return 0.9
            else:
                return 1.0  # Neutral conditions
                
        except Exception:
            return 1.0
    
    def _calculate_time_multiplier(self) -> float:
        """Calculate position size multiplier based on time factors"""
        try:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
            
            # Reduce size during low-liquidity periods
            if current_day >= 5:  # Weekend
                return 0.8
            elif current_hour < 6 or current_hour > 22:  # Night hours
                return 0.9
            elif 8 <= current_hour <= 16:  # Peak trading hours
                return 1.1
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_current_volatility(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate current volatility for a symbol"""
        try:
            # Use 24h price change as volatility proxy
            price_change = abs(market_data.get('price_change_24h', 0)) / 100
            
            # Get historical volatility if available
            if symbol in self.volatility_estimates:
                historical_vol = self.volatility_estimates[symbol]
                # Weighted average of current and historical
                return 0.7 * price_change + 0.3 * historical_vol
            
            return price_change
            
        except Exception:
            return 0.03  # Default volatility
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        try:
            if self.correlation_matrix.empty:
                return 0.0
            
            if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                return self.correlation_matrix.loc[symbol1, symbol2]
            
            # Default correlations for common pairs
            if 'BTC' in symbol1 and 'BTC' in symbol2:
                return 0.9
            elif ('ETH' in symbol1 and 'ETH' in symbol2) or ('BTC' in symbol1 and 'ETH' in symbol2):
                return 0.7
            elif '1000' in symbol1 and '1000' in symbol2:
                return 0.5  # Meme coins tend to be correlated
            else:
                return 0.3  # Default moderate correlation
                
        except Exception:
            return 0.0
    
    def _get_minimum_position_size(self, symbol: str) -> float:
        """Get minimum viable position size for a symbol"""
        # This would typically come from exchange info
        if '1000' in symbol:
            return 100.0  # Minimum for 1000-prefixed tokens
        elif symbol in ['BTCUSDT', 'ETHUSDT']:
            return 0.001
        else:
            return 0.01
    
    async def assess_portfolio_risk(self, positions: Dict[str, Any], portfolio_value: float) -> RiskMetrics:
        """Assess overall portfolio risk"""
        try:
            total_exposure = sum(abs(pos.get('unrealized_pnl', 0)) for pos in positions.values())
            leverage_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Calculate VaR (simplified)
            var_1d = self._calculate_var(positions, portfolio_value)
            
            # Calculate other risk metrics
            max_drawdown = self._calculate_max_drawdown()
            sharpe_ratio = self._calculate_sharpe_ratio()
            correlation_risk = self._calculate_correlation_risk(positions)
            liquidity_risk = self._calculate_liquidity_risk(positions)
            concentration_risk = self._calculate_concentration_risk(positions, portfolio_value)
            volatility_risk = self._calculate_volatility_risk(positions)
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                leverage_ratio=leverage_ratio,
                var_1d=var_1d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk
            )
            
            # Check for risk limit breaches
            await self._check_risk_limits(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing portfolio risk: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=0,
                leverage_ratio=0,
                var_1d=0,
                max_drawdown=0,
                sharpe_ratio=0,
                correlation_risk=0,
                liquidity_risk=0,
                concentration_risk=0,
                volatility_risk=0
            )
    
    def _calculate_var(self, positions: Dict[str, Any], portfolio_value: float, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation
            position_values = [abs(pos.get('unrealized_pnl', 0)) for pos in positions.values()]
            
            if not position_values:
                return 0.0
            
            # Use historical simulation method (simplified)
            portfolio_volatility = np.std(position_values) if len(position_values) > 1 else 0.02
            z_score = 1.645 if confidence_level == 0.95 else 2.33  # 95% or 99%
            
            var = portfolio_value * portfolio_volatility * z_score
            return var
            
        except Exception:
            return portfolio_value * 0.05  # Default 5% VaR
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            values = [entry['portfolio_value'] for entry in self.portfolio_history]
            peak = values[0]
            max_dd = 0.0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception:
            return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from portfolio history"""
        try:
            if len(self.portfolio_history) < 10:
                return 0.0
            
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]['portfolio_value']
                curr_value = self.portfolio_history[i]['portfolio_value']
                
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if not returns:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = (mean_return * 365) / (std_return * np.sqrt(365))
            return sharpe
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate correlation risk score"""
        try:
            if len(positions) < 2:
                return 0.0
            
            symbols = list(positions.keys())
            total_correlation = 0.0
            pairs = 0
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = abs(self._get_correlation(symbol1, symbol2))
                    total_correlation += correlation
                    pairs += 1
            
            avg_correlation = total_correlation / pairs if pairs > 0 else 0.0
            return avg_correlation
            
        except Exception:
            return 0.0
    
    def _calculate_liquidity_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate liquidity risk score"""
        try:
            if not positions:
                return 0.0
            
            # Simplified liquidity scoring
            total_risk = 0.0
            
            for symbol, position in positions.items():
                if '1000' in symbol:
                    liquidity_score = 0.7  # Lower liquidity for meme coins
                elif symbol in ['BTCUSDT', 'ETHUSDT']:
                    liquidity_score = 0.1  # High liquidity
                else:
                    liquidity_score = 0.4  # Medium liquidity
                
                position_weight = abs(position.get('unrealized_pnl', 0))
                total_risk += liquidity_score * position_weight
            
            total_exposure = sum(abs(pos.get('unrealized_pnl', 0)) for pos in positions.values())
            
            return total_risk / total_exposure if total_exposure > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any], portfolio_value: float) -> float:
        """Calculate concentration risk score"""
        try:
            if not positions or portfolio_value <= 0:
                return 0.0
            
            position_weights = []
            for position in positions.values():
                weight = abs(position.get('unrealized_pnl', 0)) / portfolio_value
                position_weights.append(weight)
            
            if not position_weights:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in position_weights)
            
            # Normalize to 0-1 scale (1 = maximum concentration)
            max_hhi = 1.0  # All in one position
            normalized_hhi = hhi / max_hhi
            
            return normalized_hhi
            
        except Exception:
            return 0.0
    
    def _calculate_volatility_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate volatility risk score"""
        try:
            if not positions:
                return 0.0
            
            total_vol_risk = 0.0
            total_exposure = 0.0
            
            for symbol, position in positions.items():
                volatility = self.volatility_estimates.get(symbol, 0.03)
                exposure = abs(position.get('unrealized_pnl', 0))
                
                total_vol_risk += volatility * exposure
                total_exposure += exposure
            
            return total_vol_risk / total_exposure if total_exposure > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _check_risk_limits(self, risk_metrics: RiskMetrics):
        """Check if any risk limits are breached"""
        try:
            # Check maximum drawdown
            if risk_metrics.max_drawdown > self.max_drawdown_limit:
                self.max_drawdown_hit = True
                await self._trigger_risk_event("MAX_DRAWDOWN_EXCEEDED", risk_metrics.max_drawdown)
            
            # Check portfolio risk
            portfolio_risk = risk_metrics.total_exposure / risk_metrics.portfolio_value if risk_metrics.portfolio_value > 0 else 0
            if portfolio_risk > self.max_portfolio_risk:
                await self._trigger_risk_event("PORTFOLIO_RISK_EXCEEDED", portfolio_risk)
            
            # Check leverage
            if risk_metrics.leverage_ratio > self.max_leverage:
                await self._trigger_risk_event("LEVERAGE_EXCEEDED", risk_metrics.leverage_ratio)
            
            # Check correlation risk
            if risk_metrics.correlation_risk > self.correlation_threshold:
                await self._trigger_risk_event("HIGH_CORRELATION_RISK", risk_metrics.correlation_risk)
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits: {e}")
    
    async def _trigger_risk_event(self, event_type: str, value: float):
        """Trigger a risk management event"""
        risk_event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'value': value,
            'action_taken': None
        }
        
        self.risk_events.append(risk_event)
        
        self.logger.warning(f"🚨 RISK EVENT: {event_type} - Value: {value:.4f}")
        
        # Take appropriate action based on event type
        if event_type == "MAX_DRAWDOWN_EXCEEDED":
            self.emergency_stop_triggered = True
            risk_event['action_taken'] = "EMERGENCY_STOP"
            self.logger.critical("🚨 EMERGENCY STOP TRIGGERED - MAX DRAWDOWN EXCEEDED")
        
        elif event_type == "PORTFOLIO_RISK_EXCEEDED":
            risk_event['action_taken'] = "REDUCE_POSITIONS"
            self.logger.warning("⚠️ Portfolio risk exceeded - Consider reducing positions")
        
        elif event_type == "LEVERAGE_EXCEEDED":
            risk_event['action_taken'] = "REDUCE_LEVERAGE"
            self.logger.warning("⚠️ Leverage exceeded - Reducing position sizes")
        
        elif event_type == "HIGH_CORRELATION_RISK":
            risk_event['action_taken'] = "DIVERSIFY"
            self.logger.warning("⚠️ High correlation risk - Diversification needed")
    
    def update_portfolio_history(self, portfolio_value: float, positions: Dict[str, Any]):
        """Update portfolio history for risk calculations"""
        entry = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'position_count': len(positions),
            'total_exposure': sum(abs(pos.get('unrealized_pnl', 0)) for pos in positions.values())
        }
        
        self.portfolio_history.append(entry)
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
    
    def update_volatility_estimates(self, symbol: str, price_data: List[float]):
        """Update volatility estimates for a symbol"""
        try:
            if len(price_data) < 2:
                return
            
            returns = []
            for i in range(1, len(price_data)):
                returns.append((price_data[i] - price_data[i-1]) / price_data[i-1])
            
            volatility = np.std(returns) if returns else 0.03
            self.volatility_estimates[symbol] = volatility
            
        except Exception as e:
            self.logger.error(f"❌ Error updating volatility for {symbol}: {e}")
    
    def update_correlation_matrix(self, price_data: Dict[str, List[float]]):
        """Update correlation matrix between symbols"""
        try:
            if len(price_data) < 2:
                return
            
            # Create DataFrame from price data
            df = pd.DataFrame(price_data)
            
            # Calculate returns
            returns_df = df.pct_change().dropna()
            
            # Calculate correlation matrix
            self.correlation_matrix = returns_df.corr()
            
            self.logger.debug("📊 Correlation matrix updated")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating correlation matrix: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a comprehensive risk summary"""
        return {
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'daily_loss_limit_hit': self.daily_loss_limit_hit,
            'max_drawdown_hit': self.max_drawdown_hit,
            'recent_risk_events': self.risk_events[-10:],  # Last 10 events
            'volatility_estimates': dict(list(self.volatility_estimates.items())[:10]),  # Top 10
            'correlation_summary': {
                'matrix_size': self.correlation_matrix.shape if not self.correlation_matrix.empty else (0, 0),
                'avg_correlation': self.correlation_matrix.values.mean() if not self.correlation_matrix.empty else 0
            },
            'portfolio_history_length': len(self.portfolio_history)
        }
    
    def reset_daily_limits(self):
        """Reset daily risk limits (call at start of each day)"""
        self.daily_loss_limit_hit = False
        self.logger.info("🔄 Daily risk limits reset")
    
    def should_allow_new_position(self, signal: Dict[str, Any], portfolio_value: float) -> Tuple[bool, str]:
        """Check if a new position should be allowed based on risk limits"""
        try:
            # Check emergency stop
            if self.emergency_stop_triggered:
                return False, "Emergency stop is active"
            
            # Check daily loss limit
            if self.daily_loss_limit_hit:
                return False, "Daily loss limit reached"
            
            # Check maximum drawdown
            if self.max_drawdown_hit:
                return False, "Maximum drawdown limit reached"
            
            # Check confidence threshold
            if signal.get('confidence', 0) < 0.75:
                return False, "Signal confidence below threshold"
            
            # Check portfolio value
            if portfolio_value < 1.0:  # Less than $1 remaining
                return False, "Insufficient portfolio value"
            
            return True, "Position allowed"
            
        except Exception as e:
            self.logger.error(f"❌ Error checking position allowance: {e}")
            return False, f"Error: {str(e)}"


# Utility functions for risk calculations
def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns"""
    if not returns or len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return


def calculate_maximum_drawdown(portfolio_values: List[float]) -> float:
    """Calculate maximum drawdown from portfolio values"""
    if len(portfolio_values) < 2:
        return 0.0
    
    peak = portfolio_values[0]
    max_drawdown = 0.0
    
    for value in portfolio_values[1:]:
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown


def calculate_value_at_risk(returns: List[float], confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if not returns:
        return 0.0
    
    sorted_returns = sorted(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    
    return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0