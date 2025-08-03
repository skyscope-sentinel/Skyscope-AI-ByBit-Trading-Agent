#!/usr/bin/env python3
"""
Autonomous Cryptocurrency Trading Agent - Main Execution Script
=============================================================

Ultra-sophisticated AI-powered trading system designed to transform
$2 USD into $20,000 USD through aggressive scalping and momentum trading.

🎯 MISSION: 10,000x return through autonomous trading
🧠 AI: OpenAI GPT-4 Turbo for market analysis
⚡ EXECUTION: Sub-second trade execution on ByBit
🛡️ RISK: Advanced risk management and position sizing
📊 MONITORING: Real-time performance tracking and optimization

Author: Skyscope AI Trading Systems
Version: 2.0.0
License: Proprietary
"""

import asyncio
import logging
import json
import os
import sys
import signal
import argparse
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bybit_gemini_bot.autonomous_trader import AutonomousTrader
from bybit_gemini_bot.logger_setup import setup_logging


class TradingSystemManager:
    """
    Main manager for the autonomous trading system.
    Handles initialization, monitoring, and graceful shutdown.
    """
    
    def __init__(self, config_path: str):
        """Initialize the trading system manager"""
        self.config_path = config_path
        self.config = None
        self.trader = None
        self.is_running = False
        self.shutdown_requested = False
        
        # Setup logging first
        setup_logging(logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 Trading System Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            self.logger.info("🔧 Initializing Autonomous Trading System...")
            
            # Load configuration
            if not await self._load_configuration():
                return False
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            # Initialize the autonomous trader
            self.trader = AutonomousTrader(self.config)
            
            # Perform pre-flight checks
            if not await self._perform_preflight_checks():
                return False
            
            self.logger.info("✅ Trading System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading system: {e}")
            return False
    
    async def _load_configuration(self) -> bool:
        """Load and parse configuration file"""
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"❌ Configuration file not found: {self.config_path}")
                return False
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.logger.info(f"📋 Configuration loaded from: {self.config_path}")
            
            # Mask sensitive information in logs
            masked_config = self._mask_sensitive_config(self.config.copy())
            self.logger.debug(f"Configuration: {json.dumps(masked_config, indent=2)}")
            
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON in configuration file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading configuration: {e}")
            return False
    
    def _mask_sensitive_config(self, config: dict) -> dict:
        """Mask sensitive information in configuration for logging"""
        sensitive_keys = ['api_key', 'api_secret', 'secret', 'key', 'token', 'password']
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_keys):
                        if isinstance(value, str) and len(value) > 8:
                            obj[key] = f"{value[:4]}...{value[-4:]}"
                        else:
                            obj[key] = "***MASKED***"
                    else:
                        mask_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    mask_recursive(item)
        
        mask_recursive(config)
        return config
    
    def _validate_configuration(self) -> bool:
        """Validate configuration parameters"""
        try:
            required_sections = [
                'bybit_api', 'openai_api', 'trading_symbols', 
                'risk_management', 'strategy_config'
            ]
            
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"❌ Missing required configuration section: {section}")
                    return False
            
            # Validate API keys
            if not self.config['bybit_api'].get('api_key') or 'YOUR_' in self.config['bybit_api']['api_key']:
                self.logger.error("❌ Invalid ByBit API key in configuration")
                return False
            
            if not self.config['openai_api'].get('api_key') or 'YOUR_' in self.config['openai_api']['api_key']:
                self.logger.error("❌ Invalid OpenAI API key in configuration")
                return False
            
            # Validate trading parameters
            if not self.config['trading_symbols']:
                self.logger.error("❌ No trading symbols specified")
                return False
            
            # Validate risk parameters
            risk_config = self.config['risk_management']
            if risk_config.get('max_risk_per_trade_pct', 0) <= 0:
                self.logger.error("❌ Invalid risk per trade percentage")
                return False
            
            if risk_config.get('max_leverage', 0) <= 0:
                self.logger.error("❌ Invalid maximum leverage")
                return False
            
            self.logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating configuration: {e}")
            return False
    
    async def _perform_preflight_checks(self) -> bool:
        """Perform pre-flight checks before starting trading"""
        try:
            self.logger.info("🔍 Performing pre-flight checks...")
            
            # Check API connectivity
            if not await self._check_api_connectivity():
                return False
            
            # Check account balance
            if not await self._check_account_balance():
                return False
            
            # Check market data access
            if not await self._check_market_data_access():
                return False
            
            # Check system resources
            if not self._check_system_resources():
                return False
            
            self.logger.info("✅ All pre-flight checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pre-flight checks failed: {e}")
            return False
    
    async def _check_api_connectivity(self) -> bool:
        """Check API connectivity"""
        try:
            self.logger.info("🔗 Checking API connectivity...")
            
            # Test ByBit API
            bybit_connector = self.trader.bybit_connector
            server_time = bybit_connector.get_server_time()
            
            if not server_time or server_time.get('retCode') != 0:
                self.logger.error("❌ ByBit API connectivity failed")
                return False
            
            # Test OpenAI API
            openai_analyzer = self.trader.openai_analyzer
            if not openai_analyzer.client:
                self.logger.error("❌ OpenAI API client not initialized")
                return False
            
            self.logger.info("✅ API connectivity verified")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API connectivity check failed: {e}")
            return False
    
    async def _check_account_balance(self) -> bool:
        """Check account balance"""
        try:
            self.logger.info("💰 Checking account balance...")
            
            bybit_connector = self.trader.bybit_connector
            balance_response = bybit_connector.get_wallet_balance(account_type="UNIFIED")
            
            if not balance_response or balance_response.get('retCode') != 0:
                self.logger.error("❌ Failed to retrieve account balance")
                return False
            
            # Find USDT balance
            balance_list = balance_response.get('result', {}).get('list', [])
            usdt_balance = 0.0
            
            for account in balance_list:
                for coin in account.get('coin', []):
                    if coin.get('coin') == 'USDT':
                        usdt_balance = float(coin.get('availableToWithdraw', 0))
                        break
            
            if usdt_balance < self.config.get('initial_capital_usdt', 2.0):
                self.logger.error(f"❌ Insufficient USDT balance: ${usdt_balance:.2f}")
                return False
            
            self.logger.info(f"✅ Account balance verified: ${usdt_balance:.2f} USDT")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Account balance check failed: {e}")
            return False
    
    async def _check_market_data_access(self) -> bool:
        """Check market data access"""
        try:
            self.logger.info("📊 Checking market data access...")
            
            bybit_connector = self.trader.bybit_connector
            
            # Test ticker data
            ticker_response = bybit_connector.get_tickers(category="linear", symbol="BTCUSDT")
            
            if not ticker_response or ticker_response.get('retCode') != 0:
                self.logger.error("❌ Failed to retrieve ticker data")
                return False
            
            # Test kline data
            kline_response = bybit_connector.get_kline(
                category="linear", 
                symbol="BTCUSDT", 
                interval="1", 
                limit=10
            )
            
            if not kline_response or kline_response.get('retCode') != 0:
                self.logger.error("❌ Failed to retrieve kline data")
                return False
            
            self.logger.info("✅ Market data access verified")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market data access check failed: {e}")
            return False
    
    def _check_system_resources(self) -> bool:
        """Check system resources"""
        try:
            import psutil
            
            self.logger.info("🖥️ Checking system resources...")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.logger.warning(f"⚠️ Low disk space: {disk.percent:.1f}% used")
            
            self.logger.info("✅ System resources checked")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System resource check failed: {e}")
            return True  # Don't fail startup for this
    
    async def run(self):
        """Run the autonomous trading system"""
        try:
            self.logger.info("🚀 Starting Autonomous Trading System")
            self.logger.info("=" * 60)
            self.logger.info("🎯 MISSION: Turn $2 into $20,000")
            self.logger.info("🧠 AI: OpenAI GPT-4 Turbo")
            self.logger.info("⚡ EXCHANGE: ByBit")
            self.logger.info("🛡️ RISK: Advanced Risk Management")
            self.logger.info("=" * 60)
            
            self.is_running = True
            
            # Start the autonomous trader
            await self.trader.start_trading()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"❌ Critical error in trading system: {e}")
        finally:
            await self._shutdown()
    
    async def _shutdown(self):
        """Graceful shutdown of the trading system"""
        try:
            self.logger.info("🛑 Initiating graceful shutdown...")
            self.is_running = False
            
            if self.trader:
                # Emergency shutdown of trader
                await self.trader.emergency_shutdown()
            
            # Final performance report
            if self.trader and self.trader.performance_metrics:
                metrics = self.trader.performance_metrics
                self.logger.info("📊 FINAL PERFORMANCE REPORT:")
                self.logger.info(f"💰 Final Balance: ${metrics.current_balance:.2f}")
                self.logger.info(f"📈 Total Return: {((metrics.current_balance / 2.0) - 1) * 100:.1f}%")
                self.logger.info(f"🏆 Total Trades: {metrics.total_trades}")
                self.logger.info(f"🎯 Win Rate: {metrics.win_rate:.1f}%")
                self.logger.info(f"📉 Max Drawdown: {metrics.max_drawdown:.1f}%")
            
            self.logger.info("👋 Autonomous Trading System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


def print_banner():
    """Print system banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║               🚀 AUTONOMOUS CRYPTOCURRENCY TRADING AGENT 🚀                  ║
    ║                                                                              ║
    ║                        Powered by Skyscope AI Systems                       ║
    ║                                                                              ║
    ║  🎯 MISSION: Transform $2 USD → $20,000 USD (10,000x Return)                ║
    ║  🧠 AI ENGINE: OpenAI GPT-4 Turbo                                           ║
    ║  ⚡ EXCHANGE: ByBit (Real-time execution)                                    ║
    ║  🛡️ RISK MGMT: Advanced position sizing & risk controls                     ║
    ║  📊 STRATEGY: Ultra-high frequency scalping & momentum trading              ║
    ║                                                                              ║
    ║  ⚠️  WARNING: This is a high-risk, high-reward trading system               ║
    ║      Only use funds you can afford to lose completely                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Cryptocurrency Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autonomous_trader_main.py --config configs/autonomous_trader_config.json
  python autonomous_trader_main.py --config configs/autonomous_trader_config.json --dry-run
  python autonomous_trader_main.py --help
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/autonomous_trader_config.json',
        help='Path to configuration file (default: configs/autonomous_trader_config.json)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no real trades)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Autonomous Trading Agent v2.0.0'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("Please create a configuration file or specify a valid path.")
        sys.exit(1)
    
    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    
    # Create and run the trading system
    try:
        system_manager = TradingSystemManager(args.config)
        
        # Run the system
        asyncio.run(system_manager.initialize())
        if system_manager.trader:
            asyncio.run(system_manager.run())
        else:
            print("❌ Failed to initialize trading system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()