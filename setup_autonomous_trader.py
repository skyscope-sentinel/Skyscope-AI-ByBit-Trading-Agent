#!/usr/bin/env python3
"""
Autonomous Trading Agent Setup Script
===================================

Comprehensive setup and configuration script for the autonomous cryptocurrency trading agent.
This script will:
1. Install all required dependencies
2. Configure API keys and settings
3. Perform system checks
4. Create necessary directories and files
5. Run initial tests

Author: Skyscope AI Trading Systems
Version: 2.0.0
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional


class TradingAgentSetup:
    """Setup manager for the autonomous trading agent"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "configs"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        
        self.config = {}
        self.setup_complete = False
        
    def print_banner(self):
        """Print setup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🛠️  AUTONOMOUS TRADING AGENT SETUP & CONFIGURATION 🛠️             ║
║                                                                              ║
║                        Powered by Skyscope AI Systems                       ║
║                                                                              ║
║  This setup will configure your autonomous trading system for:              ║
║  • OpenAI GPT-4 integration for market analysis                             ║
║  • ByBit API connectivity for real-time trading                             ║
║  • Advanced risk management and position sizing                             ║
║  • Real-time monitoring and performance tracking                            ║
║                                                                              ║
║  🎯 GOAL: Transform $2 USD into $20,000 USD through AI-powered trading      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}.{version.micro}")
            return False
        
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating project directories...")
        
        directories = [
            self.config_dir,
            self.logs_dir,
            self.data_dir,
            self.project_root / "backups",
            self.project_root / "reports"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def install_dependencies(self) -> bool:
        """Install required Python packages"""
        print("📦 Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True)
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ requirements.txt not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def configure_api_keys(self):
        """Interactive API key configuration"""
        print("\n🔑 API Key Configuration")
        print("=" * 50)
        
        # OpenAI API Key
        print("\n🧠 OpenAI API Configuration:")
        print("You need an OpenAI API key with GPT-4 access.")
        print("Get your API key from: https://platform.openai.com/api-keys")
        
        openai_key = input("Enter your OpenAI API key: ").strip()
        if not openai_key or openai_key.startswith("sk-") == False:
            print("⚠️ Warning: OpenAI API key format seems incorrect")
        
        # ByBit API Keys
        print("\n⚡ ByBit API Configuration:")
        print("You need ByBit API credentials with trading permissions.")
        print("Create API keys at: https://www.bybit.com/app/user/api-management")
        print("⚠️ IMPORTANT: Enable 'Contract Trading' permissions!")
        
        use_testnet = input("Use ByBit Testnet for testing? (y/n): ").strip().lower() == 'y'
        
        if use_testnet:
            print("Using ByBit Testnet (recommended for initial testing)")
            bybit_key = input("Enter your ByBit Testnet API key: ").strip()
            bybit_secret = input("Enter your ByBit Testnet API secret: ").strip()
        else:
            print("⚠️ WARNING: Using ByBit Mainnet - REAL MONEY AT RISK!")
            confirm = input("Are you sure you want to use mainnet? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("Switching to testnet for safety...")
                use_testnet = True
                bybit_key = input("Enter your ByBit Testnet API key: ").strip()
                bybit_secret = input("Enter your ByBit Testnet API secret: ").strip()
            else:
                bybit_key = input("Enter your ByBit Mainnet API key: ").strip()
                bybit_secret = input("Enter your ByBit Mainnet API secret: ").strip()
        
        # Store configuration
        self.config = {
            "trading_mode": "TESTNET" if use_testnet else "PRODUCTION",
            "initial_capital_usdt": 2.0,
            "target_capital_usdt": 20000.0,
            "max_daily_trades": 500,
            "max_concurrent_positions": 10,
            
            "bybit_api": {
                "api_key": bybit_key,
                "api_secret": bybit_secret,
                "testnet": use_testnet,
                "recv_window": 5000,
                "max_retries": 3,
                "retry_delay": 1.0
            },
            
            "openai_api": {
                "api_key": openai_key,
                "model": "gpt-4-turbo-preview",
                "max_tokens": 2000,
                "temperature": 0.05,
                "timeout": 30
            },
            
            "trading_symbols": [
                "1000PEPEUSDT", "1000SHIBUSDT", "1000FLOKIUSDT", "1000BONKUSDT",
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"
            ],
            
            "risk_management": {
                "max_risk_per_trade_pct": 5.0,
                "max_portfolio_risk_pct": 25.0,
                "stop_loss_pct": 2.0,
                "take_profit_pct": 6.0,
                "trailing_stop_pct": 1.5,
                "max_leverage": 50,
                "min_leverage": 5,
                "risk_reward_ratio": 3.0,
                "max_drawdown_pct": 15.0,
                "daily_loss_limit_pct": 10.0
            },
            
            "strategy_config": {
                "primary_strategy": "aggressive_scalping",
                "secondary_strategy": "momentum_breakout",
                "fallback_strategy": "mean_reversion",
                "confidence_threshold": 0.75,
                "min_volume_spike": 2.0,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "macd_signal_strength": 0.8,
                "volatility_threshold": 0.02
            },
            
            "execution_config": {
                "order_type": "Market",
                "slippage_tolerance_pct": 0.1,
                "partial_fill_timeout": 5,
                "order_timeout": 30,
                "retry_failed_orders": True,
                "max_order_retries": 3,
                "position_check_interval": 1,
                "market_data_refresh_rate": 0.5
            },
            
            "ai_analysis": {
                "analysis_interval_seconds": 1,
                "market_sentiment_weight": 0.3,
                "technical_analysis_weight": 0.4,
                "volume_analysis_weight": 0.3,
                "multi_timeframe_analysis": True,
                "timeframes": ["1m", "3m", "5m", "15m"],
                "lookback_periods": {
                    "short": 20,
                    "medium": 50,
                    "long": 200
                }
            },
            
            "performance_tracking": {
                "track_pnl": True,
                "track_win_rate": True,
                "track_sharpe_ratio": True,
                "track_max_drawdown": True,
                "performance_log_interval": 300,
                "daily_report": True
            },
            
            "alerts_notifications": {
                "console_logging": True,
                "file_logging": True,
                "log_level": "INFO",
                "alert_on_large_profit": True,
                "alert_on_large_loss": True,
                "profit_alert_threshold_pct": 10.0,
                "loss_alert_threshold_pct": 5.0
            },
            
            "advanced_features": {
                "auto_compound": True,
                "dynamic_position_sizing": True,
                "adaptive_risk_management": True,
                "market_regime_detection": True,
                "volatility_scaling": True,
                "momentum_filtering": True
            },
            
            "system_config": {
                "max_cpu_usage_pct": 80,
                "max_memory_usage_pct": 70,
                "heartbeat_interval": 30,
                "health_check_interval": 60,
                "auto_restart_on_error": True,
                "max_restart_attempts": 5,
                "log_rotation_days": 7,
                "max_log_size_mb": 100
            }
        }
    
    def save_configuration(self):
        """Save configuration to file"""
        print("\n💾 Saving configuration...")
        
        config_file = self.config_dir / "autonomous_trader_config.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Configuration saved to: {config_file}")
            
            # Create environment file for API keys
            env_file = self.project_root / ".env"
            with open(env_file, 'w') as f:
                f.write(f"OPENAI_API_KEY={self.config['openai_api']['api_key']}\n")
                f.write(f"BYBIT_API_KEY={self.config['bybit_api']['api_key']}\n")
                f.write(f"BYBIT_API_SECRET={self.config['bybit_api']['api_secret']}\n")
            
            print(f"✅ Environment file created: {env_file}")
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
        
        return True
    
    def test_api_connections(self) -> bool:
        """Test API connections"""
        print("\n🔍 Testing API connections...")
        
        # Test OpenAI API
        try:
            import openai
            client = openai.OpenAI(api_key=self.config['openai_api']['api_key'])
            
            # Simple test request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for testing
                messages=[{"role": "user", "content": "Hello, respond with 'API test successful'"}],
                max_tokens=10
            )
            
            if "API test successful" in response.choices[0].message.content:
                print("✅ OpenAI API connection successful")
            else:
                print("⚠️ OpenAI API responded but with unexpected content")
                
        except Exception as e:
            print(f"❌ OpenAI API test failed: {e}")
            return False
        
        # Test ByBit API
        try:
            from pybit.unified_trading import HTTP
            
            session = HTTP(
                testnet=self.config['bybit_api']['testnet'],
                api_key=self.config['bybit_api']['api_key'],
                api_secret=self.config['bybit_api']['api_secret']
            )
            
            # Test server time
            server_time = session.get_server_time()
            if server_time.get('retCode') == 0:
                print("✅ ByBit API connection successful")
            else:
                print(f"❌ ByBit API error: {server_time}")
                return False
                
        except Exception as e:
            print(f"❌ ByBit API test failed: {e}")
            return False
        
        return True
    
    def create_startup_scripts(self):
        """Create startup scripts for different platforms"""
        print("\n📜 Creating startup scripts...")
        
        # Linux/Mac startup script
        if platform.system() in ['Linux', 'Darwin']:
            startup_script = self.project_root / "start_trader.sh"
            with open(startup_script, 'w') as f:
                f.write(f"""#!/bin/bash
# Autonomous Trading Agent Startup Script

echo "🚀 Starting Autonomous Trading Agent..."
cd "{self.project_root}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Set environment variables
export PYTHONPATH="{self.project_root}:$PYTHONPATH"

# Start the trading agent
python autonomous_trader_main.py --config configs/autonomous_trader_config.json

echo "👋 Trading agent stopped"
""")
            
            # Make executable
            os.chmod(startup_script, 0o755)
            print(f"✅ Created startup script: {startup_script}")
        
        # Windows startup script
        if platform.system() == 'Windows':
            startup_script = self.project_root / "start_trader.bat"
            with open(startup_script, 'w') as f:
                f.write(f"""@echo off
REM Autonomous Trading Agent Startup Script

echo 🚀 Starting Autonomous Trading Agent...
cd /d "{self.project_root}"

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
    echo ✅ Virtual environment activated
)

REM Set environment variables
set PYTHONPATH={self.project_root};%PYTHONPATH%

REM Start the trading agent
python autonomous_trader_main.py --config configs\\autonomous_trader_config.json

echo 👋 Trading agent stopped
pause
""")
            print(f"✅ Created startup script: {startup_script}")
    
    def display_final_instructions(self):
        """Display final setup instructions"""
        print("\n" + "=" * 80)
        print("🎉 SETUP COMPLETE! 🎉")
        print("=" * 80)
        
        print("\n📋 NEXT STEPS:")
        print("1. Review your configuration in: configs/autonomous_trader_config.json")
        print("2. Start with testnet mode to verify everything works")
        print("3. Monitor the first few trades carefully")
        print("4. Only switch to mainnet when you're confident")
        
        print("\n🚀 TO START TRADING:")
        if platform.system() in ['Linux', 'Darwin']:
            print("   ./start_trader.sh")
        elif platform.system() == 'Windows':
            print("   start_trader.bat")
        else:
            print("   python autonomous_trader_main.py --config configs/autonomous_trader_config.json")
        
        print("\n⚠️  IMPORTANT WARNINGS:")
        print("• This system trades with REAL MONEY (unless using testnet)")
        print("• Cryptocurrency trading is extremely risky")
        print("• You could lose your entire investment")
        print("• Only invest what you can afford to lose completely")
        print("• Monitor the system regularly")
        print("• The AI makes autonomous decisions - be prepared!")
        
        print("\n📊 MONITORING:")
        print("• Check logs in: logs/")
        print("• Performance reports in: reports/")
        print("• Configuration in: configs/")
        
        print("\n🆘 SUPPORT:")
        print("• Read the documentation carefully")
        print("• Test thoroughly in testnet mode first")
        print("• Start with small amounts")
        print("• Monitor system performance closely")
        
        print("\n" + "=" * 80)
        print("🎯 MISSION: Turn $2 into $20,000 through AI-powered trading!")
        print("🧠 May the AI be with you! 🧠")
        print("=" * 80)
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        print("\n🔧 Starting setup process...\n")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            print("❌ Setup failed during dependency installation")
            return False
        
        # Configure API keys
        self.configure_api_keys()
        
        # Save configuration
        if not self.save_configuration():
            print("❌ Setup failed during configuration save")
            return False
        
        # Test API connections
        if not self.test_api_connections():
            print("⚠️ API connection tests failed - please check your keys")
            print("You can still proceed, but trading may not work properly")
            
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return False
        
        # Create startup scripts
        self.create_startup_scripts()
        
        # Display final instructions
        self.display_final_instructions()
        
        self.setup_complete = True
        return True


def main():
    """Main setup function"""
    setup = TradingAgentSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Setup completed successfully!")
            return 0
        else:
            print("\n❌ Setup failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())