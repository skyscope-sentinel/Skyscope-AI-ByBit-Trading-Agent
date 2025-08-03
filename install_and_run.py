#!/usr/bin/env python3
"""
One-Click Installation and Execution Script
==========================================

This script will:
1. Install all dependencies
2. Set up the environment
3. Configure the trading system
4. Start the autonomous trader

Usage: python install_and_run.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def print_banner():
    """Print installation banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           🚀 AUTONOMOUS TRADING AGENT - ONE-CLICK INSTALLER 🚀               ║
    ║                                                                              ║
    ║                        Transform $2 → $20,000 USD                           ║
    ║                                                                              ║
    ║  This installer will set up everything you need to start autonomous         ║
    ║  cryptocurrency trading with AI-powered decision making.                    ║
    ║                                                                              ║
    ║  ⚠️  WARNING: This system trades with real money!                           ║
    ║      Only use funds you can afford to lose completely.                      ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def quick_setup():
    """Quick setup with minimal configuration"""
    print("\n🔧 Quick Setup Mode")
    print("This will create a basic configuration for testing.")
    print("You can modify it later in configs/autonomous_trader_config.json")
    
    # Get API keys
    print("\n🔑 API Keys Required:")
    openai_key = input("OpenAI API Key (get from https://platform.openai.com/api-keys): ").strip()
    bybit_key = input("ByBit API Key: ").strip()
    bybit_secret = input("ByBit API Secret: ").strip()
    
    use_testnet = input("Use ByBit Testnet? (y/n) [recommended]: ").strip().lower() != 'n'
    
    # Create basic config
    config = {
        "trading_mode": "TESTNET" if use_testnet else "PRODUCTION",
        "initial_capital_usdt": 2.0,
        "target_capital_usdt": 20000.0,
        "max_daily_trades": 100,
        "max_concurrent_positions": 5,
        
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
            "1000PEPEUSDT", "1000SHIBUSDT", "BTCUSDT", "ETHUSDT"
        ],
        
        "risk_management": {
            "max_risk_per_trade_pct": 3.0,
            "max_portfolio_risk_pct": 15.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 6.0,
            "max_leverage": 25,
            "min_leverage": 5,
            "max_drawdown_pct": 10.0,
            "daily_loss_limit_pct": 8.0
        },
        
        "strategy_config": {
            "primary_strategy": "aggressive_scalping",
            "confidence_threshold": 0.75,
            "min_volume_spike": 2.0,
            "volatility_threshold": 0.02
        },
        
        "execution_config": {
            "order_type": "Market",
            "slippage_tolerance_pct": 0.1,
            "position_check_interval": 1,
            "market_data_refresh_rate": 0.5
        },
        
        "ai_analysis": {
            "analysis_interval_seconds": 2,
            "multi_timeframe_analysis": True,
            "timeframes": ["1m", "3m", "5m"],
            "lookback_periods": {
                "short": 20,
                "medium": 50,
                "long": 100
            }
        },
        
        "performance_tracking": {
            "track_pnl": True,
            "track_win_rate": True,
            "performance_log_interval": 300
        },
        
        "alerts_notifications": {
            "console_logging": True,
            "file_logging": True,
            "log_level": "INFO"
        },
        
        "advanced_features": {
            "auto_compound": True,
            "dynamic_position_sizing": True,
            "adaptive_risk_management": True,
            "volatility_scaling": True
        },
        
        "system_config": {
            "max_cpu_usage_pct": 80,
            "max_memory_usage_pct": 70,
            "heartbeat_interval": 30,
            "health_check_interval": 60,
            "auto_restart_on_error": True
        }
    }
    
    # Create directories
    os.makedirs("configs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Save config
    with open("configs/autonomous_trader_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create .env file
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={openai_key}\n")
        f.write(f"BYBIT_API_KEY={bybit_key}\n")
        f.write(f"BYBIT_API_SECRET={bybit_secret}\n")
    
    print("✅ Configuration saved")
    return True


def test_apis():
    """Test API connections"""
    print("\n🔍 Testing API connections...")
    
    try:
        # Test OpenAI
        import openai
        with open("configs/autonomous_trader_config.json") as f:
            config = json.load(f)
        
        client = openai.OpenAI(api_key=config['openai_api']['api_key'])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("✅ OpenAI API working")
        
        # Test ByBit
        from pybit.unified_trading import HTTP
        session = HTTP(
            testnet=config['bybit_api']['testnet'],
            api_key=config['bybit_api']['api_key'],
            api_secret=config['bybit_api']['api_secret']
        )
        
        server_time = session.get_server_time()
        if server_time.get('retCode') == 0:
            print("✅ ByBit API working")
            return True
        else:
            print(f"❌ ByBit API error: {server_time}")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def start_trader():
    """Start the autonomous trader"""
    print("\n🚀 Starting Autonomous Trader...")
    print("=" * 60)
    print("🎯 MISSION: Turn $2 into $20,000")
    print("🧠 AI: OpenAI GPT-4 Turbo")
    print("⚡ EXCHANGE: ByBit")
    print("=" * 60)
    print("\n⚠️  FINAL WARNING:")
    print("This system will make autonomous trading decisions.")
    print("You could lose money. Only proceed if you understand the risks.")
    
    proceed = input("\nStart trading? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("👋 Trading cancelled")
        return False
    
    try:
        # Import and run the trader
        import asyncio
        from autonomous_trader_main import TradingSystemManager
        
        manager = TradingSystemManager("configs/autonomous_trader_config.json")
        
        async def run_system():
            if await manager.initialize():
                await manager.run()
            else:
                print("❌ Failed to initialize trading system")
        
        asyncio.run(run_system())
        
    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True


def main():
    """Main installation and execution"""
    print_banner()
    
    # Check Python version
    if not check_python():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Quick setup
    if not quick_setup():
        return 1
    
    # Test APIs
    if not test_apis():
        print("⚠️ API tests failed. Please check your keys.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return 1
    
    print("\n✅ Installation complete!")
    print("\n📋 What's been set up:")
    print("• All dependencies installed")
    print("• Configuration file created")
    print("• API connections tested")
    print("• Directories created")
    print("• Environment configured")
    
    # Ask if user wants to start trading now
    start_now = input("\nStart trading now? (y/n): ").strip().lower()
    if start_now == 'y':
        start_trader()
    else:
        print("\n🎯 To start trading later, run:")
        print("python autonomous_trader_main.py --config configs/autonomous_trader_config.json")
    
    print("\n🎉 Setup complete! Good luck with your trading journey!")
    return 0


if __name__ == "__main__":
    sys.exit(main())