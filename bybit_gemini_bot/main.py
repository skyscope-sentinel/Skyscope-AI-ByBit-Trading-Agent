import logging
import os
import time  # Added for polling sleep

# Assuming logger_setup.py and config.py are in the same directory (bybit_gemini_bot)
from .logger_setup import setup_logging, DEFAULT_LOG_FILE
from .config import load_config, DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH
from .bybit_connector import BybitConnector
from .data_handler import DataHandler  # Added
from .gemini_analyzer import GeminiAnalyzer  # Added
from .risk_manager import RiskManager  # Added
from .strategy_engine import StrategyEngine  # Added


def main():
    """
    Main function to initialize and start the trading bot.
    """
    setup_logging(log_level=logging.INFO, log_file=DEFAULT_LOG_FILE)
    logger = logging.getLogger(__name__)

    logger.info("=============================================")
    logger.info("🚀 Trading Bot Starting Up...")
    logger.info("=============================================")

    config_file_to_load = DEFAULT_CONFIG_PATH
    if not os.path.exists(config_file_to_load):
        logger.warning(
            f"'{config_file_to_load}' not found. Looking for example config."
        )
        config_file_to_load = EXAMPLE_CONFIG_PATH
        if not os.path.exists(config_file_to_load):
            logger.error(
                f"Neither '{DEFAULT_CONFIG_PATH}' nor '{EXAMPLE_CONFIG_PATH}' found. Bot shutting down."
            )
            return
    logger.info(f"Attempting to load configuration from: '{config_file_to_load}'")
    config = load_config(config_file_to_load)
    if not config:
        logger.error("Failed to load configuration. Bot shutting down.")
        return
    logger.info("Configuration loaded successfully.")

    masked_config_str = "Loaded Configuration:\n"
    for key, value in config.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            if key == "gemini_api_key" and value:
                masked_value = (
                    f"{str(value)[:5]}...{str(value)[-4:]}"
                    if len(str(value)) > 9
                    else "*" * len(str(value))
                )
            elif value:
                masked_value = (
                    f"{str(value)[:3]}...{str(value)[-3:]}"
                    if len(str(value)) > 6
                    else "*" * len(str(value))
                )
            else:
                masked_value = "NOT_SET"
            masked_config_str += f"  {key}: {masked_value}\n"
        elif key in [
            "trading_symbols",
            "risk_percentage",
            "dry_run_mode",
            "use_websockets",
            "polling_interval_seconds",
        ]:
            masked_config_str += f"  {key}: {value}\n"
        else:
            masked_config_str += f"  {key}: Present\n"
    logger.info(masked_config_str)

    # --- Initialize Components ---
    data_handler = DataHandler(config_logging=False)
    risk_manager = RiskManager(config=config, config_logging=False)

    # Bybit Connector Initialization
    bybit_conn_for_se = None
    bybit_conn_initialized = False
    use_bybit_connector_flag = config.get(
        "use_bybit_testnet_connector", False
    )  # Assuming testnet for now

    if use_bybit_connector_flag:
        logger.info("Attempting to initialize Bybit Connector...")
        api_key = config.get("testnet_api_key_bybit")  # Adjust if mainnet keys are used
        api_secret = config.get("testnet_api_secret_bybit")
        if (
            api_key
            and "YOUR_" not in api_key
            and api_secret
            and "YOUR_" not in api_secret
        ):
            bybit_conn = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
                config_logging=False,
            )
            if bybit_conn.session:
                logger.info("Bybit Connector initialized successfully for REST.")
                bybit_conn_initialized = True
                bybit_conn_for_se = bybit_conn
                # Conceptual WebSocket setup (if enabled in config)
                if config.get("use_websockets", False):
                    data_handler_callback_map = {
                        "kline": data_handler.handle_ws_kline,
                        "orderbook": data_handler.handle_ws_orderbook,
                        "public_trade": data_handler.handle_ws_public_trade,
                        "orders": data_handler.handle_ws_order_update,
                        "positions": data_handler.handle_ws_position_update,
                    }
                    logger.info("--- Conceptual Bybit WebSocket Setup ---")
                    bybit_conn.connect_public_ws(data_handler_callback_map)
                    # bybit_conn.subscribe_to_kline_public(symbol="BTCUSDT", interval_minutes=1, category="linear") # Example
                    bybit_conn.start_public_ws()
                    bybit_conn.connect_private_ws(data_handler_callback_map)
                    if bybit_conn.ws_private:
                        # bybit_conn.subscribe_to_orders_private()
                        # bybit_conn.subscribe_to_positions_private()
                        bybit_conn.start_private_ws()
                    logger.info("--- Finished Conceptual Bybit WebSocket Setup ---")
            else:
                logger.error("Bybit Connector session FAILED to initialize.")
        else:
            logger.warning(
                "Bybit API key/secret not found or are placeholders. Skipping Bybit Connector initialization."
            )
    else:
        logger.info("Bybit connector usage is disabled in config.")

    if not bybit_conn_initialized:
        logger.warning(
            "Bybit connector not initialized. StrategyEngine will use a dummy Bybit connector."
        )

        class DummyBybit:
            session = None
            ws_public = None
            ws_private = None
            logger = logging.getLogger("DummyBybit")

            def __init__(self, *args, **kwargs):
                pass

            def connect_public_ws(self, cb_map):
                self.logger.info("DummyBybit: connect_public_ws")

            def start_public_ws(self):
                self.logger.info("DummyBybit: start_public_ws")

            def connect_private_ws(self, cb_map):
                self.logger.info("DummyBybit: connect_private_ws")

            def start_private_ws(self):
                self.logger.info("DummyBybit: start_private_ws")

            def get_kline(self, **kwargs):
                self.logger.info(f"Dummy: get_kline {kwargs}")
                return {"retCode": 1, "result": {"list": []}}

            def get_tickers(self, **kwargs):
                self.logger.info(f"Dummy: get_tickers {kwargs}")
                return {"retCode": 1, "result": {"list": [{"lastPrice": "0"}]}}

            def get_wallet_balance(self, **kwargs):
                self.logger.info(f"Dummy: get_wallet_balance {kwargs}")
                return {
                    "retCode": 1,
                    "result": {
                        "list": [
                            {"coin": [{"coin": "USDT", "availableToWithdraw": "0"}]}
                        ]
                    },
                }

            def get_positions(self, **kwargs):
                self.logger.info(f"Dummy: get_positions {kwargs}")
                return {"retCode": 1, "result": {"list": []}}

            def get_instrument_info(self, **kwargs):
                self.logger.info(f"Dummy: get_instrument_info {kwargs}")
                return {
                    "retCode": 1,
                    "result": {"list": [{"lotSizeFilter": {}, "priceFilter": {}}]},
                }

            def get_fee_rates(self, **kwargs):
                self.logger.info(f"Dummy: get_fee_rates {kwargs}")
                return {
                    "retCode": 1,
                    "result": {"list": [{"takerFeeRate": "0", "makerFeeRate": "0"}]},
                }

            def place_order(self, **kwargs):
                self.logger.info(f"Dummy: place_order {kwargs}")
                return {"retCode": 0, "retMsg": "OK (Dummy)"}

        bybit_conn_for_se = DummyBybit()

    # Gemini Analyzer Initialization
    gemini_analyzer_for_se = None
    gemini_analyzer_initialized = False
    use_gemini_analyzer_flag = config.get("use_gemini_analyzer", False)

    if use_gemini_analyzer_flag:
        logger.info("Attempting to initialize Gemini Analyzer...")
        gemini_api_key = config.get("gemini_api_key")
        if gemini_api_key and "YOUR_" not in gemini_api_key:
            gemini_analyzer = GeminiAnalyzer(
                api_key=gemini_api_key, config_logging=False
            )  # Model name from its default
            if gemini_analyzer.model:
                logger.info("Gemini Analyzer initialized successfully.")
                gemini_analyzer_initialized = True
                gemini_analyzer_for_se = gemini_analyzer
            else:
                logger.error(
                    "Gemini Analyzer FAILED to initialize (model not created)."
                )
        else:
            logger.warning(
                "Gemini API key not found/placeholder. Skipping Gemini Analyzer initialization."
            )
    else:
        logger.info("Gemini Analyzer usage is disabled in config.")

    if not gemini_analyzer_initialized:
        logger.warning(
            "Gemini analyzer not initialized. StrategyEngine will use a dummy Gemini analyzer."
        )

        class DummyGemini:
            model = None
            logger = logging.getLogger("DummyGemini")

            def __init__(self, *args, **kwargs):
                pass

            def analyze_market_data(self, prompt):
                self.logger.info(
                    "DummyGemini: analyze_market_data called. Returning HOLD."
                )
                return {
                    "decision": "HOLD",
                    "confidence_score": 0.5,
                    "reasoning": "Dummy response",
                }

        gemini_analyzer_for_se = DummyGemini()

    # Strategy Engine Initialization
    strategy_engine = StrategyEngine(
        config=config,
        bybit_connector=bybit_conn_for_se,
        gemini_analyzer=gemini_analyzer_for_se,
        data_handler=data_handler,
        risk_manager=risk_manager,
        config_logging=False,
    )
    logger.info("StrategyEngine initialized.")

    # --- Main Trading Loop (Polling Example) ---
    if not config.get(
        "use_websockets", False
    ):  # Run polling loop if WebSockets are not primary
        logger.info("Starting polling-based trading loop...")
        trading_symbols_list = config.get("trading_symbols", [])
        if not trading_symbols_list:
            logger.warning("No trading_symbols configured. Exiting main trading loop.")

        dry_run_mode = config.get("dry_run_mode", True)
        if dry_run_mode:
            logger.info("DRY RUN MODE ENABLED: No actual orders will be placed.")
        else:
            logger.info("LIVE TRADING MODE ENABLED: Actual orders will be placed.")

        max_cycles_per_symbol = config.get("max_test_cycles_per_symbol", 1)

        for symbol_to_trade in trading_symbols_list:
            logger.info(f"Processing symbol: {symbol_to_trade}")
            if not bybit_conn_initialized and not dry_run_mode:
                logger.warning(
                    f"Skipping LIVE trading cycle for {symbol_to_trade} as Bybit connector is not properly initialized."
                )
                continue
            if use_gemini_analyzer_flag and not gemini_analyzer_initialized:
                logger.warning(
                    f"Skipping trading cycle for {symbol_to_trade} as Gemini Analyzer is enabled in config but failed to initialize."
                )
                continue

            for cycle_num in range(max_cycles_per_symbol):
                logger.info(
                    f"--- Starting cycle {cycle_num + 1}/{max_cycles_per_symbol} for {symbol_to_trade} ---"
                )
                try:
                    strategy_engine.run_main_cycle(symbol_to_trade)
                except Exception as e_cycle:
                    logger.error(
                        f"Exception during trading cycle for {symbol_to_trade}: {e_cycle}",
                        exc_info=True,
                    )

                if max_cycles_per_symbol > 1 and cycle_num < max_cycles_per_symbol - 1:
                    polling_interval = config.get("polling_interval_seconds", 180)
                    logger.info(
                        f"Cycle finished for {symbol_to_trade}. Sleeping for {polling_interval}s..."
                    )
                    time.sleep(polling_interval)
            logger.info(f"Finished all configured cycles for symbol {symbol_to_trade}.")
        logger.info("All symbols processed for configured polling cycles.")
    else:  # WebSocket mode
        logger.info(
            "WebSocket mode is enabled. Bot would listen for WebSocket events here."
        )
        logger.info(
            "Ensure WebSocket connections are started and StrategyEngine.process_websocket_update is triggered by DataHandler/callbacks."
        )
        # In a real WS setup, the main thread might wait or manage other tasks.
        # For this conceptual phase, we'll just log and exit if no polling.
        if (
            not bybit_conn_initialized
        ):  # Still need connector for any potential REST fallbacks or actions
            logger.warning(
                "Bybit connector not initialized, WebSocket dependent actions might fail."
            )

    logger.info("Main bot operational logic finished.")

    try:
        if config.get("use_websockets", False) and bybit_conn_initialized:
            logger.info(
                "WebSocket mode: Bot would typically keep running here (e.g., time.sleep( लंबे समय तक) or join threads)."
            )
            # time.sleep(3600) # Example to keep alive for WS; not for this subtask's test
        pass
    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in main's final block: {e}", exc_info=True
        )
    finally:
        logger.info("=============================================")
        logger.info("👋 Trading Bot Shutting Down (Main Script Finished)...")
        logger.info("=============================================")
        logging.shutdown()


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    DEFAULT_CONFIG_PATH = os.path.join(project_root, "configs", "config.dev.json")
    EXAMPLE_CONFIG_PATH = os.path.join(project_root, "configs", "config.json.example")
    configs_dir_for_main = os.path.join(project_root, "configs")

    if not os.path.exists(configs_dir_for_main):
        os.makedirs(configs_dir_for_main)
        print(
            f"[Setup for main.py direct run] Created '{configs_dir_for_main}' directory."
        )

    if not os.path.exists(EXAMPLE_CONFIG_PATH):  # Corrected path here
        example_data = {
            "bybit_api_key": "YOUR_BYBIT_API_KEY_EXAMPLE",
            "bybit_api_secret": "YOUR_BYBIT_API_SECRET_EXAMPLE",
            "gemini_api_key": "YOUR_GEMINI_API_KEY_EXAMPLE",
            "testnet_api_key_bybit": "YOUR_BYBIT_TESTNET_API_KEY_EXAMPLE",
            "testnet_api_secret_bybit": "YOUR_BYBIT_TESTNET_API_SECRET_EXAMPLE",
            "mainnet_api_key_bybit": "YOUR_BYBIT_MAINNET_API_KEY_EXAMPLE",
            "mainnet_api_secret_bybit": "YOUR_BYBIT_MAINNET_API_SECRET_EXAMPLE",
            "trading_symbols": ["BTCUSDT", "ETHUSDT"],
            "risk_per_trade_percentage": 1.0,
            "stop_loss_percentage": 0.02,
            "tp_fee_multiplier": 3.0,
            "risk_reward_ratio": 1.5,
            "taker_fee_rate": 0.00055,
            "use_bybit_testnet_connector": True,  # Default to True for easy testing
            "use_gemini_analyzer": True,  # Default to True
            "use_websockets": False,
            "ai_confidence_threshold": 0.6,
            "min_kline_data_points": 20,
            "polling_interval_seconds": 180,
            "max_test_cycles_per_symbol": 1,
            "dry_run_mode": True,
        }
        with open(EXAMPLE_CONFIG_PATH, "w") as f:  # Corrected path here
            import json

            json.dump(example_data, f, indent=2)
        print(f"[Setup for main.py direct run] Created dummy '{EXAMPLE_CONFIG_PATH}'.")

    if not os.path.exists(DEFAULT_CONFIG_PATH):  # Corrected path here
        print(f"[Setup for main.py direct run] '{DEFAULT_CONFIG_PATH}' not found.")
        if os.path.exists(EXAMPLE_CONFIG_PATH):  # Corrected path here
            import shutil

            shutil.copy(
                EXAMPLE_CONFIG_PATH, DEFAULT_CONFIG_PATH
            )  # Corrected paths here
            print(
                f"[Setup for main.py direct run] Copied '{EXAMPLE_CONFIG_PATH}' to '{DEFAULT_CONFIG_PATH}'."
            )
            print(
                f"[Setup for main.py direct run] IMPORTANT: Edit '{DEFAULT_CONFIG_PATH}' with your real API keys."
            )
        else:
            print(
                f"[Setup for main.py direct run] CRITICAL: Example config '{EXAMPLE_CONFIG_PATH}' also missing."
            )

    main()
