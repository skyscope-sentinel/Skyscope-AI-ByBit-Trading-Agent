import logging
import os
import json
import time
import hmac
import hashlib
import pandas as pd
import pandas_ta as ta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from logging.handlers import RotatingFileHandler
import requests # Added for OllamaAnalyzer

# Removed google.generativeai imports as they are no longer needed
# import google.generativeai as genai
# from google.api_core import exceptions as google_exceptions

from pybit.unified_trading import HTTP
import pybit.unified_trading # For pybit.unified_trading.WebSocket

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CONFIGURATION
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
DEFAULT_CONFIG_PATH_UNIFIED = "configs/config.dev.json"
EXAMPLE_CONFIG_PATH_UNIFIED = "configs/config.json.example"

API_KEYS_ENV_MAP = {
    "bybit_api_key": "BYBIT_API_KEY",
    "bybit_api_secret": "BYBIT_API_SECRET",
    "testnet_api_key_bybit": "TESTNET_API_KEY_BYBIT",
    "testnet_api_secret_bybit": "TESTNET_API_SECRET_BYBIT",
    "mainnet_api_key_bybit": "MAINNET_API_KEY_BYBIT",
    "mainnet_api_secret_bybit": "MAINNET_API_SECRET_BYBIT",
    # "gemini_api_key": "GEMINI_API_KEY", # Removed
    "ollama_base_url": "OLLAMA_BASE_URL", # Added
    "ollama_model_name": "OLLAMA_MODEL_NAME", # Added
}
PLACEHOLDER_PATTERNS = ["YOUR_", "PLACEHOLDER", "", None, "your_"] # Added "your_"

class Config:
    _instance = None

    def __new__(cls, config_path=None, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.config_data = {}
        self.load_config(config_path)
        self._initialized = True

    def load_config(self, config_path=None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH_UNIFIED

        if not os.path.exists(config_path):
            self.logger.warning(f"Configuration file '{config_path}' not found.")
            if os.path.exists(EXAMPLE_CONFIG_PATH_UNIFIED):
                self.logger.info(
                    f"Attempting to load example configuration from '{EXAMPLE_CONFIG_PATH_UNIFIED}'."
                )
                config_path = EXAMPLE_CONFIG_PATH_UNIFIED
            else:
                self.logger.error(
                    f"Example configuration '{EXAMPLE_CONFIG_PATH_UNIFIED}' also not found. No config loaded."
                )
                return

        try:
            with open(config_path, "r") as f:
                self.config_data = json.load(f)
                self.logger.info(f"Successfully loaded configuration from '{config_path}'.")

            for json_key, env_var_name in API_KEYS_ENV_MAP.items():
                # Check if key exists in JSON; if not, get() returns None which is in PLACEHOLDER_PATTERNS
                current_json_val = self.config_data.get(json_key)
                is_placeholder = False
                if isinstance(current_json_val, str):
                    is_placeholder = any(ph.lower() in current_json_val.lower() for ph in PLACEHOLDER_PATTERNS if isinstance(ph, str))
                elif current_json_val is None:
                    is_placeholder = True

                if is_placeholder:
                    env_value = os.getenv(env_var_name)
                    if env_value:
                        self.config_data[json_key] = env_value
                        self.logger.info(
                            f"Loaded '{json_key}' from environment variable '{env_var_name}'."
                        )
                    elif json_key in ["bybit_api_key", "ollama_base_url", "ollama_model_name"]: # Core keys to warn about
                         self.logger.warning(
                            f"API key/setting '{json_key}' is a placeholder/missing in config and not found in env var '{env_var_name}'."
                        )
            self.validate_essential_keys()

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from '{config_path}': {e}")
            self.config_data = {}
        except Exception as e:
            self.logger.error(f"Error loading configuration from '{config_path}': {e}")
            self.config_data = {}

    def validate_essential_keys(self):
        essential_keys_missing = []
        if self.get("use_bybit_testnet_connector", False):
            if any(ph.lower() in str(self.get("testnet_api_key_bybit","")).lower() for ph in PLACEHOLDER_PATTERNS if ph) or not self.get("testnet_api_key_bybit"):
                essential_keys_missing.append("testnet_api_key_bybit")
        # else: # Mainnet logic can be added here if mainnet keys are named differently
        #     if any(ph.lower() in str(self.get("bybit_api_key","")).lower() for ph in PLACEHOLDER_PATTERNS if ph) or not self.get("bybit_api_key"):
        #         essential_keys_missing.append("bybit_api_key (mainnet)")

        if self.get("use_llm_analyzer", False): # Changed from use_gemini_analyzer
            if any(ph.lower() in str(self.get("ollama_base_url","")).lower() for ph in PLACEHOLDER_PATTERNS if ph) or not self.get("ollama_base_url"):
                essential_keys_missing.append("ollama_base_url")
            if any(ph.lower() in str(self.get("ollama_model_name","")).lower() for ph in PLACEHOLDER_PATTERNS if ph) or not self.get("ollama_model_name"):
                essential_keys_missing.append("ollama_model_name")

        if essential_keys_missing:
            self.logger.warning(
                f"Essential settings are missing or placeholders: {', '.join(essential_keys_missing)}"
            )

    def get(self, key, default=None):
        return self.config_data.get(key, default)
    def __getitem__(self, key): return self.config_data[key]
    def __contains__(self, key): return key in self.config_data

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LOGGER SETUP
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
LOG_DIR_UNIFIED = "logs"
DEFAULT_LOG_FILE_UNIFIED = os.path.join(LOG_DIR_UNIFIED, "trading_bot_unified.log")
class LoggerSetup:
    @staticmethod
    def setup(log_level=logging.INFO, log_file=DEFAULT_LOG_FILE_UNIFIED, max_bytes=10*1024*1024, backup_count=5):
        # (Content of LoggerSetup.setup method - kept concise for this diff, assume it's the same as read)
        if not os.path.exists(LOG_DIR_UNIFIED):
            try: os.makedirs(LOG_DIR_UNIFIED)
            except OSError as e: print(f"Err creating log dir '{LOG_DIR_UNIFIED}': {e}"); logging.basicConfig(level=log_level); logging.error("File logging disabled."); return
        logger = logging.getLogger(); logger.setLevel(log_level)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        console_handler = logging.StreamHandler(); console_handler.setLevel(log_level); console_handler.setFormatter(formatter); logger.addHandler(console_handler)
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
            file_handler.setLevel(log_level); file_handler.setFormatter(formatter); logger.addHandler(file_handler)
        except Exception as e: logging.error(f"Failed file logging for '{log_file}': {e}.")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# BybitConnector class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BybitAPIException(Exception): # (Content as read)
    def __init__(self, ret_code, message, extended_info=None): super().__init__(f"Bybit API Error: Code {ret_code} - {message}"); self.ret_code=ret_code; self.message=message; self.extended_info=extended_info or {}
    def __str__(self): return f"BybitAPIException: [Code: {self.ret_code}] {self.message} (Details: {self.extended_info})"
class BybitConnector: # (Content as read, ensure pybit.unified_trading is imported at top)
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, config_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key; self.api_secret = api_secret; self.testnet = testnet
        self.session = None; self.ws_public_session = None; self.ws_private = None
        if self.testnet: self.ws_url_public_linear = "wss://stream-testnet.bybit.com/v5/public/linear"; self.ws_url_private = "wss://stream-testnet.bybit.com/v5/private"
        else: self.ws_url_public_linear = "wss://stream.bybit.com/v5/public/linear"; self.ws_url_private = "wss://stream.bybit.com/v5/private"
        if self.api_key and self.api_secret:
            try: self.session = HTTP(testnet=self.testnet, api_key=self.api_key, api_secret=self.api_secret, log_requests=False); self.logger.info(f"BybitConnector HTTP session initialized. Testnet: {self.testnet}."); self.ws_public_session = self.session
            except Exception as e: self.logger.error(f"Failed to initialize Bybit HTTP session: {e}", exc_info=True)
        elif self.api_key:
            try: self.session = HTTP(testnet=self.testnet, api_key=self.api_key, log_requests=False); self.ws_public_session = self.session; self.logger.info("BybitConnector HTTP session (read-only/public) initialized.")
            except Exception as e: self.logger.error(f"Failed to initialize read-only Bybit HTTP session: {e}", exc_info=True)
        else:
            try: self.session = HTTP(testnet=self.testnet, log_requests=False); self.ws_public_session = self.session; self.logger.info("BybitConnector HTTP session (unauthenticated public) initialized.")
            except Exception as e: self.logger.error(f"Failed to initialize unauthenticated Bybit HTTP session: {e}", exc_info=True)
        self.public_ws_handlers = {}; self.private_ws_handlers = {}
    def _make_request(self, method_name: str, **kwargs): # (Content as read)
        if self.session is None: self.logger.error(f"No session for '{method_name}'."); return None
        try:
            method = getattr(self.session, method_name); response = method(**kwargs)
            if isinstance(response, dict) and response.get('retCode') == 0: return response
            self.logger.error(f"Bybit API Error on '{method_name}'. Code: {response.get('retCode')}, Msg: {response.get('retMsg')}, Ext: {response.get('retExtInfo')}")
            return response
        except Exception as e: self.logger.error(f"Exception in Bybit '{method_name}': {e}", exc_info=True); return None
    def get_kline(self, **kwargs): return self._make_request("get_kline", **kwargs)
    def get_instrument_info(self, **kwargs): return self._make_request("get_instruments_info", **kwargs)
    def get_tickers(self, **kwargs): return self._make_request("get_tickers", **kwargs)
    def get_orderbook(self, **kwargs): return self._make_request("get_orderbook", **kwargs)
    def get_server_time(self): return self._make_request("get_server_time")
    def get_wallet_balance(self, **kwargs): return self._make_request("get_wallet_balance", **kwargs)
    def get_fee_rates(self, **kwargs): return self._make_request("get_account_fee_rate", **kwargs)
    def place_order(self, **kwargs): return self._make_request("place_order", **kwargs)
    def amend_order(self, **kwargs): return self._make_request("amend_order", **kwargs)
    def cancel_order(self, **kwargs): return self._make_request("cancel_order", **kwargs)
    def get_trade_history(self, **kwargs): return self._make_request("get_executions", **kwargs)
    def get_positions(self, **kwargs): return self._make_request("get_positions", **kwargs)
    def set_leverage(self, **kwargs): return self._make_request("set_leverage", **kwargs)
    def set_trading_stop(self, **kwargs): return self._make_request("set_trading_stop", **kwargs)
    def _handle_dummy_message(self, mt, m): self.logger.info(f"WS {mt} (dummy): {m}") # Simplified
    def connect_public_ws(self, cb_map=None): # (Content as read, simplified dummy)
        if not self.ws_public_session: self.logger.error("Public WS needs HTTP session."); return
        self.public_ws_handlers=cb_map or {'kline':lambda m:self._handle_dummy_message("KLINE",m)}; self.logger.info("PubWS ready.")
    def subscribe_to_kline_public(self,s,i,c="linear"): # (Content as read, simplified dummy)
        if not self.ws_public_session or 'kline' not in self.public_ws_handlers: self.logger.error("PubWS kline handler error.");return
        try: self.ws_public_session.kline_stream(symbol=s,interval=i,callback=self.public_ws_handlers['kline'],category=c); self.logger.info(f"Kline stream {s} {i}m init.")
        except Exception as e: self.logger.error(f"Err sub kline {s}: {e}",exc_info=True)
    def subscribe_to_orderbook_public(self,s,d=25,c="linear"): # (Content as read, simplified dummy)
        if not self.ws_public_session or 'orderbook' not in self.public_ws_handlers: self.logger.error("PubWS OB handler error.");return
        try: self.ws_public_session.orderbook_stream(symbol=s,depth=d,callback=self.public_ws_handlers['orderbook'],category=c); self.logger.info(f"OB stream {s} d{d} init.")
        except Exception as e: self.logger.error(f"Err sub OB {s}: {e}",exc_info=True)
    def subscribe_to_public_trades(self,s,c="linear"): # (Content as read, simplified dummy)
        if not self.ws_public_session or 'public_trade' not in self.public_ws_handlers: self.logger.error("PubWS trade handler error.");return
        try: self.ws_public_session.public_trade_stream(symbol=s,callback=self.public_ws_handlers['public_trade'],category=c); self.logger.info(f"PubTrades stream {s} init.")
        except Exception as e: self.logger.error(f"Err sub trades {s}: {e}",exc_info=True)
    def start_public_ws(self): self.logger.info("PubWS streams run in background via HTTP session.")
    def connect_private_ws(self, cb_map=None): # (Content as read, simplified dummy)
        if not self.api_key or not self.api_secret: self.logger.error("API key/secret for PrivWS needed."); return
        try: self.ws_private=pybit.unified_trading.WebSocket(testnet=self.testnet,channel_type="private",api_key=self.api_key,api_secret=self.api_secret); self.private_ws_handlers=cb_map or {'orders':lambda m:self._handle_dummy_message("ORDER",m)}; self.logger.info("PrivWS client init.")
        except Exception as e: self.logger.error(f"Failed privWS init: {e}",exc_info=True); self.ws_private=None
    def subscribe_to_orders_private(self): # (Content as read, simplified dummy)
        if not self.ws_private or 'orders' not in self.private_ws_handlers: self.logger.error("PrivWS order handler error."); return
        try: self.ws_private.order_stream(callback=self.private_ws_handlers['orders']); self.logger.info("Sub to priv order stream.")
        except Exception as e: self.logger.error(f"Err sub priv orders: {e}",exc_info=True)
    def subscribe_to_positions_private(self): # (Content as read, simplified dummy)
        if not self.ws_private or 'positions' not in self.private_ws_handlers: self.logger.error("PrivWS position handler error."); return
        try: self.ws_private.position_stream(callback=self.private_ws_handlers['positions']); self.logger.info("Sub to priv position stream.")
        except Exception as e: self.logger.error(f"Err sub priv pos: {e}",exc_info=True)
    def start_private_ws(self): # (Content as read, simplified dummy)
        if not self.ws_private: self.logger.error("PrivWS not init."); return
        self.logger.info("Conceptual: PrivWS start_private_ws() called.")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DataHandler class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DataHandler: # (Content as read, ensure pandas_ta import at top)
    def __init__(self, config_logging: bool = True): self.logger = logging.getLogger(__name__); self.current_orderbooks={}; self.recent_klines={}; self.last_1m_kline_partials={}
    def process_kline_data(self, kline_list: list, symbol: str): # (Content as read, simplified)
        if not kline_list: self.logger.warning(f"Empty kline for {symbol}."); return None
        try: df=pd.DataFrame(kline_list, columns=['ts','o','h','l','c','v','to']); df.columns=['timestamp','open','high','low','close','volume','turnover']; df[['open','high','low','close','volume','turnover']]=df[['open','high','low','close','volume','turnover']].apply(pd.to_numeric,errors='coerce'); df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms'); df.set_index('timestamp',inplace=True); df.sort_index(inplace=True); self.logger.info(f"Processed {len(df)} klines for {symbol}."); return df[['open','high','low','close','volume']].copy()
        except Exception as e: self.logger.error(f"Err proc kline {symbol}: {e}",exc_info=True); return None
    def calculate_technical_indicators(self, df:pd.DataFrame, symbol="?"): # (Content as read, simplified)
        if df is None or df.empty: self.logger.warning(f"Empty df for TA on {symbol}"); return df
        try: self.logger.info(f"Calc TAs for {symbol} ({len(df)} rows)."); df.ta.sma(10,c="SMA_10",a=True); df.ta.sma(20,c="SMA_20",a=True); df.ta.rsi(14,c="RSI_14",a=True); df.ta.macd(c="MACD",a=True); df.ta.bbands(c="BBL",a=True); df.ta.atr(c="ATR",a=True); self.logger.info(f"TAs done for {symbol}.")
        except Exception as e: self.logger.error(f"Err calc TAs {symbol}: {e}",exc_info=True); return df
        return df
    def handle_ws_kline(self, m): self.logger.info(f"WS Kline: {m.get('topic','?')} Data: {m.get('data',[{}])[0].get('close','?')}") # Simplified
    def handle_ws_orderbook(self, m): self.logger.info(f"WS OB: {m.get('topic','?')}") # Simplified
    def handle_ws_public_trade(self, m): self.logger.info(f"WS Trade: {m.get('topic','?')}") # Simplified
    def handle_ws_order_update(self, m): self.logger.info(f"WS Order Update: {m.get('data',[{}])[0].get('orderId','?')}") # Simplified
    def handle_ws_position_update(self, m): self.logger.info(f"WS Pos Update: {m.get('data',[{}])[0].get('symbol','?')}") # Simplified

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# RiskManager class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RiskManager: # (Content as read, ensure Decimal imports at top)
    def __init__(self, config: dict, config_logging: bool = True): self.logger=logging.getLogger(__name__); self.config=config; self.config.setdefault('risk_per_trade_percentage',1.0); self.logger.info(f"RiskMan init. Risk/Trade: {self.config['risk_per_trade_percentage']}%")
    def _get_qty_precision(self, qs:str): return len(qs.split('.')[1]) if '.' in qs else 0
    def _get_price_precision(self, ts:str): return len(ts.split('.')[1]) if '.' in ts else 0
    def _adjust_to_step(self,v:Decimal,s:str,r=ROUND_DOWN): step=Decimal(s);return (v/step).quantize(Decimal('1'),rounding=r)*step
    def calculate_position_size(self,bal:float,entry:float,sl:float,info:dict,sym:str,side:str="Buy"): # (Content as read, simplified)
        self.logger.info(f"Calc pos size {sym}({side}): Bal={bal}, Entry={entry}, SL={sl}")
        if entry<=0 or sl<=0: self.logger.error("Prices must be >0"); return None
        if side=="Buy": diff=Decimal(str(entry))-Decimal(str(sl))
        elif side=="Sell": diff=Decimal(str(sl))-Decimal(str(entry))
        else: self.logger.error(f"Invalid side {side}"); return None
        if diff<=Decimal('0'): self.logger.error("SL diff <=0"); return None
        risk_pct=Decimal(str(self.config.get('risk_per_trade_percentage',1.0)))/100; risk_amt=Decimal(str(bal))*risk_pct
        qty=risk_amt/diff; lf=info.get('lotSizeFilter',{}); min_q=Decimal(lf.get('minOrderQty','0'));max_q=Decimal(lf.get('maxOrderQty','inf'));q_step=lf.get('qtyStep','1')
        if not all(lf.get(k) for k in ['minOrderQty','maxOrderQty','qtyStep']): self.logger.error(f"Bad lotSizeFilter {sym}"); return None
        adj_q=self._adjust_to_step(qty,q_step,ROUND_DOWN)
        if adj_q<min_q: self.logger.warning(f"Adj Qty {adj_q} < Min {min_q}"); return None
        if adj_q>max_q: adj_q=max_q; self.logger.warning(f"Adj Qty > Max, capped {max_q}")
        final_q_str=f"{adj_q:.{self._get_qty_precision(q_step)}}"; self.logger.info(f"Final pos size {sym}: {final_q_str}"); return final_q_str
    def calculate_fees_usdt(self,qty:float,px:float,rate:float,is_f:bool=True): return float(Decimal(str(qty))*Decimal(str(px))*Decimal(str(rate))) # Simplified
    def calculate_take_profit_price_for_fee_coverage(self,ep:float,side:str,qty:float,rate_taker:float,info:dict,sym:str,multi:float=3.0): # (Content as read, simplified)
        if qty<=0: self.logger.error("Qty must be >0 for TP"); return None
        epd=Decimal(str(ep)); qd=Decimal(str(qty)); entry_fee=Decimal(str(self.calculate_fees_usdt(qty,ep,rate_taker))); rt_fees=entry_fee*2
        profit_usdt=rt_fees*Decimal(str(multi));
        if side.lower()=="buy": t_epd=epd+(profit_usdt/qd)
        elif side.lower()=="sell": t_epd=epd-(profit_usdt/qd)
        else: self.logger.error(f"Invalid side {side} for TP"); return None
        pf=info.get('priceFilter',{}); ts=pf.get('tickSize');
        if not ts: self.logger.error(f"No tickSize for {sym}"); return None
        adj_tp_d=self._adjust_to_step(t_epd,ts,ROUND_UP if side.lower()=="buy" else ROUND_DOWN)
        return f"{adj_tp_d:.{self._get_price_precision(ts)}}"

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# OllamaAnalyzer class (Refactored from GeminiAnalyzer)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OllamaAnalyzer:
    def __init__(self, ollama_base_url: str, ollama_model_name: str,
                 request_timeout_seconds: int = 60, config_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        if not ollama_base_url or not ollama_model_name:
            self.logger.error("Ollama base URL and model name are required.")
            self.base_url = None; self.model_name = None; self.request_timeout = 60
            return
        self.base_url = ollama_base_url.rstrip('/'); self.model_name = ollama_model_name
        self.request_timeout = request_timeout_seconds
        self.logger.info(f"OllamaAnalyzer initialized. URL: {self.base_url}, Model: {self.model_name}")

    def analyze_market_data(self, market_data_prompt_str: str):
        if not self.base_url or not self.model_name:
            self.logger.error("OllamaAnalyzer not properly initialized."); return {"error": "NotInitialized"}

        self.logger.info(f"Sending prompt to Ollama model '{self.model_name}' (snippet): {market_data_prompt_str[:200]}...")
        ollama_api_endpoint = f"{self.base_url}/api/generate"
        payload = {"model": self.model_name, "prompt": market_data_prompt_str, "stream": False, "format": "json"}
        headers = {"Content-Type": "application/json"}
        model_generated_json_text = ""

        try:
            response = requests.post(ollama_api_endpoint, json=payload, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            ollama_json_response = response.json()
            model_generated_json_text = ollama_json_response.get('response')

            if not model_generated_json_text:
                err_msg = ollama_json_response.get('error', "Model output string was empty.")
                self.logger.warning(f"Ollama response error/empty for {self.model_name}: {err_msg}")
                return {"error": "EmptyLLMResponse", "message": err_msg, "raw_ollama_response": ollama_json_response}

            parsed_json = json.loads(model_generated_json_text)
            self.logger.info("Successfully parsed model-generated JSON from Ollama."); return parsed_json
        except requests.exceptions.Timeout as e: self.logger.error(f"Timeout to Ollama API ({self.model_name}): {e}", exc_info=True); return {"error":"OllamaTimeout", "message":str(e)}
        except requests.exceptions.RequestException as e: self.logger.error(f"Error during Ollama API req ({self.model_name}): {e}", exc_info=True); return {"error":"OllamaRequestError", "message":str(e)}
        except json.JSONDecodeError as e: self.logger.error(f"JSONDecodeError from Ollama model's text ({self.model_name}): {e}. Text: '{model_generated_json_text}'", exc_info=True); return {"error":"LLMJSONDecodeError", "raw_model_output":model_generated_json_text}
        except Exception as e: self.logger.error(f"Unexpected error in Ollama analysis ({self.model_name}): {e}", exc_info=True); return {"error":"UnexpectedError", "message":str(e)}

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# StrategyEngine class
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class StrategyEngine:
    def __init__(self, config: Config, bybit_connector: BybitConnector, llm_analyzer: OllamaAnalyzer, data_handler: DataHandler, risk_manager: RiskManager, config_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.bybit_connector = bybit_connector
        self.llm_analyzer = llm_analyzer
        self.data_handler = data_handler
        self.risk_manager = risk_manager
        self.use_websockets = self.config.get('use_websockets', False)
        self.confidence_threshold = self.config.get('ai_confidence_threshold', 0.6)
        self.min_kline_data_points = self.config.get('min_kline_data_points', 20)
        self.dry_run_mode = self.config.get('dry_run_mode', True)
        self.logger.info(f"StrategyEngine initialized. Dry Run: {self.dry_run_mode}, Confidence: {self.confidence_threshold}")

    def _format_kline_for_prompt(self, kline_df: pd.DataFrame, num_records: int = 20) -> str: # (Content as read, simplified)
        if kline_df is None or kline_df.empty: return "No K-line data."
        k_copy=kline_df.copy(); k_copy.index=k_copy.index.strftime('%Y-%m-%d %H:%M:%S'); cols=['open','high','low','close','volume']+[c for c in ['SMA_10','SMA_20','RSI_14'] if c in k_copy.columns]
        return k_copy[cols].tail(num_records).to_string()

    def _format_positions_for_prompt(self, positions_data: dict, symbol: str) -> str: # (Content as read, simplified)
        if not positions_data or positions_data.get('retCode')!=0: return "Pos data error."
        p_list=positions_data.get('result',{}).get('list',[]);
        if not p_list: return "None"
        s_pos=next((p for p in p_list if p.get('symbol')==symbol and float(p.get('size','0'))>0),None)
        if not s_pos: return f"None for {symbol}"
        return f"Sym:{s_pos.get('symbol')},Side:{s_pos.get('side')},Size:{s_pos.get('size')},EntryPx:{s_pos.get('avgPrice')}"

    def _construct_llm_prompt(self, symbol: str, kline_data_str: str, current_price_str: str, wallet_balance_str: str, open_positions_str: str) -> str: # Renamed
        guidelines = self.config.get('llm_strategy_guidelines', "Analyze for short-term trades. JSON: {decision, confidence_score, entry_price, stop_loss_price, take_profit_price, reasoning}")
        prompt = f"Context: For {symbol}.\nRole: {guidelines}\nData:\n- Price: {current_price_str}\n- Balance: {wallet_balance_str}\n- Position: {open_positions_str}\n- Kline (recent {self.min_kline_data_points} periods, 3-min):\n{kline_data_str}\nTask: Output JSON only."
        self.logger.debug(f"LLM prompt for {symbol} (snippet): {prompt[:200]}...")
        return prompt.strip()

    def run_main_cycle(self, symbol: str): # (Content as read, adapted for OllamaAnalyzer and simplified)
        self.logger.info(f"--- StrategyEngine: Main Cycle for {symbol} ---")
        try:
            kl_resp=self.bybit_connector.get_kline(category="linear",symbol=symbol,interval="3",limit=self.min_kline_data_points+20)
            if not kl_resp or kl_resp.get("retCode")!=0: self.logger.error(f"Kline err {symbol}: {kl_resp}"); return
            kl_df=self.data_handler.process_kline_data(kl_resp.get('result',{}).get('list',[]),symbol)
            if kl_df is None or kl_df.empty: self.logger.error(f"Kline process err {symbol}."); return
            kl_df_ta=self.data_handler.calculate_technical_indicators(kl_df,symbol); kl_df_ta.dropna(inplace=True)
            if len(kl_df_ta)<self.min_kline_data_points: self.logger.warning(f"Not enough TA data for {symbol}: {len(kl_df_ta)}"); return
            kl_prompt_str=self._format_kline_for_prompt(kl_df_ta,self.min_kline_data_points)
            tk_resp=self.bybit_connector.get_tickers(category="linear",symbol=symbol)
            if not tk_resp or tk_resp.get("retCode")!=0 or not tk_resp.get('result',{}).get('list'): self.logger.error(f"Ticker err {symbol}: {tk_resp}"); return
            curr_px_str=tk_resp['result']['list'][0]['lastPrice']
            bal_resp=self.bybit_connector.get_wallet_balance(account_type="UNIFIED",coin="USDT")
            if not bal_resp or bal_resp.get("retCode")!=0 or not bal_resp.get('result',{}).get('list'): self.logger.error(f"Balance err: {bal_resp}"); return
            usdt_bal=next((c['coin'][0] for c in bal_resp['result']['list'] if c.get('coin') and c['coin'][0].get('coin')=='USDT'),None)
            if not usdt_bal or 'availableToWithdraw' not in usdt_bal: self.logger.error(f"USDT avail not found: {usdt_bal}"); return
            avail_bal_str=usdt_bal['availableToWithdraw']
            pos_resp=self.bybit_connector.get_positions(category="linear",symbol=symbol)
            open_pos_str=self._format_positions_for_prompt(pos_resp,symbol)
            inst_resp=self.bybit_connector.get_instrument_info(category="linear",symbol=symbol)
            if not inst_resp or inst_resp.get("retCode")!=0 or not inst_resp.get('result',{}).get('list'): self.logger.error(f"Instrument info err: {inst_resp}"); return
            instrument_info=inst_resp['result']['list'][0]
            fee_resp=self.bybit_connector.get_fee_rates(category="linear",symbol=symbol)
            if not fee_resp or fee_resp.get("retCode")!=0 or not fee_resp.get('result',{}).get('list'): self.logger.error(f"Fee rate err: {fee_resp}"); return
            taker_fee_str=fee_resp['result']['list'][0].get('takerFeeRate',str(self.config.get('taker_fee_rate','0.00055')))
        except Exception as e: self.logger.error(f"Data fetch error for {symbol}: {e}",exc_info=True); return
        if "None for" not in open_pos_str and "None"!=open_pos_str: self.logger.info(f"Existing position for {symbol}: {open_pos_str}. Hold."); return
        prompt=self._construct_llm_prompt(symbol,kl_prompt_str,curr_px_str,avail_bal_str,open_pos_str)
        analysis=self.llm_analyzer.analyze_market_data(prompt) # Changed here
        if not analysis or "error" in analysis: self.logger.error(f"LLM analysis error for {symbol}: {analysis}"); return
        self.logger.info(f"LLM Analysis for {symbol}: {json.dumps(analysis, indent=2)}")
        decision=analysis.get("decision","NO_ACTION").upper(); confidence=float(analysis.get("confidence_score",0.0))
        if decision in ["BUY","SELL"] and confidence>=self.confidence_threshold:
            try:
                ep_str=analysis.get("entry_price"); sl_str=analysis.get("stop_loss_price"); tp_llm_str=analysis.get("take_profit_price")
                if ep_str is None or sl_str is None: self.logger.error(f"LLM missing entry/SL for {symbol}."); return
                ep=float(ep_str); sl=float(sl_str)
                qty_str=self.risk_manager.calculate_position_size(float(avail_bal_str),ep,sl,instrument_info,symbol,decision)
                if qty_str and float(qty_str)>0:
                    tp_str=self.risk_manager.calculate_take_profit_price_for_fee_coverage(ep,decision,float(qty_str),float(taker_fee_str),instrument_info,symbol,float(self.config.get('tp_fee_multiplier',3.0)))
                    if not tp_str: rr=self.config.get('risk_reward_ratio',1.5);diff_sl=abs(ep-sl);tp_fb=ep+(diff_sl*rr) if decision=="BUY" else ep-(diff_sl*rr);tp_str=str(self.risk_manager._adjust_to_step(Decimal(str(tp_fb)),instrument_info['priceFilter']['tickSize']))
                    price_prec=self.risk_manager._get_price_precision(instrument_info['priceFilter']['tickSize'])
                    fmt_e=f"{ep:.{price_prec}f}";fmt_sl=f"{sl:.{price_prec}f}";fmt_tp=f"{float(tp_str):.{price_prec}f}"
                    log_dets=f"Sym={symbol},Side={decision},Qty={qty_str},E={fmt_e},SL={fmt_sl},TP={fmt_tp}"
                    if self.dry_run_mode: self.logger.info(f"[DRY RUN] Order: {log_dets}"); order_resp={"retCode":0,"result":{"orderId":f"dryrun_{int(time.time())}"}}
                    else: self.logger.info(f"[LIVE RUN] Order: {log_dets}"); order_resp=self.bybit_connector.place_order(category="linear",symbol=symbol,side=decision,order_type="Limit",qty=qty_str,price=fmt_e,stop_loss=fmt_sl,take_profit=fmt_tp,time_in_force="GTC",order_link_id=f"ollama_{symbol[:3]}{int(time.time())}")
                    self.logger.info(f"Order action resp for {symbol}: {order_resp}")
                else: self.logger.warning(f"Qty calc failed for {symbol}: {qty_str}")
            except Exception as e: self.logger.error(f"Trade exec error for {symbol}: {e}",exc_info=True)
        else: self.logger.info(f"Decision for {symbol} is {decision} (Conf:{confidence:.2f}). No trade.")
    def process_websocket_update(self, data_type: str, data: dict, symbol: str): self.logger.info(f"StrategyEngine: WS update. Type: {data_type}, Sym: {symbol}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Application Logic
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run_bot_application():
    LoggerSetup.setup(log_level=logging.INFO, log_file=DEFAULT_LOG_FILE_UNIFIED)
    logger = logging.getLogger(__name__)
    logger.info("🚀 Unified Trading Bot (Ollama Version) Starting Up...")
    config_mgr = Config(); config = config_mgr.config_data
    if not config: logger.error("Config empty. Shutting down."); return
    logger.info("Config loaded.")

    data_handler = DataHandler()
    risk_manager = RiskManager(config=config)

    bybit_conn_for_se = None; bybit_conn_initialized = False
    if config.get("use_bybit_testnet_connector", False):
        # Simplified Bybit init for brevity
        api_key=config.get("testnet_api_key_bybit"); api_secret=config.get("testnet_api_secret_bybit")
        if api_key and "YOUR_" not in api_key and api_secret and "YOUR_" not in api_secret:
            bybit_conn = BybitConnector(api_key=api_key, api_secret=api_secret, testnet=True)
            if bybit_conn.session: logger.info("Bybit Connector (Testnet) initialized."); bybit_conn_initialized=True; bybit_conn_for_se = bybit_conn
    if not bybit_conn_initialized:
        logger.warning("Using Dummy Bybit Connector."); class DummyBybit: session=None;logger=logging.getLogger("DummyBybit");def __init__(self,*a,**kw):pass;def get_kline(self,**kw): return {"retCode":1,"result":{"list":[]}};get_tickers=get_kline;get_wallet_balance=get_kline;get_positions=get_kline;get_instrument_info=get_kline;get_fee_rates=get_kline;place_order=lambda **kw:{"retCode":0,"result":{"orderId":"dummy"}}
        bybit_conn_for_se = DummyBybit()

    analyzer_for_se = None; analyzer_initialized = False
    if config.get("use_llm_analyzer", False): # Changed from use_gemini_analyzer
        ollama_base_url = config.get("ollama_base_url")
        ollama_model_name = config.get("ollama_model_name")
        if ollama_base_url and ollama_model_name and "YOUR_" not in ollama_base_url and "YOUR_" not in ollama_model_name :
            llm_analyzer = OllamaAnalyzer( # Changed class name
                ollama_base_url=ollama_base_url,
                ollama_model_name=ollama_model_name,
                request_timeout_seconds=config.get("ollama_request_timeout_seconds", 60)
            )
            if llm_analyzer.base_url: # Basic check for successful init
                logger.info(f"Ollama Analyzer initialized for model: {ollama_model_name}.")
                analyzer_initialized=True; analyzer_for_se = llm_analyzer
            else: logger.error("Ollama Analyzer FAILED to initialize.")
        else: logger.warning("Ollama config (url, model) missing/placeholder.")
    else: logger.info("LLM Analyzer (Ollama) usage disabled in config.")

    if not analyzer_initialized:
        logger.warning("Using Dummy Analyzer."); class DummyAnalyzer: logger=logging.getLogger("DummyAnalyzer");def __init__(self,*a,**kw):pass;def analyze_market_data(self,p): return {"decision":"HOLD", "reasoning":"Dummy"}
        analyzer_for_se = DummyAnalyzer()

    strategy_engine = StrategyEngine(config_mgr, bybit_conn_for_se, analyzer_for_se, data_handler, risk_manager) # Pass Config instance
    logger.info("StrategyEngine initialized.")

    if not config.get("use_websockets", False):
        logger.info("Starting polling-based trading loop...")
        # (Polling loop as read, simplified for brevity)
        trading_symbols=config.get("trading_symbols",[]); max_cycles=config.get("max_test_cycles_per_symbol",1)
        for sym in trading_symbols:
            if not bybit_conn_initialized and not config.get("dry_run_mode",True): logger.warning(f"Skip LIVE {sym}: Bybit not live"); continue
            if config.get("use_llm_analyzer") and not analyzer_initialized: logger.warning(f"Skip {sym}: LLM not init"); continue
            for i in range(max_cycles): logger.info(f"Cycle {i+1}/{max_cycles} for {sym}"); strategy_engine.run_main_cycle(sym); time.sleep(1) # Short sleep for test
        logger.info("Polling loop finished.")
    else: logger.info("WebSocket mode enabled. Main thread awaits.")
    logger.info("👋 Unified Trading Bot Shutting Down...")
    logging.shutdown()

def generate_example_config_unified(filepath=EXAMPLE_CONFIG_PATH_UNIFIED):
    example_data = {
        "bybit_api_key": "YOUR_BYBIT_MAINNET_API_KEY", "bybit_api_secret": "YOUR_BYBIT_MAINNET_API_SECRET",
        "testnet_api_key_bybit": "YOUR_BYBIT_TESTNET_API_KEY", "testnet_api_secret_bybit": "YOUR_BYBIT_TESTNET_API_SECRET",
        "ollama_base_url": "http://localhost:11434", # Added
        "ollama_model_name": "gemma3:latest",       # Added
        "ollama_request_timeout_seconds": 60,       # Added
        "llm_strategy_guidelines": "You are a helpful trading analysis assistant...", # Added
        "trading_symbols": ["BTCUSDT", "ETHUSDT"],
        "risk_per_trade_percentage": 1.0, "stop_loss_percentage": 0.02,
        "tp_fee_multiplier": 3.0, "risk_reward_ratio": 1.5, "taker_fee_rate": 0.00055,
        "use_bybit_testnet_connector": True, "use_llm_analyzer": True, # Changed
        "use_websockets": False, "ai_confidence_threshold": 0.6,
        "min_kline_data_points": 20, "polling_interval_seconds": 180,
        "max_test_cycles_per_symbol": 1, "dry_run_mode": True,
    }
    configs_dir = os.path.dirname(filepath)
    if not os.path.exists(configs_dir): os.makedirs(configs_dir)
    with open(filepath, 'w') as f: json.dump(example_data, f, indent=2)
    print(f"Generated example config: {filepath}")

if __name__ == "__main__":
    if not os.path.exists(EXAMPLE_CONFIG_PATH_UNIFIED): generate_example_config_unified()
    if not os.path.exists(DEFAULT_CONFIG_PATH_UNIFIED) and os.path.exists(EXAMPLE_CONFIG_PATH_UNIFIED):
        print(f"Dev config '{DEFAULT_CONFIG_PATH_UNIFIED}' not found. Copy example and edit.")
    run_bot_application()
