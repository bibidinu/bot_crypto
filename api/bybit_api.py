"""
Wrapper per l'API di Bybit V5 con supporto completo per futures
"""
import time
import hmac
import hashlib
import json
import urllib.parse
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
import pandas as pd
from datetime import datetime

from config.credentials import BYBIT_CREDENTIALS
from config.settings import BOT_MODE, BotMode, DEFAULT_LEVERAGE, DEFAULT_MARGIN_MODE
from utils.logger import get_logger

logger = get_logger(__name__)

class BybitAPI:
    """
    Classe per interagire con l'API di Bybit V5 con supporto completo per futures
    """
    
    def __init__(self, mode: BotMode = None):
        """
        Inizializza il client API Bybit
        
        Args:
            mode: Modalità del bot (demo o live)
        """
        if mode is None:
            mode = BOT_MODE
            
        self.mode = mode
        self.credentials = BYBIT_CREDENTIALS["demo" if mode == BotMode.DEMO else "live"]
        self.api_key = self.credentials["api_key"]
        self.api_secret = self.credentials["api_secret"]
        self.testnet = self.credentials["testnet"]
        
        # Base URL per le chiamate API
        self.base_url = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        
        # Limiti e configurazioni di rate limit
        self.rate_limit_reset = 0
        self.retries = 3
        self.retry_delay = 1  # secondi
        
        # Cache per dati di mercato frequentemente richiesti
        self.ticker_cache = {}
        self.ticker_cache_time = {}
        
        logger.info(f"BybitAPI inizializzata in modalità: {mode.value}")
    
    def _get_signature(self, timestamp: str, params: Dict[str, Any]) -> str:
        """
        Genera la firma HMAC per l'autenticazione
        
        Args:
            timestamp: Timestamp corrente in millisecondi
            params: Parametri della richiesta
            
        Returns:
            Firma HMAC
        """
        param_str = ""
        
        # Aggiungi timestamp ai parametri
        if isinstance(params, dict):
            params_with_ts = {**params, "timestamp": timestamp}
            param_str = urllib.parse.urlencode(params_with_ts)
        else:
            param_str = f"timestamp={timestamp}"
            
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(param_str, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None, 
                 signed: bool = False, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Esegue una richiesta all'API di Bybit
        
        Args:
            method: Metodo HTTP (GET, POST, etc.)
            endpoint: Endpoint API
            params: Parametri di query
            signed: Se la richiesta richiede una firma
            data: Dati da inviare nel body (per POST)
            
        Returns:
            Risposta JSON dell'API
        """
        url = f"{self.base_url}{endpoint}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if signed:
            timestamp = str(int(time.time() * 1000))
            signature = self._get_signature(timestamp, params if params else {})
            
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-SIGN": signature
            })
        
        for retry in range(self.retries):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, params=params)
                elif method == "POST":
                    response = requests.post(url, headers=headers, params=params, data=json.dumps(data) if data else None)
                elif method == "DELETE":
                    response = requests.delete(url, headers=headers, params=params, data=json.dumps(data) if data else None)
                else:
                    raise ValueError(f"Metodo HTTP non supportato: {method}")
                
                # Check for rate limit headers
                if 'x-bapi-limit-status' in response.headers:
                    limit_status = int(response.headers['x-bapi-limit-status'])
                    if limit_status > 90:  # if we're using more than 90% of our rate limit
                        logger.warning(f"Rate limit at {limit_status}%, backing off")
                        time.sleep(self.retry_delay * (retry+1))
                
                response.raise_for_status()
                result = response.json()
                
                if result.get("retCode") != 0:
                    error_msg = f"Errore API Bybit: {result.get('retMsg')} (Code: {result.get('retCode')})"
                    logger.error(error_msg)
                    
                    # Handle specific error codes
                    if result.get("retCode") == 10006:  # Too many requests
                        if retry < self.retries - 1:
                            wait_time = self.retry_delay * (2 ** retry)
                            logger.info(f"Rate limit exceeded, waiting {wait_time}s before retry")
                            time.sleep(wait_time)
                            continue
                    elif result.get("retCode") in [10018, 10019]:  # Position mode issues
                        # Try to fix position mode
                        self._handle_position_mode_error(data.get("symbol") if data else None)
                        if retry < self.retries - 1:
                            continue
                            
                    raise Exception(error_msg)
                    
                return result
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Errore di richiesta API: {str(e)}")
                if retry < self.retries - 1:
                    wait_time = self.retry_delay * (2 ** retry)
                    logger.info(f"Retrying in {wait_time}s... ({retry+1}/{self.retries})")
                    time.sleep(wait_time)
                else:
                    raise
        
        raise Exception("Max retries exceeded")
    
    def _handle_position_mode_error(self, symbol: Optional[str] = None) -> None:
        """
        Gestisce gli errori di position mode
        
        Args:
            symbol: Simbolo per cui impostare la position mode
        """
        try:
            if not symbol:
                return
                
            # Set position mode to both for the symbol
            self.set_position_mode(symbol, "MergedSingle")
            logger.info(f"Position mode set to MergedSingle for {symbol}")
        except Exception as e:
            logger.error(f"Errore nella gestione del position mode: {str(e)}")
    
    def _convert_interval(self, interval: str) -> str:
        """
        Converte l'intervallo nel formato accettato da Bybit
        
        Args:
            interval: Intervallo (es. '1m', '1h', '1d')
        
        Returns:
            Intervallo nel formato Bybit
        """
        # Mappatura degli intervalli al formato di Bybit
        interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30", 
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M"
        }
        
        if interval in interval_map:
            return interval_map[interval]
        else:
            logger.warning(f"Intervallo {interval} non supportato, usando 15m")
            return "15"  # Default a 15 minuti
    
    def _convert_symbol_to_contract(self, symbol: str) -> str:
        """
        Converte un simbolo standard in formato contratto Bybit
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            
        Returns:
            Simbolo in formato contratto Bybit
        """
        # Rimuove lo slash per ottenere il formato Bybit
        return symbol.replace("/", "")
    
    def _get_contract_type(self, symbol: str) -> str:
        """
        Determina il tipo di contratto per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Tipo di contratto ('linear', 'inverse' o 'spot')
        """
        # Per default, usiamo linear per futures USDT e inverse per quelli in USD
        if symbol.endswith("USDT") or symbol.endswith("/USDT"):
            return "linear"
        elif symbol.endswith("USD") or symbol.endswith("/USD"):
            return "inverse"
        else:
            return "spot"
    
    def _determine_category(self, symbol: str, category: Optional[str] = None) -> str:
        """
        Determina la categoria giusta per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            category: Categoria specificata (overrides auto-detection)
            
        Returns:
            Categoria ('linear', 'inverse', o 'spot')
        """
        if category:
            return category
            
        if symbol.endswith("USDT") or symbol.endswith("/USDT"):
            return "linear"
        elif symbol.endswith("USD") or symbol.endswith("/USD") and not symbol.endswith("USDT"):
            return "inverse"
        else:
            return "spot"
    
    # --- Metodi per il Market Data ---
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None,
                   category: Optional[str] = None) -> pd.DataFrame:
        """
        Ottiene i dati OHLCV (candele) per una coppia di trading
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            interval: Intervallo temporale (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Numero massimo di candele da recuperare
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            category: Categoria (spot, linear, inverse)
            
        Returns:
            DataFrame pandas con i dati OHLCV
        """
        endpoint = "/v5/market/kline"
        
        # Converti l'intervallo nel formato supportato da Bybit
        bybit_interval = self._convert_interval(interval)
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "interval": bybit_interval,
            "limit": min(limit, 1000)  # Massimo 1000 candele per richiesta
        }
        
        if start_time:
            params["start"] = start_time
        
        if end_time:
            params["end"] = end_time
            
        response = self._request("GET", endpoint, params=params)
        
        # Trasforma la risposta in un DataFrame pandas
        klines = []
        for item in response["result"]["list"]:
            klines.append({
                "timestamp": int(item[0]),
                "datetime": datetime.fromtimestamp(int(item[0]) / 1000),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5])
            })
            
        df = pd.DataFrame(klines)
        if df.empty:
            return df
            
        # Ordina per timestamp in ordine crescente
        df = df.sort_values("timestamp")
        
        return df
    
    def get_ticker(self, symbol: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene informazioni correnti sul prezzo di un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dati del ticker
        """
        endpoint = "/v5/market/tickers"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        # Check cache (valid for 1 second)
        cache_key = f"{cat}_{symbol}"
        now = time.time()
        if cache_key in self.ticker_cache and now - self.ticker_cache_time.get(cache_key, 0) < 1:
            return self.ticker_cache[cache_key]
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol)
        }
        
        response = self._request("GET", endpoint, params=params)
        
        if not response["result"]["list"]:
            raise ValueError(f"Nessun dato ticker trovato per il simbolo {symbol}")
            
        # Update cache
        self.ticker_cache[cache_key] = response["result"]["list"][0]
        self.ticker_cache_time[cache_key] = now
            
        return response["result"]["list"][0]
    
    def get_order_book(self, symbol: str, limit: int = 50, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene l'order book corrente per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            limit: Profondità dell'order book (max 200)
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dati dell'order book
        """
        endpoint = "/v5/market/orderbook"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "limit": min(limit, 200)
        }
        
        return self._request("GET", endpoint, params=params)["result"]
    
    def get_instrument_info(self, symbol: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene informazioni dettagliate su un simbolo/strumento
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Informazioni sullo strumento
        """
        endpoint = "/v5/market/instruments-info"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol)
        }
        
        response = self._request("GET", endpoint, params=params)
        
        if not response["result"]["list"]:
            raise ValueError(f"Nessuna informazione trovata per il simbolo {symbol}")
            
        return response["result"]["list"][0]
    
    def get_funding_rate(self, symbol: str, start_time: Optional[int] = None, 
                        end_time: Optional[int] = None, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Ottiene i tassi di funding per un simbolo perpetual future
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            limit: Numero massimo di risultati
            
        Returns:
            Lista di tassi di funding
        """
        endpoint = "/v5/market/funding/history"
        
        params = {
            "category": "linear",  # Solo futures hanno funding rate
            "symbol": self._convert_symbol_to_contract(symbol),
            "limit": min(limit, 200)
        }
        
        if start_time:
            params["startTime"] = start_time
            
        if end_time:
            params["endTime"] = end_time
        
        response = self._request("GET", endpoint, params=params)
        
        return response["result"]["list"]
    
    def get_open_interest(self, symbol: str, interval: str = "1h", 
                         limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene l'open interest per un simbolo future
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            interval: Intervallo temporale (5min, 15min, 30min, 1h, 4h, 1d)
            limit: Numero massimo di risultati
            
        Returns:
            Lista di dati open interest
        """
        endpoint = "/v5/market/open-interest"
        
        # Determina la categoria
        cat = self._determine_category(symbol)
        if cat == "spot":
            raise ValueError("Open interest è disponibile solo per futures")
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "intervalTime": interval,
            "limit": min(limit, 200)
        }
        
        response = self._request("GET", endpoint, params=params)
        
        return response["result"]["list"]
    
    # --- Metodi per il Trading ---
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, time_in_force: str = "GTC",
                   reduce_only: bool = False, close_on_trigger: bool = False,
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                   tp_trigger_by: str = "LastPrice", sl_trigger_by: str = "LastPrice",
                   position_idx: int = 0, category: Optional[str] = None,
                   stop_loss_params: Optional[Dict[str, Any]] = None,
                   take_profit_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Piazza un nuovo ordine
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            side: Direzione dell'ordine ("Buy" o "Sell")
            order_type: Tipo di ordine ("Market", "Limit")
            qty: Quantità da acquistare/vendere
            price: Prezzo per ordini limit (opzionale per ordini market)
            time_in_force: Validità dell'ordine ("GTC", "IOC", "FOK")
            reduce_only: Se l'ordine può solo ridurre la posizione
            close_on_trigger: Se chiudere la posizione quando viene attivato
            stop_loss: Prezzo di stop loss (opzionale)
            take_profit: Prezzo di take profit (opzionale)
            tp_trigger_by: Tipo di trigger per il take profit
            sl_trigger_by: Tipo di trigger per lo stop loss
            position_idx: Indice della posizione (0: unidirezionale, 1: Buy, 2: Sell)
            category: Categoria (spot, linear, inverse)
            stop_loss_params: Parametri avanzati per stop loss (dict)
            take_profit_params: Parametri avanzati per take profit (dict)
            
        Returns:
            Dettagli dell'ordine piazzato
        """
        endpoint = "/v5/order/create"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {}
        
        data = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
            "positionIdx": position_idx
        }
        
        # Per futures, aggiungi reduce_only e close_on_trigger
        if cat in ["linear", "inverse"]:
            data["reduceOnly"] = reduce_only
            data["closeOnTrigger"] = close_on_trigger
        
        if price and order_type != "Market":
            data["price"] = str(price)
            
        # Simple stop loss / take profit
        if stop_loss:
            data["stopLoss"] = str(stop_loss)
            data["slTriggerBy"] = sl_trigger_by
            
        if take_profit:
            data["takeProfit"] = str(take_profit)
            data["tpTriggerBy"] = tp_trigger_by
            
        # Advanced stop loss / take profit parameters
        if stop_loss_params:
            data.update(stop_loss_params)
            
        if take_profit_params:
            data.update(take_profit_params)
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def place_conditional_order(self, symbol: str, side: str, order_type: str, qty: float, 
                               trigger_price: float, trigger_by: str = "LastPrice",
                               price: Optional[float] = None, time_in_force: str = "GTC",
                               reduce_only: bool = False, close_on_trigger: bool = False,
                               position_idx: int = 0, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Piazza un ordine condizionale (stop o take profit)
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            side: Direzione dell'ordine ("Buy" o "Sell")
            order_type: Tipo di ordine ("Market", "Limit")
            qty: Quantità da acquistare/vendere
            trigger_price: Prezzo di attivazione
            trigger_by: Tipo di trigger ("LastPrice", "IndexPrice", "MarkPrice")
            price: Prezzo per ordini limit (opzionale per ordini market)
            time_in_force: Validità dell'ordine ("GTC", "IOC", "FOK")
            reduce_only: Se l'ordine può solo ridurre la posizione
            close_on_trigger: Se chiudere la posizione quando viene attivato
            position_idx: Indice della posizione (0: unidirezionale, 1: Buy, 2: Sell)
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dettagli dell'ordine piazzato
        """
        endpoint = "/v5/order/create-order"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {}
        
        data = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "triggerPrice": str(trigger_price),
            "triggerBy": trigger_by,
            "timeInForce": time_in_force,
            "positionIdx": position_idx,
            "orderFilter": "tpslOrder"  # o "StopOrder" per ordini stop
        }
        
        # Per futures, aggiungi reduce_only e close_on_trigger
        if cat in ["linear", "inverse"]:
            data["reduceOnly"] = reduce_only
            data["closeOnTrigger"] = close_on_trigger
        
        if price and order_type != "Market":
            data["price"] = str(price)
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def cancel_order(self, symbol: str, order_id: Optional[str] = None, 
                    category: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancella un ordine esistente
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            order_id: ID dell'ordine da cancellare
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dettagli della cancellazione
        """
        endpoint = "/v5/order/cancel"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {}
        
        data = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol)
        }
        
        if order_id:
            data["orderId"] = order_id
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def cancel_all_orders(self, symbol: Optional[str] = None, 
                         category: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancella tutti gli ordini per un simbolo o tutti i simboli
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dettagli della cancellazione
        """
        endpoint = "/v5/order/cancel-all"
        
        # Imposta la categoria
        if category is None:
            if symbol:
                category = self._determine_category(symbol)
            else:
                category = "linear"  # Default a linear se non specificato
        
        params = {}
        
        data = {
            "category": category
        }
        
        if symbol:
            data["symbol"] = self._convert_symbol_to_contract(symbol)
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def amend_order(self, symbol: str, order_id: str, qty: Optional[float] = None,
                   price: Optional[float] = None, take_profit: Optional[float] = None,
                   stop_loss: Optional[float] = None, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Modifica un ordine esistente
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            order_id: ID dell'ordine da modificare
            qty: Nuova quantità (opzionale)
            price: Nuovo prezzo (opzionale)
            take_profit: Nuovo take profit (opzionale)
            stop_loss: Nuovo stop loss (opzionale)
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dettagli dell'ordine modificato
        """
        endpoint = "/v5/order/amend"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {}
        
        data = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol),
            "orderId": order_id
        }
        
        if qty is not None:
            data["qty"] = str(qty)
            
        if price is not None:
            data["price"] = str(price)
            
        if take_profit is not None:
            data["takeProfit"] = str(take_profit)
            
        if stop_loss is not None:
            data["stopLoss"] = str(stop_loss)
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def get_open_orders(self, symbol: Optional[str] = None, 
                        category: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene gli ordini aperti
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            category: Categoria (spot, linear, inverse)
            limit: Numero massimo di ordini da recuperare
            
        Returns:
            Lista di ordini aperti
        """
        endpoint = "/v5/order/realtime"
        
        # Imposta la categoria
        if category is None:
            if symbol:
                category = self._determine_category(symbol)
            else:
                category = "linear"  # Default a linear se non specificato
        
        params = {
            "category": category,
            "limit": min(limit, 50)
        }
        
        if symbol:
            params["symbol"] = self._convert_symbol_to_contract(symbol)
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    def get_order_history(self, symbol: Optional[str] = None, 
                         category: Optional[str] = None,
                         limit: int = 50, 
                         order_status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ottiene la cronologia degli ordini
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            category: Categoria (spot, linear, inverse)
            limit: Numero massimo di ordini da recuperare
            order_status: Stato degli ordini da recuperare (opzionale)
            
        Returns:
            Lista di ordini storici
        """
        endpoint = "/v5/order/history"
        
        # Imposta la categoria
        if category is None:
            if symbol:
                category = self._determine_category(symbol)
            else:
                category = "linear"  # Default a linear se non specificato
        
        params = {
            "category": category,
            "limit": min(limit, 100)
        }
        
        if symbol:
            params["symbol"] = self._convert_symbol_to_contract(symbol)
            
        if order_status:
            params["orderStatus"] = order_status
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    # --- Metodi per Account ---
    
    def get_wallet_balance(self, coin: Optional[str] = None, 
                          account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Ottiene il bilancio del wallet
        
        Args:
            coin: Simbolo della valuta (opzionale)
            account_type: Tipo di account ("UNIFIED", "CONTRACT", "SPOT")
            
        Returns:
            Bilancio del wallet
        """
        endpoint = "/v5/account/wallet-balance"
        
        params = {
            "accountType": account_type
        }
        
        if coin:
            params["coin"] = coin
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]
    
    def get_positions(self, symbol: Optional[str] = None, 
                     category: Optional[str] = "linear",
                     settle_coin: Optional[str] = None,
                     limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene le posizioni aperte
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            category: Categoria (linear, inverse)
            settle_coin: Valuta di settlement (opzionale)
            limit: Numero massimo di posizioni da recuperare
            
        Returns:
            Lista di posizioni aperte
        """
        endpoint = "/v5/position/list"
        
        # Verifica che la categoria sia valida per le posizioni
        if category not in ["linear", "inverse"]:
            category = "linear"  # Default a linear se non valido
        
        params = {
            "category": category,
            "limit": min(limit, 200)
        }
        
        if symbol:
            params["symbol"] = self._convert_symbol_to_contract(symbol)
            
        if settle_coin:
            params["settleCoin"] = settle_coin
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    def set_leverage(self, symbol: str, leverage: float, 
                    buy_leverage: Optional[float] = None,
                    sell_leverage: Optional[float] = None,
                    category: Optional[str] = None) -> Dict[str, Any]:
        """
        Imposta il leverage per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            leverage: Valore del leverage
            buy_leverage: Leverage specifico per posizioni long
            sell_leverage: Leverage specifico per posizioni short
            category: Categoria (linear, inverse)
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/set-leverage"
        
        # Determina la categoria
        if category is None:
            category = self._determine_category(symbol)
            if category == "spot":
                raise ValueError("Il leverage è disponibile solo per futures")
        
        params = {}
        
        # Se buy_leverage o sell_leverage non sono specificati, usa leverage per entrambi
        buy_lev = buy_leverage if buy_leverage is not None else leverage
        sell_lev = sell_leverage if sell_leverage is not None else leverage
        
        data = {
            "category": category,
            "symbol": self._convert_symbol_to_contract(symbol),
            "buyLeverage": str(buy_lev),
            "sellLeverage": str(sell_lev)
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def set_margin_mode(self, symbol: str, margin_mode: str = "ISOLATED", 
                       category: Optional[str] = None) -> Dict[str, Any]:
        """
        Imposta la modalità di margine per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            margin_mode: Modalità di margine ("ISOLATED", "CROSS")
            category: Categoria (linear, inverse)
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/switch-isolated"
        
        # Determina la categoria
        if category is None:
            category = self._determine_category(symbol)
            if category == "spot":
                raise ValueError("Il margin mode è disponibile solo per futures")
        
        params = {}
        
        # Verifica che il margin mode sia valido
        margin_mode = margin_mode.upper()
        if margin_mode not in ["ISOLATED", "CROSS"]:
            raise ValueError("Margin mode non valido. Usa 'ISOLATED' o 'CROSS'")
        
        tradeMode = 1 if margin_mode == "ISOLATED" else 0
        
        data = {
            "category": category,
            "symbol": self._convert_symbol_to_contract(symbol),
            "tradeMode": tradeMode
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def set_position_mode(self, symbol: str, mode: str = "MergedSingle", 
                         category: Optional[str] = None) -> Dict[str, Any]:
        """
        Imposta la modalità di posizione per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            mode: Modalità di posizione ("MergedSingle", "BothSide")
            category: Categoria (linear, inverse)
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/switch-mode"
        
        # Determina la categoria
        if category is None:
            category = self._determine_category(symbol)
            if category == "spot":
                raise ValueError("Il position mode è disponibile solo per futures")
        
        params = {}
        
        # Verifica che il position mode sia valido
        if mode not in ["MergedSingle", "BothSide"]:
            raise ValueError("Position mode non valido. Usa 'MergedSingle' o 'BothSide'")
        
        coin = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '').replace('USD', '')
        
        data = {
            "category": category,
            "symbol": None,  # Deve essere None per impostare per coin
            "coin": coin,
            "mode": mode
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def set_tpsl_mode(self, symbol: str, mode: str = "Full",
                     category: Optional[str] = None) -> Dict[str, Any]:
        """
        Imposta la modalità di Take Profit / Stop Loss per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            mode: Modalità TP/SL ("Full", "Partial")
            category: Categoria (linear, inverse)
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/set-tpsl-mode"
        
        # Determina la categoria
        if category is None:
            category = self._determine_category(symbol)
            if category == "spot":
                raise ValueError("Il TPSL mode è disponibile solo per futures")
        
        params = {}
        
        # Verifica che il tpsl mode sia valido
        if mode not in ["Full", "Partial"]:
            raise ValueError("TPSL mode non valido. Usa 'Full' o 'Partial'")
        
        data = {
            "category": category,
            "symbol": self._convert_symbol_to_contract(symbol),
            "tpSlMode": mode
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def get_trading_fee(self, symbol: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene le commissioni di trading per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            category: Categoria (spot, linear, inverse)
            
        Returns:
            Dettagli sulle commissioni
        """
        endpoint = "/v5/account/fee-rate"
        
        # Determina la categoria
        cat = self._determine_category(symbol, category)
        
        params = {
            "category": cat,
            "symbol": self._convert_symbol_to_contract(symbol)
        }
        
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]
    
    def set_risk_limit(self, symbol: str, risk_id: int, 
                      category: Optional[str] = None,
                      position_idx: int = 0) -> Dict[str, Any]:
        """
        Imposta il limite di rischio per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTC/USDT")
            risk_id: ID del livello di rischio
            category: Categoria (linear, inverse)
            position_idx: Indice della posizione (0: unidirezionale, 1: Buy, 2: Sell)
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/set-risk-limit"
        
        # Determina la categoria
        if category is None:
            category = self._determine_category(symbol)
            if category == "spot":
                raise ValueError("Il risk limit è disponibile solo per futures")
        
        params = {}
        
        data = {
            "category": category,
            "symbol": self._convert_symbol_to_contract(symbol),
            "riskId": risk_id,
            "positionIdx": position_idx
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def get_user_trades(self, symbol: Optional[str] = None,
                       category: Optional[str] = None, 
                       limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene la cronologia delle transazioni dell'utente
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            category: Categoria (spot, linear, inverse)
            limit: Numero massimo di transazioni da recuperare
            
        Returns:
            Lista di transazioni
        """
        endpoint = "/v5/execution/list"
        
        # Imposta la categoria
        if category is None:
            if symbol:
                category = self._determine_category(symbol)
            else:
                category = "linear"  # Default a linear se non specificato
        
        params = {
            "category": category,
            "limit": min(limit, 100)
        }
        
        if symbol:
            params["symbol"] = self._convert_symbol_to_contract(symbol)
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]