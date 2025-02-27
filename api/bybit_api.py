"""
Wrapper per l'API di Bybit V5
"""
import time
import hmac
import hashlib
import json
import urllib.parse
from typing import Dict, List, Optional, Union, Any
import requests
import pandas as pd
from datetime import datetime

from config.credentials import BYBIT_CREDENTIALS
from config.settings import BOT_MODE, BotMode
from utils.logger import get_logger

logger = get_logger(__name__)

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
        
class BybitAPI:
    """
    Classe per interagire con l'API di Bybit V5
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
        self.base_url = "https://api-demo.bybit.com" if self.testnet else "https://api.bybit.com"
        
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
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params, data=json.dumps(data) if data else None)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params, data=json.dumps(data) if data else None)
            else:
                raise ValueError(f"Metodo HTTP non supportato: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("retCode") != 0:
                logger.error(f"Errore API Bybit: {result.get('retMsg')}")
                raise Exception(f"Errore API Bybit: {result.get('retMsg')}")
                
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore di richiesta API: {str(e)}")
            raise
    
    # --- Metodi per il Market Data ---
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Ottiene i dati OHLCV (candele) per una coppia di trading
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            interval: Intervallo temporale (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Numero massimo di candele da recuperare
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            
        Returns:
            DataFrame pandas con i dati OHLCV
        """
        endpoint = "/v5/market/kline"
        
        params = {
            "category": "spot",
            "symbol": symbol.replace("/", ""),
            "interval": interval,
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
        # Ordina per timestamp in ordine crescente
        df = df.sort_values("timestamp")
        
        return df
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene informazioni correnti sul prezzo di un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            
        Returns:
            Dati del ticker
        """
        endpoint = "/v5/market/tickers"
        
        params = {
            "category": "spot",
            "symbol": symbol.replace("/", "")
        }
        
        response = self._request("GET", endpoint, params=params)
        
        if not response["result"]["list"]:
            raise ValueError(f"Nessun dato ticker trovato per il simbolo {symbol}")
            
        return response["result"]["list"][0]
    
    def get_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Ottiene l'order book corrente per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            limit: Profondità dell'order book (max 200)
            
        Returns:
            Dati dell'order book
        """
        endpoint = "/v5/market/orderbook"
        
        params = {
            "category": "spot",
            "symbol": symbol.replace("/", ""),
            "limit": min(limit, 200)
        }
        
        return self._request("GET", endpoint, params=params)["result"]
    
    # --- Metodi per il Trading ---
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, time_in_force: str = "GTC",
                   reduce_only: bool = False, close_on_trigger: bool = False,
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Piazza un nuovo ordine
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            side: Direzione dell'ordine ("Buy" o "Sell")
            order_type: Tipo di ordine ("Market", "Limit")
            qty: Quantità da acquistare/vendere
            price: Prezzo per ordini limit (opzionale per ordini market)
            time_in_force: Validità dell'ordine ("GTC", "IOC", "FOK")
            reduce_only: Se l'ordine può solo ridurre la posizione
            close_on_trigger: Se chiudere la posizione quando viene attivato
            stop_loss: Prezzo di stop loss (opzionale)
            take_profit: Prezzo di take profit (opzionale)
            
        Returns:
            Dettagli dell'ordine piazzato
        """
        endpoint = "/v5/order/create"
        
        params = {}
        
        data = {
            "category": "spot",
            "symbol": symbol.replace("/", ""),
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
            "closeOnTrigger": close_on_trigger
        }
        
        if price and order_type != "Market":
            data["price"] = str(price)
            
        if stop_loss:
            data["stopLoss"] = str(stop_loss)
            
        if take_profit:
            data["takeProfit"] = str(take_profit)
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def cancel_order(self, symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancella un ordine esistente
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            order_id: ID dell'ordine da cancellare
            
        Returns:
            Dettagli della cancellazione
        """
        endpoint = "/v5/order/cancel"
        
        params = {}
        
        data = {
            "category": "spot",
            "symbol": symbol.replace("/", "")
        }
        
        if order_id:
            data["orderId"] = order_id
            
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ottiene gli ordini aperti
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            
        Returns:
            Lista di ordini aperti
        """
        endpoint = "/v5/order/realtime"
        
        params = {
            "category": "spot"
        }
        
        if symbol:
            params["symbol"] = symbol.replace("/", "")
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene la cronologia degli ordini
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            limit: Numero massimo di ordini da recuperare
            
        Returns:
            Lista di ordini storici
        """
        endpoint = "/v5/order/history"
        
        params = {
            "category": "spot",
            "limit": min(limit, 100)
        }
        
        if symbol:
            params["symbol"] = symbol.replace("/", "")
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    # --- Metodi per Account ---
    
    def get_wallet_balance(self, coin: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene il bilancio del wallet
        
        Args:
            coin: Simbolo della valuta (opzionale)
            
        Returns:
            Bilancio del wallet
        """
        endpoint = "/v5/account/wallet-balance"
        
        params = {
            "accountType": "SPOT"
        }
        
        if coin:
            params["coin"] = coin
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ottiene le posizioni aperte
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            
        Returns:
            Lista di posizioni aperte
        """
        endpoint = "/v5/position/list"
        
        params = {
            "category": "linear"  # Per futures
        }
        
        if symbol:
            params["symbol"] = symbol.replace("/", "")
            
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]["list"]
    
    def set_leverage(self, symbol: str, leverage: float, leverage_mode: str = "isolated") -> Dict[str, Any]:
        """
        Imposta il leverage per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            leverage: Valore del leverage
            leverage_mode: Modalità di leverage ("isolated" o "cross")
            
        Returns:
            Risultato dell'operazione
        """
        endpoint = "/v5/position/set-leverage"
        
        params = {}
        
        data = {
            "category": "linear",
            "symbol": symbol.replace("/", ""),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        
        return self._request("POST", endpoint, params=params, signed=True, data=data)["result"]
    
    def get_trading_fee(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene le commissioni di trading per un simbolo
        
        Args:
            symbol: Simbolo della coppia (es. "BTCUSDT")
            
        Returns:
            Dettagli sulle commissioni
        """
        endpoint = "/v5/account/fee-rate"
        
        params = {
            "category": "spot",
            "symbol": symbol.replace("/", "")
        }
        
        response = self._request("GET", endpoint, params=params, signed=True)
        
        return response["result"]
