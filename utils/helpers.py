"""
Modulo con funzioni helper utili per il trading bot
"""
import time
import datetime
import os
import json
import re
import hashlib
import random
import string
from typing import Dict, List, Optional, Any, Union, Tuple
from decimal import Decimal, ROUND_DOWN

from utils.logger import get_logger

logger = get_logger(__name__)

def truncate_float(value: float, decimals: int) -> float:
    """
    Tronca (non arrotonda) un float a un numero specifico di decimali
    
    Args:
        value: Valore da troncare
        decimals: Numero di decimali
        
    Returns:
        Valore troncato
    """
    factor = 10 ** decimals
    return float(int(value * factor) / factor)

def round_to_tick_size(value: float, tick_size: float) -> float:
    """
    Arrotonda un valore al tick size specificato
    
    Args:
        value: Valore da arrotondare
        tick_size: Dimensione del tick
        
    Returns:
        Valore arrotondato
    """
    return float(Decimal(str(value)).quantize(Decimal(str(tick_size)), rounding=ROUND_DOWN))

def format_number(value: float, decimals: int = 8) -> str:
    """
    Formatta un numero con un numero specifico di decimali,
    rimuovendo gli zeri finali dopo il punto decimale
    
    Args:
        value: Valore da formattare
        decimals: Numero massimo di decimali
        
    Returns:
        Stringa formattata
    """
    formatted = f"{value:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted

def timestamp_to_datetime(timestamp: int) -> datetime.datetime:
    """
    Converte un timestamp in un oggetto datetime
    
    Args:
        timestamp: Timestamp in millisecondi o secondi
        
    Returns:
        Oggetto datetime
    """
    # Verifica se il timestamp è in millisecondi
    if timestamp > 1000000000000:
        timestamp = timestamp / 1000
    
    return datetime.datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime.datetime) -> int:
    """
    Converte un oggetto datetime in un timestamp
    
    Args:
        dt: Oggetto datetime
        
    Returns:
        Timestamp in millisecondi
    """
    return int(dt.timestamp() * 1000)

def timeframe_to_seconds(timeframe: str) -> int:
    """
    Converte un timeframe in secondi
    
    Args:
        timeframe: Timeframe (es. "1m", "1h", "1d")
        
    Returns:
        Secondi
    """
    timeframe = timeframe.lower()
    
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60 * 60
    elif timeframe.endswith("d"):
        return int(timeframe[:-1]) * 60 * 60 * 24
    elif timeframe.endswith("w"):
        return int(timeframe[:-1]) * 60 * 60 * 24 * 7
    else:
        return int(timeframe)

def normalize_symbol(symbol: str) -> str:
    """
    Normalizza un simbolo in formato standard
    
    Args:
        symbol: Simbolo da normalizzare
        
    Returns:
        Simbolo normalizzato
    """
    symbol = symbol.upper()
    
    # Se contiene già uno slash, restituiscilo
    if "/" in symbol:
        return symbol
    
    # Altrimenti, cerca di aggiungere lo slash
    for quote in ["USDT", "USD", "BTC", "ETH", "BUSD", "USDC"]:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}"
    
    # Se non è stato possibile determinare la coppia, restituisci il simbolo originale
    return symbol

def format_price_for_precision(price: float, precision: int) -> str:
    """
    Formatta un prezzo con la precisione specificata
    
    Args:
        price: Prezzo da formattare
        precision: Precisione (numero di decimali)
        
    Returns:
        Prezzo formattato
    """
    format_str = f"{{:.{precision}f}}"
    return format_str.format(price)

def calculate_position_size(capital: float, risk_percentage: float, 
                           entry_price: float, stop_loss_price: float) -> float:
    """
    Calcola la dimensione della posizione in base al rischio
    
    Args:
        capital: Capitale disponibile
        risk_percentage: Percentuale di rischio (0-100)
        entry_price: Prezzo di entrata
        stop_loss_price: Prezzo di stop loss
        
    Returns:
        Dimensione della posizione
    """
    # Calcola il rischio in valuta
    risk_amount = capital * (risk_percentage / 100)
    
    # Calcola la differenza percentuale tra entry e stop loss
    price_diff = abs(entry_price - stop_loss_price)
    
    # Se il prezzo di entrata è zero o uguale allo stop loss, restituisci zero
    if entry_price == 0 or price_diff == 0:
        return 0
    
    # Calcola la dimensione della posizione
    position_size = risk_amount / price_diff
    
    return position_size

def get_timeframe_for_period(period_days: int) -> str:
    """
    Determina il timeframe più adatto per un periodo specifico
    
    Args:
        period_days: Periodo in giorni
        
    Returns:
        Timeframe adatto
    """
    if period_days <= 1:
        return "15m"
    elif period_days <= 3:
        return "1h"
    elif period_days <= 7:
        return "4h"
    elif period_days <= 30:
        return "1d"
    else:
        return "1w"

def generate_random_id(length: int = 8) -> str:
    """
    Genera un ID casuale alfanumerico
    
    Args:
        length: Lunghezza dell'ID
        
    Returns:
        ID casuale
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Esegue una divisione sicura, restituendo un valore di default in caso di divisione per zero
    
    Args:
        numerator: Numeratore
        denominator: Denominatore
        default: Valore di default
        
    Returns:
        Risultato della divisione o valore di default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def format_large_number(num: float) -> str:
    """
    Formatta un numero grande in una forma più leggibile
    
    Args:
        num: Numero da formattare
        
    Returns:
        Numero formattato
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"

def load_json_file(filepath: str) -> Optional[Dict]:
    """
    Carica un file JSON
    
    Args:
        filepath: Percorso del file
        
    Returns:
        Dati JSON o None in caso di errore
    """
    try:
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Errore nel caricamento del file JSON {filepath}: {str(e)}")
        return None

def save_json_file(filepath: str, data: Any, pretty: bool = True) -> bool:
    """
    Salva dati in un file JSON
    
    Args:
        filepath: Percorso del file
        data: Dati da salvare
        pretty: Se formattare il JSON per la leggibilità
        
    Returns:
        True se il salvataggio è riuscito
    """
    try:
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
                
        return True
    except Exception as e:
        logger.error(f"Errore nel salvataggio del file JSON {filepath}: {str(e)}")
        return False

def retry_function(func, max_retries: int = 3, retry_delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions_to_retry: tuple = (Exception,)):
    """
    Decorator per riprovare una funzione in caso di eccezione
    
    Args:
        func: Funzione da eseguire
        max_retries: Numero massimo di tentativi
        retry_delay: Ritardo iniziale tra i tentativi
        backoff_factor: Fattore di crescita del ritardo
        exceptions_to_retry: Tuple di eccezioni da riprovare
        
    Returns:
        Risultato della funzione o solleva l'ultima eccezione dopo tutti i tentativi
    """
    def wrapper(*args, **kwargs):
        last_exception = None
        delay = retry_delay
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions_to_retry as e:
                last_exception = e
                logger.warning(f"Tentativo {attempt+1}/{max_retries} fallito: {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Nuovo tentativo tra {delay:.2f} secondi")
                    time.sleep(delay)
                    delay *= backoff_factor
        
        # Se arriviamo qui, tutti i tentativi sono falliti
        logger.error(f"Tutti i tentativi falliti: {str(last_exception)}")
        raise last_exception
    
    return wrapper

def parse_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Analizza un timeframe e restituisce il valore e l'unità
    
    Args:
        timeframe: Timeframe da analizzare (es. "1m", "1h")
        
    Returns:
        Tupla (valore, unità)
    """
    match = re.match(r'^(\d+)([mhdw])$', timeframe.lower())
    
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        return value, unit
    
    raise ValueError(f"Formato timeframe non valido: {timeframe}")

def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
    """
    Calcola l'ATR (Average True Range)
    
    Args:
        high: Lista dei prezzi massimi
        low: Lista dei prezzi minimi
        close: Lista dei prezzi di chiusura
        period: Periodo per il calcolo
        
    Returns:
        Lista di valori ATR
    """
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Le liste high, low e close devono avere la stessa lunghezza")
    
    # Calcola i True Range
    tr = []
    
    for i in range(len(high)):
        if i == 0:
            # Per il primo elemento, TR = High - Low
            tr.append(high[i] - low[i])
        else:
            # TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            ))
    
    # Calcola l'ATR
    atr = []
    
    for i in range(len(tr)):
        if i < period:
            # Per i primi 'period' elementi, usa la media semplice
            atr.append(sum(tr[:i+1]) / (i+1))
        else:
            # Per gli elementi successivi, usa la media mobile esponenziale
            atr.append((atr[i-1] * (period-1) + tr[i]) / period)
    
    return atr

def format_time_interval(seconds: int) -> str:
    """
    Formatta un intervallo di tempo in una stringa leggibile
    
    Args:
        seconds: Secondi totali
        
    Returns:
        Stringa formattata
    """
    if seconds < 60:
        return f"{seconds} secondi"
    
    minutes = seconds // 60
    seconds %= 60
    
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    
    hours = minutes // 60
    minutes %= 60
    
    if hours < 24:
        return f"{hours}h {minutes}m"
    
    days = hours // 24
    hours %= 24
    
    return f"{days}d {hours}h {minutes}m"
