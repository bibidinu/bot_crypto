"""
Modulo per la raccolta e l'analisi dei dati di mercato
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import ta
from functools import lru_cache

from api.exchange_interface import ExchangeInterface
from utils.logger import get_logger
from config.settings import (
    DEFAULT_TRADING_PAIRS, DEFAULT_TIMEFRAME, 
    AVAILABLE_TIMEFRAMES, DATA_LOOKBACK_PERIODS
)

logger = get_logger(__name__)

class MarketData:
    """
    Classe per la raccolta e l'analisi dei dati di mercato
    """
    
    def __init__(self, exchange: ExchangeInterface):
        """
        Inizializza il raccoglitore di dati di mercato
        
        Args:
            exchange: Istanza dell'exchange
        """
        self.exchange = exchange
        self.logger = get_logger(__name__)
        
        # Cache per i dati di mercato (symbol_timeframe -> DataFrame)
        self.data_cache = {}
        self.last_update = {}
        
        # Coda per richieste di aggiornamento asincrone
        self.update_queue = queue.Queue()
        
        # Thread per l'aggiornamento asincrono
        self.update_thread = None
        self.running = False
        
        # Liste di simboli e timeframe da monitorare
        self.symbols = DEFAULT_TRADING_PAIRS
        self.timeframes = AVAILABLE_TIMEFRAMES
        
        # Cache per dati elaborati
        self.indicators_cache = {}
        
        # Tracciamento degli errori per evitare continui tentativi falliti
        self.error_count = {}
        self.error_cooldown = {}
        
        self.logger.info("MarketData inizializzato")
    
    def start_background_updates(self) -> None:
        """
        Avvia il thread di aggiornamento in background
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            self.logger.warning("Thread di aggiornamento già in esecuzione")
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._background_update_worker)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Thread di aggiornamento avviato")
    
    def stop_background_updates(self) -> None:
        """
        Ferma il thread di aggiornamento in background
        """
        self.running = False
        
        if self.update_thread is not None:
            try:
                self.update_thread.join(timeout=5.0)
            except:
                pass
                
        self.logger.info("Thread di aggiornamento fermato")
    
    def _background_update_worker(self) -> None:
        """Thread worker per aggiornamenti in background"""
        self.logger.info("Worker di aggiornamento avviato")
        
        while self.running:
            try:
                # Elabora le richieste in coda
                try:
                    request = self.update_queue.get(timeout=1.0)
                    
                    symbol = request.get("symbol")
                    timeframe = request.get("timeframe")
                    
                    # Verifica se questa combinazione è in cooldown per errori
                    cache_key = f"{symbol}_{timeframe}"
                    now = datetime.now()
                    
                    if cache_key in self.error_cooldown and now < self.error_cooldown[cache_key]:
                        self.logger.debug(f"Saltando aggiornamento di {symbol} {timeframe} (in cooldown)")
                        self.update_queue.task_done()
                        continue
                    
                    self.logger.debug(f"Aggiornamento dati per {symbol} {timeframe}")
                    
                    # Aggiorna i dati
                    try:
                        self.get_market_data(symbol, timeframe, force_update=True)
                        # Resetta il contatore degli errori se l'aggiornamento ha successo
                        self.error_count[cache_key] = 0
                    except Exception as e:
                        # Incrementa il contatore degli errori
                        if cache_key not in self.error_count:
                            self.error_count[cache_key] = 0
                        self.error_count[cache_key] += 1
                        
                        # Se ci sono troppi errori consecutivi, metti in cooldown
                        if self.error_count[cache_key] >= 3:
                            # Metti in cooldown per 30 minuti
                            self.error_cooldown[cache_key] = now + timedelta(minutes=30)
                            self.logger.warning(f"{symbol} {timeframe} in cooldown per 30 minuti dopo {self.error_count[cache_key]} errori")
                    
                    self.update_queue.task_done()
                    
                except queue.Empty:
                    # Nessuna richiesta in coda
                    pass
                
                # Aggiorna periodicamente i dati
                now = datetime.now()
                
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        cache_key = f"{symbol}_{timeframe}"
                        
                        # Salta se in cooldown
                        if cache_key in self.error_cooldown and now < self.error_cooldown[cache_key]:
                            continue
                        
                        # Determina l'intervallo di aggiornamento in base al timeframe
                        update_interval = self._get_update_interval(timeframe)
                        
                        # Verifica se è necessario un aggiornamento
                        if (cache_key not in self.last_update or
                            now - self.last_update[cache_key] > update_interval):
                            
                            # Aggiungi una richiesta di aggiornamento
                            self.update_queue.put({
                                "symbol": symbol,
                                "timeframe": timeframe
                            })
                
                # Sleep per non sovraccaricare
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel thread di aggiornamento: {str(e)}")
                time.sleep(5.0)  # Sleep più lungo in caso di errore
    
    def _get_update_interval(self, timeframe: str) -> timedelta:
        """
        Determina l'intervallo di aggiornamento in base al timeframe
        
        Args:
            timeframe: Intervallo temporale
            
        Returns:
            Intervallo di aggiornamento come timedelta
        """
        # Converte il timeframe in minuti
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            minutes = int(timeframe[:-1]) * 1440
        else:
            minutes = 15  # Default
        
        # Aggiorna più frequentemente per timeframe più brevi
        # Meno frequentemente per timeframe più lunghi
        if minutes < 60:  # < 1h
            return timedelta(minutes=minutes)
        elif minutes < 1440:  # < 1d
            return timedelta(minutes=minutes // 2)
        else:  # >= 1d
            return timedelta(hours=6)
    
    def get_market_data(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, 
                       limit: int = DATA_LOOKBACK_PERIODS, 
                       force_update: bool = False) -> pd.DataFrame:
        """
        Ottiene i dati di mercato per un simbolo e timeframe
        
        Args:
            symbol: Simbolo della coppia
            timeframe: Intervallo temporale
            limit: Numero di candele da recuperare
            force_update: Se forzare l'aggiornamento dai dati cache
            
        Returns:
            DataFrame con i dati OHLCV
        """
        cache_key = f"{symbol}_{timeframe}"
        now = datetime.now()
        
        # Verifica se questa combinazione è in cooldown per errori
        if cache_key in self.error_cooldown and now < self.error_cooldown[cache_key]:
            self.logger.debug(f"Utilizzando dati in cache per {symbol} {timeframe} (in cooldown)")
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            else:
                return pd.DataFrame()
        
        # Verifica se è necessario un aggiornamento
        needs_update = (
            force_update or 
            cache_key not in self.data_cache or
            cache_key not in self.last_update or
            now - self.last_update[cache_key] > self._get_update_interval(timeframe)
        )
        
        if needs_update:
            try:
                # Ottieni i dati dall'exchange
                data = self.exchange.get_klines(symbol, timeframe, limit=limit)
                
                if data.empty:
                    self.logger.warning(f"Nessun dato disponibile per {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Aggiorna la cache
                self.data_cache[cache_key] = data
                self.last_update[cache_key] = now
                
                # Invalida la cache degli indicatori
                if cache_key in self.indicators_cache:
                    del self.indicators_cache[cache_key]
                
                # Reset errori
                if cache_key in self.error_count:
                    self.error_count[cache_key] = 0
                if cache_key in self.error_cooldown:
                    del self.error_cooldown[cache_key]
                
                self.logger.debug(f"Dati aggiornati per {symbol} {timeframe}: {len(data)} candele")
                
            except Exception as e:
                self.logger.error(f"Errore nel recupero dati per {symbol} {timeframe}: {str(e)}")
                
                # Incrementa contatore errori
                if cache_key not in self.error_count:
                    self.error_count[cache_key] = 0
                self.error_count[cache_key] += 1
                
                # Se troppe richieste fallite, metti in cooldown
                if self.error_count[cache_key] >= 3:
                    self.error_cooldown[cache_key] = now + timedelta(minutes=30)
                    self.logger.warning(f"{symbol} {timeframe} in cooldown per 30 minuti dopo {self.error_count[cache_key]} errori")
                
                # Se la cache esiste, usa quella
                if cache_key in self.data_cache:
                    self.logger.warning(f"Utilizzo dati cache per {symbol} {timeframe}")
                    return self.data_cache[cache_key]
                else:
                    # Altrimenti ritorna un DataFrame vuoto
                    return pd.DataFrame()
        
        return self.data_cache[cache_key]