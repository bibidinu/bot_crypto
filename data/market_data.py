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
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge indicatori tecnici al DataFrame
        
        Args:
            data: DataFrame con i dati OHLCV
            
        Returns:
            DataFrame con indicatori aggiunti
        """
        if data.empty:
            return data
            
        # Crea una copia per non modificare l'originale
        df = data.copy()
        
        # --- RSI ---
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # --- MACD ---
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # --- EMA ---
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # --- Bollinger Bands ---
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_mavg'] = bollinger.bollinger_mavg()
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # --- ATR ---
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        
        # --- Stochastic ---
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # --- ADX ---
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        return df
    
    def get_data_with_indicators(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, 
                                limit: int = DATA_LOOKBACK_PERIODS, 
                                force_update: bool = False) -> pd.DataFrame:
        """
        Ottiene i dati di mercato con indicatori tecnici
        
        Args:
            symbol: Simbolo della coppia
            timeframe: Intervallo temporale
            limit: Numero di candele da recuperare
            force_update: Se forzare l'aggiornamento dai dati cache
            
        Returns:
            DataFrame con dati OHLCV e indicatori
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Verifica se i dati con indicatori sono in cache
        if not force_update and cache_key in self.indicators_cache:
            return self.indicators_cache[cache_key]
        
        # Ottieni i dati di mercato base
        data = self.get_market_data(symbol, timeframe, limit, force_update)
        
        if data.empty:
            return data
        
        # Aggiungi indicatori
        try:
            data_with_indicators = self.add_indicators(data)
            
            # Aggiorna la cache
            self.indicators_cache[cache_key] = data_with_indicators
            
            return data_with_indicators
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta degli indicatori per {symbol} {timeframe}: {str(e)}")
            return data
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Ottiene il prezzo più recente per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Ultimo prezzo
        """
        try:
            ticker = self.exchange.get_ticker(symbol)
            price = float(ticker.get("lastPrice", 0.0))
            return price
        except Exception as e:
            self.logger.error(f"Errore nel recupero del prezzo per {symbol}: {str(e)}")
            
            # Prova a ottenere il prezzo dai dati di mercato
            cache_key = f"{symbol}_{DEFAULT_TIMEFRAME}"
            if cache_key in self.data_cache:
                data = self.data_cache[cache_key]
                if not data.empty:
                    return data.iloc[-1]['close']
            
            return 0.0
    
    def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene un riepilogo del mercato per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Dizionario con il riepilogo del mercato
        """
        try:
            # Ottieni dati con indicatori
            data = self.get_data_with_indicators(symbol)
            
            if data.empty:
                return {}
            
            # Ottieni l'ultimo candle
            last = data.iloc[-1]
            
            # Calcola rendimenti
            daily_change = ((last['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']) * 100
            
            weekly_data = self.get_data_with_indicators(symbol, "1d", limit=7)
            weekly_change = 0.0
            if not weekly_data.empty and len(weekly_data) > 1:
                weekly_change = ((last['close'] - weekly_data.iloc[0]['close']) / weekly_data.iloc[0]['close']) * 100
            
            # Determina il trend
            if last['ema_20'] > last['ema_50'] and last['ema_50'] > last['ema_200']:
                trend = "bullish"
            elif last['ema_20'] < last['ema_50'] and last['ema_50'] < last['ema_200']:
                trend = "bearish"
            else:
                trend = "neutral"
            
            # Calcola supporti e resistenze
            supports = self._find_support_levels(data)
            resistances = self._find_resistance_levels(data)
            
            # Prossimi livelli di supporto e resistenza
            next_support = max([s for s in supports if s < last['close']], default=0.0)
            next_resistance = min([r for r in resistances if r > last['close']], default=0.0)
            
            # Volatilità
            volatility = last['atr'] / last['close'] * 100
            
            return {
                "symbol": symbol,
                "last_price": last['close'],
                "daily_change_pct": daily_change,
                "weekly_change_pct": weekly_change,
                "trend": trend,
                "rsi": last['rsi'],
                "volume": last['volume'],
                "atr": last['atr'],
                "volatility_pct": volatility,
                "next_support": next_support,
                "next_resistance": next_resistance,
                "bb_upper": last['bb_upper'],
                "bb_lower": last['bb_lower'],
                "bb_width": (last['bb_upper'] - last['bb_lower']) / last['bb_mavg'] * 100,
                "timestamp": last.get('datetime', datetime.now())
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero del riepilogo per {symbol}: {str(e)}")
            return {}
    
    def _find_support_levels(self, data: pd.DataFrame, window: int = 10) -> List[float]:
        """
        Trova i livelli di supporto
        
        Args:
            data: DataFrame con i dati OHLCV
            window: Finestra per la ricerca di minimi
            
        Returns:
            Lista di livelli di supporto
        """
        try:
            if data.empty:
                return []
                
            df = data.copy()
            
            # Trova i minimi locali
            df['min'] = df['low'].rolling(window=window, center=True).min()
            
            # Un punto è un minimo locale se il min nella finestra è pari al low del punto
            df['is_min'] = (df['low'] == df['min'])
            
            # Filtra solo i minimi locali
            mins = df[df['is_min']]['low'].tolist()
            
            # Raggruppa i livelli simili (entro l'1%)
            supports = []
            for m in mins:
                # Verifica se è già presente un supporto simile
                if not any(abs(s - m) / m < 0.01 for s in supports):
                    supports.append(m)
            
            return sorted(supports)
            
        except Exception as e:
            self.logger.error(f"Errore nella ricerca dei supporti: {str(e)}")
            return []
    
    def _find_resistance_levels(self, data: pd.DataFrame, window: int = 10) -> List[float]:
        """
        Trova i livelli di resistenza
        
        Args:
            data: DataFrame con i dati OHLCV
            window: Finestra per la ricerca di massimi
            
        Returns:
            Lista di livelli di resistenza
        """
        try:
            if data.empty:
                return []
                
            df = data.copy()
            
            # Trova i massimi locali
            df['max'] = df['high'].rolling(window=window, center=True).max()
            
            # Un punto è un massimo locale se il max nella finestra è pari al high del punto
            df['is_max'] = (df['high'] == df['max'])
            
            # Filtra solo i massimi locali
            maxs = df[df['is_max']]['high'].tolist()
            
            # Raggruppa i livelli simili (entro l'1%)
            resistances = []
            for m in maxs:
                # Verifica se è già presente una resistenza simile
                if not any(abs(r - m) / m < 0.01 for r in resistances):
                    resistances.append(m)
            
            return sorted(resistances)
            
        except Exception as e:
            self.logger.error(f"Errore nella ricerca delle resistenze: {str(e)}")
            return []