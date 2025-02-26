"""
Classe base per tutte le strategie di trading
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from api.exchange_interface import ExchangeInterface
from config.settings import DEFAULT_TIMEFRAME
from utils.logger import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    """Tipi di segnali di trading"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"

class EntryType(Enum):
    """Tipi di entrata"""
    LONG = "long"
    SHORT = "short"

class Signal:
    """Classe per rappresentare un segnale di trading"""
    
    def __init__(self, 
                 symbol: str, 
                 signal_type: SignalType, 
                 entry_type: Optional[EntryType] = None,
                 price: float = 0.0, 
                 strength: float = 0.0, 
                 stop_loss: Optional[float] = None, 
                 take_profits: Optional[List[float]] = None,
                 reason: str = "",
                 indicators: Dict[str, Any] = None):
        """
        Inizializza un segnale di trading
        
        Args:
            symbol: Simbolo della coppia
            signal_type: Tipo di segnale
            entry_type: Tipo di entrata (long o short)
            price: Prezzo consigliato per l'entrata
            strength: Forza del segnale (0.0-1.0)
            stop_loss: Livello di stop loss
            take_profits: Lista di livelli take profit
            reason: Motivo del segnale
            indicators: Valori degli indicatori rilevanti
        """
        self.symbol = symbol
        self.signal_type = signal_type
        self.entry_type = entry_type
        self.price = price
        self.strength = min(max(strength, 0.0), 1.0)  # Limita tra 0 e 1
        self.stop_loss = stop_loss
        self.take_profits = take_profits or []
        self.reason = reason
        self.indicators = indicators or {}
        self.timestamp = pd.Timestamp.now()
    
    def __str__(self) -> str:
        """Rappresentazione stringa del segnale"""
        return (f"Signal({self.symbol}, {self.signal_type.value}, "
                f"strength={self.strength:.2f}, price={self.price})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte il segnale in dizionario"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "entry_type": self.entry_type.value if self.entry_type else None,
            "price": self.price,
            "strength": self.strength,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "reason": self.reason,
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Crea un segnale da un dizionario"""
        return cls(
            symbol=data["symbol"],
            signal_type=SignalType(data["signal_type"]),
            entry_type=EntryType(data["entry_type"]) if data.get("entry_type") else None,
            price=data["price"],
            strength=data["strength"],
            stop_loss=data.get("stop_loss"),
            take_profits=data.get("take_profits"),
            reason=data.get("reason", ""),
            indicators=data.get("indicators", {})
        )

class StrategyBase(ABC):
    """Classe base per tutte le strategie di trading"""
    
    def __init__(self, exchange: ExchangeInterface, name: str = "BaseStrategy"):
        """
        Inizializza la strategia
        
        Args:
            exchange: Istanza dell'exchange
            name: Nome della strategia
        """
        self.exchange = exchange
        self.name = name
        self.timeframe = DEFAULT_TIMEFRAME
        self.logger = get_logger(f"{__name__}.{name}")
        
        # Performance metrics
        self.trades_total = 0
        self.trades_won = 0
        self.trades_lost = 0
        self.profit_factor = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        
        self.logger.info(f"Strategia {self.name} inizializzata")
    
    @abstractmethod
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Genera un segnale di trading per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di mercato
            
        Returns:
            Un oggetto Signal con il suggerimento
        """
        pass
    
    def get_market_data(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """
        Ottiene i dati di mercato per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            limit: Numero di candele da recuperare
            
        Returns:
            DataFrame con i dati di mercato
        """
        try:
            data = self.exchange.get_klines(symbol, self.timeframe, limit=limit)
            return data
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei dati per {symbol}: {str(e)}")
            raise
    
    def analyze(self, symbol: str, limit: int = 200) -> Signal:
        """
        Analizza il mercato e genera un segnale per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            limit: Numero di candele da analizzare
            
        Returns:
            Un oggetto Signal con il suggerimento
        """
        try:
            data = self.get_market_data(symbol, limit)
            if data.empty:
                self.logger.warning(f"Nessun dato disponibile per {symbol}")
                return Signal(symbol, SignalType.HOLD, price=0, strength=0, 
                             reason="Dati insufficienti")
            
            return self.generate_signal(symbol, data)
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi di {symbol}: {str(e)}")
            return Signal(symbol, SignalType.HOLD, price=0, strength=0, 
                         reason=f"Errore: {str(e)}")
    
    def update_performance(self, win: bool, profit_pct: float) -> None:
        """
        Aggiorna le metriche di performance della strategia
        
        Args:
            win: Se il trade è stato vincente
            profit_pct: Percentuale di profitto/perdita
        """
        self.trades_total += 1
        
        if win:
            self.trades_won += 1
        else:
            self.trades_lost += 1
        
        self.win_rate = self.trades_won / self.trades_total if self.trades_total > 0 else 0
        
        # Altre metriche possono essere calcolate qui
        self.logger.info(f"Performance aggiornata: {self.trades_won}/{self.trades_total} trades vinti ({self.win_rate:.2%})")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche di performance della strategia
        
        Returns:
            Dizionario con le statistiche di performance
        """
        return {
            "name": self.name,
            "trades_total": self.trades_total,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown
        }
    
    def set_timeframe(self, timeframe: str) -> None:
        """
        Imposta l'intervallo temporale per l'analisi
        
        Args:
            timeframe: Intervallo temporale (es. "15m")
        """
        self.timeframe = timeframe
        self.logger.info(f"Timeframe impostato a {timeframe}")
    
    def __str__(self) -> str:
        """Rappresentazione stringa della strategia"""
        return f"{self.name}(win_rate={self.win_rate:.2%}, trades={self.trades_total})"
