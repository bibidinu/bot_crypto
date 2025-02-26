"""
Modelli di dati per il database
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import json

class Trade:
    """Modello per un trade"""
    
    def __init__(self, 
                 symbol: str,
                 entry_type: str,
                 entry_price: float,
                 size: float,
                 exit_price: Optional[float] = None,
                 profit: Optional[float] = None,
                 profit_percent: Optional[float] = None,
                 entry_time: Optional[str] = None,
                 exit_time: Optional[str] = None,
                 strategy: Optional[str] = None,
                 status: str = "open",
                 id: Optional[str] = None,
                 **kwargs):
        """
        Inizializza un trade
        
        Args:
            symbol: Simbolo della coppia
            entry_type: Tipo di entrata (long/short)
            entry_price: Prezzo di entrata
            size: Dimensione del trade
            exit_price: Prezzo di uscita (opzionale)
            profit: Profitto in valuta (opzionale)
            profit_percent: Profitto in percentuale (opzionale)
            entry_time: Timestamp di entrata (opzionale)
            exit_time: Timestamp di uscita (opzionale)
            strategy: Strategia utilizzata (opzionale)
            status: Stato del trade (open/closed)
            id: ID del trade (generato se non fornito)
            **kwargs: Metadati aggiuntivi
        """
        self.symbol = symbol
        self.entry_type = entry_type
        self.entry_price = entry_price
        self.size = size
        self.exit_price = exit_price
        self.profit = profit
        self.profit_percent = profit_percent
        self.entry_time = entry_time or datetime.now().isoformat()
        self.exit_time = exit_time
        self.strategy = strategy
        self.status = status
        self.id = id or f"trade_{uuid.uuid4()}"
        
        # Metadati aggiuntivi
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte il trade in un dizionario
        
        Returns:
            Dizionario con i dati del trade
        """
        data = {
            "id": self.id,
            "symbol": self.symbol,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "size": self.size,
            "exit_price": self.exit_price,
            "profit": self.profit,
            "profit_percent": self.profit_percent,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "strategy": self.strategy,
            "status": self.status
        }
        
        # Aggiungi i metadati
        data.update(self.metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """
        Crea un trade da un dizionario
        
        Args:
            data: Dizionario con i dati del trade
            
        Returns:
            Istanza di Trade
        """
        # Estrai i campi principali
        trade_data = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "entry_type": data.get("entry_type"),
            "entry_price": data.get("entry_price"),
            "size": data.get("size"),
            "exit_price": data.get("exit_price"),
            "profit": data.get("profit"),
            "profit_percent": data.get("profit_percent"),
            "entry_time": data.get("entry_time"),
            "exit_time": data.get("exit_time"),
            "strategy": data.get("strategy"),
            "status": data.get("status")
        }
        
        # Estrai i metadati
        metadata = {k: v for k, v in data.items() if k not in trade_data}
        
        # Crea l'istanza
        instance = cls(**trade_data)
        instance.metadata = metadata
        
        return instance
    
    def close(self, exit_price: float, exit_time: Optional[str] = None) -> None:
        """
        Chiude il trade
        
        Args:
            exit_price: Prezzo di uscita
            exit_time: Timestamp di uscita (opzionale)
        """
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now().isoformat()
        self.status = "closed"
        
        # Calcola il profitto
        if self.entry_type.lower() == "long":
            self.profit = (exit_price - self.entry_price) * self.size
        else:  # short
            self.profit = (self.entry_price - exit_price) * self.size
        
        # Calcola il profitto percentuale
        if self.entry_price > 0:
            if self.entry_type.lower() == "long":
                self.profit_percent = (exit_price / self.entry_price - 1) * 100
            else:  # short
                self.profit_percent = (self.entry_price / exit_price - 1) * 100

class Signal:
    """Modello per un segnale di trading"""
    
    def __init__(self,
                 symbol: str,
                 signal_type: str,
                 price: float,
                 strength: float = 0.0,
                 entry_type: Optional[str] = None,
                 stop_loss: Optional[float] = None,
                 take_profits: Optional[List[float]] = None,
                 reason: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 id: Optional[str] = None,
                 **kwargs):
        """
        Inizializza un segnale
        
        Args:
            symbol: Simbolo della coppia
            signal_type: Tipo di segnale (buy/sell/hold)
            price: Prezzo al momento del segnale
            strength: Forza del segnale (0-1)
            entry_type: Tipo di entrata (long/short)
            stop_loss: Livello di stop loss (opzionale)
            take_profits: Livelli di take profit (opzionale)
            reason: Motivo del segnale (opzionale)
            timestamp: Timestamp del segnale (opzionale)
            id: ID del segnale (generato se non fornito)
            **kwargs: Metadati aggiuntivi
        """
        self.symbol = symbol
        self.signal_type = signal_type
        self.price = price
        self.strength = strength
        self.entry_type = entry_type
        self.stop_loss = stop_loss
        self.take_profits = take_profits or []
        self.reason = reason
        self.timestamp = timestamp or datetime.now().isoformat()
        self.id = id or f"signal_{uuid.uuid4()}"
        
        # Metadati aggiuntivi
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte il segnale in un dizionario
        
        Returns:
            Dizionario con i dati del segnale
        """
        data = {
            "id": self.id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "price": self.price,
            "strength": self.strength,
            "entry_type": self.entry_type,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "reason": self.reason,
            "timestamp": self.timestamp
        }
        
        # Aggiungi i metadati
        data.update(self.metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """
        Crea un segnale da un dizionario
        
        Args:
            data: Dizionario con i dati del segnale
            
        Returns:
            Istanza di Signal
        """
        # Estrai i campi principali
        signal_data = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "signal_type": data.get("signal_type"),
            "price": data.get("price"),
            "strength": data.get("strength"),
            "entry_type": data.get("entry_type"),
            "stop_loss": data.get("stop_loss"),
            "take_profits": data.get("take_profits"),
            "reason": data.get("reason"),
            "timestamp": data.get("timestamp")
        }
        
        # Estrai i metadati
        metadata = {k: v for k, v in data.items() if k not in signal_data}
        
        # Crea l'istanza
        instance = cls(**signal_data)
        instance.metadata = metadata
        
        return instance

class Position:
    """Modello per una posizione di trading"""
    
    def __init__(self,
                 symbol: str,
                 entry_type: str,
                 entry_price: float,
                 size: float,
                 stop_loss: Optional[float] = None,
                 take_profits: Optional[List[float]] = None,
                 entry_time: Optional[str] = None,
                 status: str = "open",
                 realized_pnl: float = 0.0,
                 unrealized_pnl: float = 0.0,
                 id: Optional[str] = None,
                 **kwargs):
        """
        Inizializza una posizione
        
        Args:
            symbol: Simbolo della coppia
            entry_type: Tipo di entrata (long/short)
            entry_price: Prezzo di entrata
            size: Dimensione della posizione
            stop_loss: Livello di stop loss (opzionale)
            take_profits: Livelli di take profit (opzionale)
            entry_time: Timestamp di entrata (opzionale)
            status: Stato della posizione (open/partial/closed)
            realized_pnl: Profitto/perdita realizzato
            unrealized_pnl: Profitto/perdita non realizzato
            id: ID della posizione (generato se non fornito)
            **kwargs: Metadati aggiuntivi
        """
        self.symbol = symbol
        self.entry_type = entry_type
        self.entry_price = entry_price
        self.size = size
        self.original_size = size  # Dimensione originale della posizione
        self.stop_loss = stop_loss
        self.take_profits = take_profits or []
        self.entry_time = entry_time or datetime.now().isoformat()
        self.status = status
        self.realized_pnl = realized_pnl
        self.unrealized_pnl = unrealized_pnl
        self.id = id or f"position_{uuid.uuid4()}"
        
        # Metadati aggiuntivi
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la posizione in un dizionario
        
        Returns:
            Dizionario con i dati della posizione
        """
        data = {
            "id": self.id,
            "symbol": self.symbol,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "size": self.size,
            "original_size": self.original_size,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "entry_time": self.entry_time,
            "status": self.status,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl
        }
        
        # Aggiungi i metadati
        data.update(self.metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Crea una posizione da un dizionario
        
        Args:
            data: Dizionario con i dati della posizione
            
        Returns:
            Istanza di Position
        """
        # Estrai i campi principali
        position_data = {
            "id": data.get("id"),
            "symbol": data.get("symbol"),
            "entry_type": data.get("entry_type"),
            "entry_price": data.get("entry_price"),
            "size": data.get("size"),
            "stop_loss": data.get("stop_loss"),
            "take_profits": data.get("take_profits"),
            "entry_time": data.get("entry_time"),
            "status": data.get("status"),
            "realized_pnl": data.get("realized_pnl", 0.0),
            "unrealized_pnl": data.get("unrealized_pnl", 0.0)
        }
        
        # Estrai i metadati
        metadata = {k: v for k, v in data.items() if k not in position_data}
        
        # Crea l'istanza
        instance = cls(**position_data)
        
        # Imposta la dimensione originale
        if "original_size" in data:
            instance.original_size = data["original_size"]
        else:
            instance.original_size = instance.size
            
        instance.metadata = metadata
        
        return instance
    
    def update_pnl(self, current_price: float) -> None:
        """
        Aggiorna il PnL non realizzato
        
        Args:
            current_price: Prezzo corrente
        """
        if self.entry_type.lower() == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
    
    def partial_close(self, size: float, close_price: float) -> float:
        """
        Chiude parzialmente la posizione
        
        Args:
            size: Dimensione da chiudere
            close_price: Prezzo di chiusura
            
        Returns:
            PnL realizzato dalla chiusura parziale
        """
        if size > self.size:
            size = self.size
            
        # Calcola il PnL realizzato
        if self.entry_type.lower() == "long":
            pnl = (close_price - self.entry_price) * size
        else:  # short
            pnl = (self.entry_price - close_price) * size
            
        # Aggiorna la posizione
        self.size -= size
        self.realized_pnl += pnl
        
        # Aggiorna lo stato
        if self.size <= 0:
            self.status = "closed"
        else:
            self.status = "partial"
            
        return pnl
    
    def close(self, close_price: float) -> float:
        """
        Chiude completamente la posizione
        
        Args:
            close_price: Prezzo di chiusura
            
        Returns:
            PnL totale realizzato
        """
        # Chiudi la posizione rimanente
        pnl = self.partial_close(self.size, close_price)
        
        # Assicurati che lo stato sia "closed"
        self.status = "closed"
        
        return pnl

class Performance:
    """Modello per i dati di performance"""
    
    def __init__(self,
                 start_time: str,
                 end_time: str,
                 total_trades: int,
                 win_trades: int,
                 loss_trades: int,
                 win_rate: float,
                 profit_factor: float,
                 total_profit: float,
                 max_drawdown: float,
                 id: Optional[str] = None,
                 **kwargs):
        """
        Inizializza un record di performance
        
        Args:
            start_time: Timestamp di inizio
            end_time: Timestamp di fine
            total_trades: Numero totale di trade
            win_trades: Numero di trade vincenti
            loss_trades: Numero di trade perdenti
            win_rate: Rapporto di vincita (0-1)
            profit_factor: Fattore di profitto
            total_profit: Profitto totale
            max_drawdown: Drawdown massimo
            id: ID della performance (generato se non fornito)
            **kwargs: Metadati aggiuntivi
        """
        self.start_time = start_time
        self.end_time = end_time
        self.total_trades = total_trades
        self.win_trades = win_trades
        self.loss_trades = loss_trades
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.total_profit = total_profit
        self.max_drawdown = max_drawdown
        self.id = id or f"perf_{uuid.uuid4()}"
        
        # Metadati aggiuntivi
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la performance in un dizionario
        
        Returns:
            Dizionario con i dati della performance
        """
        data = {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_trades": self.total_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown
        }
        
        # Aggiungi i metadati
        data.update(self.metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Performance':
        """
        Crea una performance da un dizionario
        
        Args:
            data: Dizionario con i dati della performance
            
        Returns:
            Istanza di Performance
        """
        # Estrai i campi principali
        perf_data = {
            "id": data.get("id"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time"),
            "total_trades": data.get("total_trades"),
            "win_trades": data.get("win_trades"),
            "loss_trades": data.get("loss_trades"),
            "win_rate": data.get("win_rate"),
            "profit_factor": data.get("profit_factor"),
            "total_profit": data.get("total_profit"),
            "max_drawdown": data.get("max_drawdown")
        }
        
        # Estrai i metadati
        metadata = {k: v for k, v in data.items() if k not in perf_data}
        
        # Crea l'istanza
        instance = cls(**perf_data)
        instance.metadata = metadata
        
        return instance
