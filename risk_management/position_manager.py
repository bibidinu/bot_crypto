"""
Modulo per la gestione delle posizioni e del rischio
"""
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pandas as pd
from datetime import datetime
import uuid

from api.exchange_interface import ExchangeInterface
from strategy.strategy_base import Signal, SignalType, EntryType
from utils.logger import get_logger
from config.settings import RISK_PER_TRADE_PERCENT, MAX_POSITION_SIZE_PERCENT, TAKE_PROFIT_LEVELS

logger = get_logger(__name__)

class PositionStatus(Enum):
    """Stati possibili di una posizione"""
    PENDING = "pending"      # Ordine piazzato ma non ancora riempito
    OPEN = "open"            # Posizione aperta
    PARTIAL = "partial"      # Posizione parzialmente chiusa (take profit parziale)
    CLOSED = "closed"        # Posizione completamente chiusa
    CANCELLED = "cancelled"  # Ordine cancellato

class Position:
    """Classe per rappresentare una posizione di trading"""
    
    def __init__(self, 
                 symbol: str, 
                 entry_type: EntryType, 
                 entry_price: float, 
                 size: float,
                 stop_loss: float,
                 take_profits: List[float],
                 take_profit_sizes: List[float],
                 exchange: str = "bybit"):
        """
        Inizializza una posizione
        
        Args:
            symbol: Simbolo della coppia
            entry_type: Tipo di entrata (long o short)
            entry_price: Prezzo di entrata
            size: Dimensione della posizione
            stop_loss: Livello di stop loss
            take_profits: Lista di livelli di take profit
            take_profit_sizes: Lista di dimensioni per ogni take profit
            exchange: Nome dell'exchange
        """
        self.id = str(uuid.uuid4())
        self.symbol = symbol
        self.entry_type = entry_type
        self.entry_price = entry_price
        self.size = size
        self.original_size = size
        self.stop_loss = stop_loss
        self.take_profits = take_profits
        self.take_profit_sizes = take_profit_sizes
        self.exchange = exchange
        
        # Stato e metriche della posizione
        self.status = PositionStatus.PENDING
        self.entry_time = datetime.now()
        self.close_time = None
        self.current_price = entry_price
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.fees = 0.0
        
        # Tracking dei take profits raggiunti
        self.reached_take_profits = [False] * len(take_profits)
        
        # Trailing stop
        self.trailing_stop_active = False
        self.trailing_stop_level = 0.0
        
        # Ordini relativi a questa posizione
        self.entry_order_id = None
        self.stop_loss_order_id = None
        self.take_profit_order_ids = []
        
        # Note e metadati
        self.notes = ""
        self.metadata = {}
        
        logger.info(f"Nuova posizione creata: {self}")
    
    def __str__(self) -> str:
        """Rappresentazione stringa della posizione"""
        return (f"Position({self.symbol}, {self.entry_type.value}, "
                f"price={self.entry_price}, size={self.size}, "
                f"status={self.status.value})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte la posizione in dizionario"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "entry_type": self.entry_type.value,
            "entry_price": self.entry_price,
            "size": self.size,
            "original_size": self.original_size,
            "stop_loss": self.stop_loss,
            "take_profits": self.take_profits,
            "take_profit_sizes": self.take_profit_sizes,
            "exchange": self.exchange,
            "status": self.status.value,
            "entry_time": self.entry_time.isoformat(),
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "current_price": self.current_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees": self.fees,
            "reached_take_profits": self.reached_take_profits,
            "trailing_stop_active": self.trailing_stop_active,
            "trailing_stop_level": self.trailing_stop_level,
            "entry_order_id": self.entry_order_id,
            "stop_loss_order_id": self.stop_loss_order_id,
            "take_profit_order_ids": self.take_profit_order_ids,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Crea una posizione da un dizionario"""
        position = cls(
            symbol=data["symbol"],
            entry_type=EntryType(data["entry_type"]),
            entry_price=data["entry_price"],
            size=data["size"],
            stop_loss=data["stop_loss"],
            take_profits=data["take_profits"],
            take_profit_sizes=data["take_profit_sizes"],
            exchange=data["exchange"]
        )
        
        position.id = data["id"]
        position.original_size = data["original_size"]
        position.status = PositionStatus(data["status"])
        position.entry_time = datetime.fromisoformat(data["entry_time"])
        if data["close_time"]:
            position.close_time = datetime.fromisoformat(data["close_time"])
        position.current_price = data["current_price"]
        position.realized_pnl = data["realized_pnl"]
        position.unrealized_pnl = data["unrealized_pnl"]
        position.fees = data["fees"]
        position.reached_take_profits = data["reached_take_profits"]
        position.trailing_stop_active = data["trailing_stop_active"]
        position.trailing_stop_level = data["trailing_stop_level"]
        position.entry_order_id = data["entry_order_id"]
        position.stop_loss_order_id = data["stop_loss_order_id"]
        position.take_profit_order_ids = data["take_profit_order_ids"]
        position.notes = data["notes"]
        
        return position
    
    def update_status(self, new_status: PositionStatus) -> None:
        """
        Aggiorna lo stato della posizione
        
        Args:
            new_status: Nuovo stato
        """
        self.status = new_status
        
        if new_status == PositionStatus.CLOSED:
            self.close_time = datetime.now()
            
        logger.info(f"Posizione {self.id} aggiornata a stato {new_status.value}")
    
    def update_price(self, current_price: float) -> None:
        """
        Aggiorna il prezzo corrente e ricalcola i PnL
        
        Args:
            current_price: Prezzo corrente del mercato
        """
        self.current_price = current_price
        
        # Calcola unrealized PnL
        if self.entry_type == EntryType.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
    
    def check_take_profits(self) -> Optional[int]:
        """
        Controlla se un livello di take profit è stato raggiunto
        
        Returns:
            Indice del take profit raggiunto, o None
        """
        if self.status != PositionStatus.OPEN and self.status != PositionStatus.PARTIAL:
            return None
        
        for i, (tp, reached) in enumerate(zip(self.take_profits, self.reached_take_profits)):
            if reached:
                continue
                
            if (self.entry_type == EntryType.LONG and self.current_price >= tp) or \
               (self.entry_type == EntryType.SHORT and self.current_price <= tp):
                self.reached_take_profits[i] = True
                return i
                
        return None
    
    def check_stop_loss(self) -> bool:
        """
        Controlla se lo stop loss è stato raggiunto
        
        Returns:
            True se lo stop loss è stato raggiunto
        """
        if self.status != PositionStatus.OPEN and self.status != PositionStatus.PARTIAL:
            return False
            
        if (self.entry_type == EntryType.LONG and self.current_price <= self.stop_loss) or \
           (self.entry_type == EntryType.SHORT and self.current_price >= self.stop_loss):
            return True
            
        return False
    
    def update_trailing_stop(self, trailing_distance: float) -> None:
        """
        Aggiorna il trailing stop
        
        Args:
            trailing_distance: Distanza del trailing stop in percentuale
        """
        if not self.trailing_stop_active:
            return
            
        # Calcola nuovo livello di trailing stop
        if self.entry_type == EntryType.LONG:
            new_stop = self.current_price * (1 - trailing_distance / 100)
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
                logger.info(f"Trailing stop aggiornato a {new_stop} per posizione {self.id}")
        else:  # SHORT
            new_stop = self.current_price * (1 + trailing_distance / 100)
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
                logger.info(f"Trailing stop aggiornato a {new_stop} per posizione {self.id}")
    
    def partial_close(self, size_to_close: float, close_price: float) -> float:
        """
        Chiude parzialmente la posizione
        
        Args:
            size_to_close: Dimensione da chiudere
            close_price: Prezzo di chiusura
            
        Returns:
            PnL realizzato
        """
        if size_to_close > self.size:
            size_to_close = self.size
            
        # Calcola PnL realizzato
        if self.entry_type == EntryType.LONG:
            pnl = (close_price - self.entry_price) * size_to_close
        else:  # SHORT
            pnl = (self.entry_price - close_price) * size_to_close
            
        # Aggiorna dimensione e PnL
        self.size -= size_to_close
        self.realized_pnl += pnl
        
        if self.size <= 0:
            self.update_status(PositionStatus.CLOSED)
        else:
            self.update_status(PositionStatus.PARTIAL)
            
        logger.info(f"Chiusura parziale di {size_to_close} a {close_price} per posizione {self.id}, PnL: {pnl}")
        
        return pnl

class PositionManager:
    """Classe per gestire le posizioni e il rischio"""
    
    def __init__(self, exchange: ExchangeInterface):
        """
        Inizializza il gestore delle posizioni
        
        Args:
            exchange: Istanza dell'exchange
        """
        self.exchange = exchange
        self.logger = get_logger(__name__)
        
        # Dizionario delle posizioni (id -> Position)
        self.positions = {}
        
        # Tracking delle metriche complessive
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self.logger.info("PositionManager inizializzato")
    
    def get_wallet_balance(self) -> float:
        """
        Ottiene il bilancio del wallet
        
        Returns:
            Bilancio totale in USDT
        """
        try:
            balance = self.exchange.get_wallet_balance()
            # In un'implementazione reale, bisognerebbe analizzare
            # la struttura specifica della risposta dell'exchange
            return float(balance["totalEquity"])
        except Exception as e:
            self.logger.error(f"Errore nel recupero del bilancio: {str(e)}")
            return 0.0
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float) -> float:
        """
        Calcola la dimensione della posizione in base al rischio
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            stop_loss: Livello di stop loss
            
        Returns:
            Dimensione della posizione
        """
        # Ottieni il bilancio del wallet
        balance = self.get_wallet_balance()
        
        # Calcola il rischio in valuta
        risk_amount = balance * (RISK_PER_TRADE_PERCENT / 100)
        
        # Calcola il rischio percentuale dall'entry al stop loss
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.warning(f"Prezzi non validi per il calcolo del rischio: {entry_price}, {stop_loss}")
            return 0.0
            
        risk_percent = abs((entry_price - stop_loss) / entry_price)
        
        if risk_percent <= 0:
            self.logger.warning(f"Percentuale di rischio non valida: {risk_percent}")
            return 0.0
        
        # Calcola la dimensione della posizione
        position_size = risk_amount / (entry_price * risk_percent)
        
        # Limita la dimensione della posizione al massimo consentito
        max_position_size = balance * (MAX_POSITION_SIZE_PERCENT / 100) / entry_price
        position_size = min(position_size, max_position_size)
        
        self.logger.info(f"Dimensione posizione calcolata per {symbol}: {position_size}")
        
        return position_size
    
    def open_position(self, signal: Signal) -> Optional[Position]:
        """
        Apre una nuova posizione in base al segnale
        
        Args:
            signal: Segnale di trading
            
        Returns:
            Nuova posizione o None se non è stato possibile aprirla
        """
        try:
            if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
                self.logger.info(f"Nessuna posizione da aprire per segnale {signal.signal_type.value}")
                return None
                
            if signal.entry_type is None:
                self.logger.warning(f"Tipo di entrata mancante nel segnale: {signal}")
                return None
                
            # Verifica se c'è già una posizione aperta per questo simbolo
            for pos in self.positions.values():
                if pos.symbol == signal.symbol and pos.status in [PositionStatus.OPEN, PositionStatus.PENDING, PositionStatus.PARTIAL]:
                    self.logger.warning(f"Posizione già esistente per {signal.symbol}: {pos}")
                    return None
            
            # Calcola dimensione della posizione
            size = self.calculate_position_size(
                signal.symbol, 
                signal.price, 
                signal.stop_loss if signal.stop_loss else signal.price * 0.95  # Default SL a -5%
            )
            
            if size <= 0:
                self.logger.warning(f"Dimensione posizione non valida: {size}")
                return None
                
            # Prepara i take profit
            take_profits = signal.take_profits or []
            
            # Calcola le dimensioni per i take profit
            take_profit_sizes = []
            remaining_size = size
            
            for tp in TAKE_PROFIT_LEVELS:
                tp_size = size * (tp["size_percent"] / 100)
                take_profit_sizes.append(tp_size)
                remaining_size -= tp_size
                
            # Assicurati che tutte le dimensioni siano corrette
            if remaining_size > 0 and len(take_profit_sizes) > 0:
                take_profit_sizes[-1] += remaining_size
                
            # Crea la posizione
            side = "Buy" if signal.entry_type == EntryType.LONG else "Sell"
            
            # Piazza l'ordine sull'exchange
            try:
                order = self.exchange.place_order(
                    symbol=signal.symbol,
                    side=side,
                    order_type="Market",
                    qty=size,
                    reduce_only=False,
                    stop_loss=signal.stop_loss,
                    take_profit=take_profits[0] if take_profits else None
                )
                
                # Crea l'oggetto posizione
                position = Position(
                    symbol=signal.symbol,
                    entry_type=signal.entry_type,
                    entry_price=signal.price,
                    size=size,
                    stop_loss=signal.stop_loss,
                    take_profits=take_profits,
                    take_profit_sizes=take_profit_sizes,
                    exchange="bybit"
                )
                
                # Aggiorna con l'ID dell'ordine
                position.entry_order_id = order.get("orderId")
                position.update_status(PositionStatus.OPEN)
                
                # Aggiungi alla lista delle posizioni
                self.positions[position.id] = position
                
                self.logger.info(f"Posizione aperta: {position}")
                
                return position
                
            except Exception as e:
                self.logger.error(f"Errore nell'apertura della posizione: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Errore generale nell'apertura della posizione: {str(e)}")
            return None
    
    def close_position(self, position_id: str, price: Optional[float] = None) -> bool:
        """
        Chiude una posizione esistente
        
        Args:
            position_id: ID della posizione
            price: Prezzo di chiusura (se None, usa il prezzo di mercato)
            
        Returns:
            True se la chiusura è avvenuta con successo
        """
        try:
            if position_id not in self.positions:
                self.logger.warning(f"Posizione non trovata: {position_id}")
                return False
                
            position = self.positions[position_id]
            
            if position.status == PositionStatus.CLOSED:
                self.logger.info(f"Posizione già chiusa: {position}")
                return True
                
            # Se il prezzo non è specificato, usa il prezzo corrente
            close_price = price or position.current_price
            
            # Piazza l'ordine di chiusura
            side = "Sell" if position.entry_type == EntryType.LONG else "Buy"
            
            try:
                order = self.exchange.place_order(
                    symbol=position.symbol,
                    side=side,
                    order_type="Market",
                    qty=position.size,
                    reduce_only=True
                )
                
                # Calcola il PnL
                if position.entry_type == EntryType.LONG:
                    pnl = (close_price - position.entry_price) * position.size
                else:  # SHORT
                    pnl = (position.entry_price - close_price) * position.size
                    
                # Aggiorna la posizione
                position.realized_pnl += pnl
                position.unrealized_pnl = 0
                position.size = 0
                position.update_status(PositionStatus.CLOSED)
                
                # Aggiorna le statistiche
                self.total_profit += position.realized_pnl
                self.total_trades += 1
                
                if position.realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                    
                self.logger.info(f"Posizione chiusa: {position}, PnL: {pnl}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Errore nella chiusura della posizione: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore generale nella chiusura della posizione: {str(e)}")
            return False
    
    def update_positions(self) -> None:
        """
        Aggiorna tutte le posizioni con i prezzi correnti
        e verifica stop loss e take profit
        """
        try:
            for position_id, position in list(self.positions.items()):
                if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIAL]:
                    continue
                    
                # Ottieni il prezzo corrente
                try:
                    ticker = self.exchange.get_ticker(position.symbol)
                    current_price = float(ticker["lastPrice"])
                    
                    # Aggiorna il prezzo della posizione
                    position.update_price(current_price)
                    
                    # Verifica take profit
                    tp_index = position.check_take_profits()
                    if tp_index is not None:
                        self.logger.info(f"Take profit {tp_index+1} raggiunto per posizione {position_id}")
                        
                        # Chiudi parzialmente la posizione
                        tp_size = position.take_profit_sizes[tp_index]
                        tp_price = position.take_profits[tp_index]
                        
                        # Piazza l'ordine di chiusura parziale
                        side = "Sell" if position.entry_type == EntryType.LONG else "Buy"
                        
                        try:
                            order = self.exchange.place_order(
                                symbol=position.symbol,
                                side=side,
                                order_type="Market",
                                qty=tp_size,
                                reduce_only=True
                            )
                            
                            # Aggiorna la posizione
                            position.partial_close(tp_size, tp_price)
                            
                            # Aggiorna lo stop loss dopo il primo take profit
                            if tp_index == 0 and position.status == PositionStatus.PARTIAL:
                                # Sposta lo stop loss al break even
                                position.stop_loss = position.entry_price
                                self.logger.info(f"Stop loss spostato al break even per posizione {position_id}")
                                
                        except Exception as e:
                            self.logger.error(f"Errore nella chiusura parziale: {str(e)}")
                    
                    # Verifica stop loss
                    if position.check_stop_loss():
                        self.logger.info(f"Stop loss raggiunto per posizione {position_id}")
                        self.close_position(position_id, position.stop_loss)
                    
                    # Aggiorna trailing stop se attivo
                    position.update_trailing_stop(1.0)  # 1% di distanza
                    
                except Exception as e:
                    self.logger.error(f"Errore nell'aggiornamento della posizione {position_id}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Errore generale nell'aggiornamento delle posizioni: {str(e)}")
    
    def get_open_positions(self) -> List[Position]:
        """
        Ottiene tutte le posizioni aperte
        
        Returns:
            Lista di posizioni aperte
        """
        return [p for p in self.positions.values() 
                if p.status in [PositionStatus.OPEN, PositionStatus.PARTIAL]]
    
    def get_position_history(self, limit: int = 50) -> List[Position]:
        """
        Ottiene la cronologia delle posizioni chiuse
        
        Args:
            limit: Numero massimo di posizioni da recuperare
            
        Returns:
            Lista di posizioni chiuse
        """
        closed_positions = [p for p in self.positions.values() 
                           if p.status == PositionStatus.CLOSED]
        
        # Ordina per data di chiusura (più recenti prima)
        closed_positions.sort(key=lambda p: p.close_time or datetime.min, reverse=True)
        
        return closed_positions[:limit]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche di performance
        
        Returns:
            Dizionario con le statistiche
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calcola il profit factor se ci sono trade perdenti
        avg_win = 0
        avg_loss = 0
        total_win = 0
        total_loss = 0
        
        for p in self.positions.values():
            if p.status == PositionStatus.CLOSED:
                if p.realized_pnl > 0:
                    total_win += p.realized_pnl
                else:
                    total_loss += abs(p.realized_pnl)
        
        if self.winning_trades > 0:
            avg_win = total_win / self.winning_trades
            
        if self.losing_trades > 0:
            avg_loss = total_loss / self.losing_trades
            
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        return {
            "total_profit": self.total_profit,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "open_positions": len(self.get_open_positions())
        }
    
    def save_positions(self, filepath: str) -> bool:
        """
        Salva le posizioni su file
        
        Args:
            filepath: Percorso del file
            
        Returns:
            True se il salvataggio è avvenuto con successo
        """
        import json
        
        try:
            positions_dict = {id: pos.to_dict() for id, pos in self.positions.items()}
            
            with open(filepath, 'w') as f:
                json.dump(positions_dict, f, indent=4)
                
            self.logger.info(f"Posizioni salvate in {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio delle posizioni: {str(e)}")
            return False
    
    def load_positions(self, filepath: str) -> bool:
        """
        Carica le posizioni da file
        
        Args:
            filepath: Percorso del file
            
        Returns:
            True se il caricamento è avvenuto con successo
        """
        import json
        import os
        
        if not os.path.exists(filepath):
            self.logger.warning(f"File delle posizioni non trovato: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                positions_dict = json.load(f)
                
            self.positions = {}
            for id, pos_dict in positions_dict.items():
                self.positions[id] = Position.from_dict(pos_dict)
                
            self.logger.info(f"Posizioni caricate da {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento delle posizioni: {str(e)}")
            return False
