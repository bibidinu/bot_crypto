"""
Interfaccia generica per exchange di criptovalute
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd

class ExchangeInterface(ABC):
    """
    Classe astratta base per tutte le API di exchange
    Implementa tutti i metodi richiesti da qualsiasi exchange supportato
    """
    
    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 200, 
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Ottiene i dati OHLCV (candele) per una coppia di trading
        
        Args:
            symbol: Simbolo della coppia
            interval: Intervallo temporale
            limit: Numero massimo di candele da recuperare
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            
        Returns:
            DataFrame pandas con i dati OHLCV
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene informazioni correnti sul prezzo di un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Dati del ticker
        """
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """
        Ottiene l'order book corrente per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            limit: Profondità dell'order book
            
        Returns:
            Dati dell'order book
        """
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Piazza un nuovo ordine
        
        Args:
            symbol: Simbolo della coppia
            side: Direzione dell'ordine ("Buy" o "Sell")
            order_type: Tipo di ordine ("Market", "Limit")
            qty: Quantità da acquistare/vendere
            price: Prezzo per ordini limit (opzionale per ordini market)
            **kwargs: Altri parametri specifici dell'exchange
            
        Returns:
            Dettagli dell'ordine piazzato
        """
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancella un ordine esistente
        
        Args:
            symbol: Simbolo della coppia
            order_id: ID dell'ordine da cancellare
            
        Returns:
            Dettagli della cancellazione
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ottiene gli ordini aperti
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            
        Returns:
            Lista di ordini aperti
        """
        pass
    
    @abstractmethod
    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Ottiene la cronologia degli ordini
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            limit: Numero massimo di ordini da recuperare
            
        Returns:
            Lista di ordini storici
        """
        pass
    
    @abstractmethod
    def get_wallet_balance(self, coin: Optional[str] = None) -> Dict[str, Any]:
        """
        Ottiene il bilancio del wallet
        
        Args:
            coin: Simbolo della valuta (opzionale)
            
        Returns:
            Bilancio del wallet
        """
        pass
    
    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Ottiene le posizioni aperte
        
        Args:
            symbol: Simbolo della coppia (opzionale)
            
        Returns:
            Lista di posizioni aperte
        """
        pass
