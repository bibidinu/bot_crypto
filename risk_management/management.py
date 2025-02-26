"""
Modulo per il calcolo e la gestione del rischio
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from api.exchange_interface import ExchangeInterface
from utils.logger import get_logger
from config.settings import (
    RISK_PER_TRADE_PERCENT, MAX_CONCURRENT_TRADES,
    MAX_POSITION_SIZE_PERCENT, DEFAULT_TRADING_PAIRS
)

logger = get_logger(__name__)

class RiskCalculator:
    """Classe per il calcolo e la gestione del rischio"""
    
    def __init__(self, exchange: ExchangeInterface):
        """
        Inizializza il calcolatore di rischio
        
        Args:
            exchange: Istanza dell'exchange
        """
        self.exchange = exchange
        self.logger = get_logger(__name__)
        
        # Rischio per trade e capitale massimo per simbolo
        self.risk_per_trade_percent = RISK_PER_TRADE_PERCENT
        self.max_position_size_percent = MAX_POSITION_SIZE_PERCENT
        self.max_concurrent_trades = MAX_CONCURRENT_TRADES
        
        # Volatilità dei simboli
        self.symbol_volatility = {}
        self.volatility_update_time = {}
        
        # Correlazione tra simboli
        self.correlation_matrix = None
        self.correlation_update_time = None
        
        # Rischio complessivo del portafoglio
        self.portfolio_risk = 0.0
        
        self.logger.info("RiskCalculator inizializzato")
    
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
    
    def calculate_risk(self, symbol: str, entry_price: float, 
                      stop_loss: float, position_size: float) -> Dict[str, float]:
        """
        Calcola il rischio per un trade specifico
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            stop_loss: Livello di stop loss
            position_size: Dimensione della posizione
            
        Returns:
            Dizionario con le metriche di rischio
        """
        try:
            # Bilancio del wallet
            balance = self.get_wallet_balance()
            
            # Rischio in valore assoluto
            risk_value = abs(entry_price - stop_loss) * position_size
            
            # Rischio in percentuale del capitale
            risk_percent = (risk_value / balance) * 100 if balance > 0 else 0
            
            # Risk-to-reward ratio (assumendo un target di 3x il rischio)
            # Qui si potrebbe utilizzare un target specifico se disponibile
            r2r = 3.0
            
            return {
                "risk_value": risk_value,
                "risk_percent": risk_percent,
                "risk_to_reward": r2r,
                "max_position_value": balance * (self.max_position_size_percent / 100)
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo del rischio: {str(e)}")
            return {
                "risk_value": 0.0,
                "risk_percent": 0.0,
                "risk_to_reward": 0.0,
                "max_position_value": 0.0
            }
    
    def calculate_optimal_position_size(self, symbol: str, entry_price: float, 
                                       stop_loss: float) -> float:
        """
        Calcola la dimensione ottimale della posizione in base al rischio
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            stop_loss: Livello di stop loss
            
        Returns:
            Dimensione ottimale della posizione
        """
        try:
            # Bilancio del wallet
            balance = self.get_wallet_balance()
            
            # Importo di rischio (percentuale del capitale)
            risk_amount = balance * (self.risk_per_trade_percent / 100)
            
            # Calcola la dimensione della posizione in base allo stop loss
            if entry_price <= 0 or stop_loss <= 0:
                return 0.0
                
            # Differenza percentuale tra entry e stop loss
            price_diff_percent = abs((entry_price - stop_loss) / entry_price)
            
            if price_diff_percent <= 0:
                return 0.0
                
            # Calcola la dimensione della posizione
            position_size = risk_amount / (entry_price * price_diff_percent)
            
            # Limita la dimensione al massimo consentito
            max_position_size = balance * (self.max_position_size_percent / 100) / entry_price
            position_size = min(position_size, max_position_size)
            
            # Considera anche la volatilità del simbolo
            volatility = self.get_symbol_volatility(symbol)
            if volatility > 0:
                # Riduce la dimensione per simboli più volatili
                volatility_factor = 1.0 / (1.0 + volatility)
                position_size *= volatility_factor
            
            self.logger.info(f"Dimensione posizione calcolata per {symbol}: {position_size}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo della dimensione posizione: {str(e)}")
            return 0.0
    
    def get_symbol_volatility(self, symbol: str, refresh: bool = False) -> float:
        """
        Calcola la volatilità di un simbolo
        
        Args:
            symbol: Simbolo della coppia
            refresh: Se forzare il ricalcolo della volatilità
            
        Returns:
            Valore di volatilità
        """
        # Verifica se i dati di volatilità sono aggiornati
        now = datetime.now()
        if (symbol in self.symbol_volatility and not refresh and
            symbol in self.volatility_update_time and
            now - self.volatility_update_time[symbol] < timedelta(hours=6)):
            return self.symbol_volatility[symbol]
            
        try:
            # Ottieni i dati storici
            data = self.exchange.get_klines(symbol, "1d", limit=30)
            
            if data.empty:
                return 0.0
                
            # Calcola i rendimenti giornalieri
            returns = data['close'].pct_change().dropna()
            
            # Calcola la volatilità come deviazione standard dei rendimenti
            volatility = returns.std()
            
            # Aggiorna i dati
            self.symbol_volatility[symbol] = volatility
            self.volatility_update_time[symbol] = now
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo della volatilità per {symbol}: {str(e)}")
            return 0.0
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]]) -> float:
        """
        Calcola il rischio complessivo del portafoglio
        
        Args:
            positions: Lista delle posizioni aperte
            
        Returns:
            Rischio complessivo in percentuale
        """
        try:
            # Bilancio del wallet
            balance = self.get_wallet_balance()
            
            if balance <= 0 or not positions:
                return 0.0
                
            # Calcola il rischio totale
            total_risk = 0.0
            
            for pos in positions:
                # Ottieni il rischio per ogni posizione
                symbol = pos.get("symbol")
                entry_price = pos.get("entry_price", 0.0)
                stop_loss = pos.get("stop_loss", 0.0)
                size = pos.get("size", 0.0)
                
                risk = self.calculate_risk(symbol, entry_price, stop_loss, size)
                total_risk += risk["risk_value"]
            
            # Rischio in percentuale del capitale
            portfolio_risk = (total_risk / balance) * 100
            
            # Aggiorna il rischio del portafoglio
            self.portfolio_risk = portfolio_risk
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo del rischio del portafoglio: {str(e)}")
            return 0.0
    
    def calculate_symbol_correlation(self, symbols: List[str] = None, 
                                    days: int = 30) -> pd.DataFrame:
        """
        Calcola la matrice di correlazione tra i simboli
        
        Args:
            symbols: Lista dei simboli (se None, usa quelli predefiniti)
            days: Numero di giorni per il calcolo
            
        Returns:
            DataFrame con la matrice di correlazione
        """
        now = datetime.now()
        
        # Usa i simboli predefiniti se non specificati
        if symbols is None:
            symbols = DEFAULT_TRADING_PAIRS
            
        # Verifica se i dati di correlazione sono aggiornati
        if (self.correlation_matrix is not None and
            self.correlation_update_time is not None and
            now - self.correlation_update_time < timedelta(days=1)):
            return self.correlation_matrix
            
        try:
            # Ottieni i dati storici per tutti i simboli
            price_data = {}
            
            for symbol in symbols:
                data = self.exchange.get_klines(symbol, "1d", limit=days)
                if not data.empty:
                    price_data[symbol] = data['close']
            
            if not price_data:
                return pd.DataFrame()
                
            # Crea un DataFrame con i prezzi di chiusura
            df = pd.DataFrame(price_data)
            
            # Calcola i rendimenti giornalieri
            returns = df.pct_change().dropna()
            
            # Calcola la matrice di correlazione
            correlation = returns.corr()
            
            # Aggiorna i dati
            self.correlation_matrix = correlation
            self.correlation_update_time = now
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo della correlazione: {str(e)}")
            return pd.DataFrame()
    
    def is_trade_allowed(self, symbol: str, entry_type: str, 
                        open_positions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Determina se un nuovo trade è consentito in base alle regole di rischio
        
        Args:
            symbol: Simbolo della coppia
            entry_type: Tipo di entrata (long o short)
            open_positions: Lista delle posizioni aperte
            
        Returns:
            Tupla (consentito, motivo)
        """
        try:
            # Verifica il numero massimo di trade contemporanei
            if len(open_positions) >= self.max_concurrent_trades:
                return False, f"Numero massimo di trade raggiunto ({self.max_concurrent_trades})"
                
            # Verifica se c'è già una posizione per questo simbolo
            for pos in open_positions:
                if pos.get("symbol") == symbol:
                    return False, f"Posizione già esistente per {symbol}"
            
            # Calcola il rischio del portafoglio
            portfolio_risk = self.calculate_portfolio_risk(open_positions)
            
            # Verifica se il rischio complessivo è troppo elevato
            if portfolio_risk > 5.0:  # 5% di rischio massimo
                return False, f"Rischio portafoglio troppo elevato: {portfolio_risk:.2f}%"
            
            # Verifica la correlazione con altre posizioni
            if len(open_positions) > 0:
                symbols = [p.get("symbol") for p in open_positions] + [symbol]
                correlation = self.calculate_symbol_correlation(symbols)
                
                if not correlation.empty:
                    # Verifica se il nuovo simbolo è altamente correlato con posizioni esistenti
                    for pos in open_positions:
                        pos_symbol = pos.get("symbol")
                        if pos_symbol in correlation.columns and symbol in correlation.columns:
                            corr_value = correlation.loc[symbol, pos_symbol]
                            
                            # Se la correlazione è alta e la direzione è la stessa, potrebbe aumentare il rischio
                            pos_entry_type = pos.get("entry_type", "")
                            if abs(corr_value) > 0.8 and pos_entry_type == entry_type:
                                return False, f"Alta correlazione con {pos_symbol} ({corr_value:.2f}) nello stesso verso"
            
            return True, "Trade consentito"
            
        except Exception as e:
            self.logger.error(f"Errore nella verifica del trade: {str(e)}")
            return False, f"Errore: {str(e)}"
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                           entry_type: str, atr_value: Optional[float] = None) -> float:
        """
        Calcola un livello di stop loss ottimale
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            entry_type: Tipo di entrata (long o short)
            atr_value: Valore ATR (se noto)
            
        Returns:
            Livello di stop loss ottimale
        """
        try:
            # Se l'ATR non è fornito, calcola la volatilità
            if atr_value is None:
                # Ottieni i dati storici
                data = self.exchange.get_klines(symbol, "1h", limit=24)
                
                if data.empty:
                    # Usa lo stop loss predefinito
                    if entry_type.lower() == "long":
                        return entry_price * 0.98  # -2%
                    else:
                        return entry_price * 1.02  # +2%
                
                # Calcola l'ATR
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift())
                low_close = abs(data['low'] - data['close'].shift())
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr_value = true_range.mean()
            
            # Calcola lo stop loss basato sull'ATR (2x ATR)
            atr_multiple = 2.0
            
            if entry_type.lower() == "long":
                stop_loss = entry_price - (atr_value * atr_multiple)
            else:
                stop_loss = entry_price + (atr_value * atr_multiple)
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo dello stop loss: {str(e)}")
            
            # In caso di errore, usa lo stop loss predefinito
            if entry_type.lower() == "long":
                return entry_price * 0.98  # -2%
            else:
                return entry_price * 1.02  # +2%
    
    def calculate_take_profits(self, symbol: str, entry_price: float, 
                              entry_type: str, stop_loss: float) -> List[float]:
        """
        Calcola livelli di take profit ottimali
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            entry_type: Tipo di entrata (long o short)
            stop_loss: Livello di stop loss
            
        Returns:
            Lista di livelli take profit
        """
        try:
            # Calcola il rischio in punti
            risk = abs(entry_price - stop_loss)
            
            # Calcola i take profit in base a multipli del rischio
            r_multiples = [1.5, 3.0, 5.0]  # Multipli di rischio per i TP
            take_profits = []
            
            for r in r_multiples:
                if entry_type.lower() == "long":
                    tp = entry_price + (risk * r)
                else:
                    tp = entry_price - (risk * r)
                    
                take_profits.append(tp)
            
            return take_profits
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo dei take profit: {str(e)}")
            
            # In caso di errore, usa take profit predefiniti
            if entry_type.lower() == "long":
                return [
                    entry_price * 1.015,  # +1.5%
                    entry_price * 1.03,   # +3%
                    entry_price * 1.05    # +5%
                ]
            else:
                return [
                    entry_price * 0.985,  # -1.5%
                    entry_price * 0.97,   # -3%
                    entry_price * 0.95    # -5%
                ]
