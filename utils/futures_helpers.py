"""
Funzioni di utilità specializzate per il trading di futures
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

class FuturesHelpers:
    """
    Classe di utilità per calcoli e helper specifici per futures
    """
    
    @staticmethod
    def calculate_position_value(
        entry_price: float, 
        quantity: float, 
        leverage: float
    ) -> float:
        """
        Calcola il valore della posizione con leva
        
        Args:
            entry_price: Prezzo di entrata
            quantity: Quantità del contratto
            leverage: Leva utilizzata
        
        Returns:
            Valore totale della posizione
        """
        return entry_price * quantity * leverage
    
    @staticmethod
    def calculate_funding_impact(
        position_value: float, 
        funding_rate: float, 
        holding_periods: int = 1
    ) -> Dict[str, float]:
        """
        Calcola l'impatto del funding rate su una posizione
        
        Args:
            position_value: Valore della posizione
            funding_rate: Tasso di funding
            holding_periods: Numero di periodi di funding
        
        Returns:
            Dizionario con dettagli dell'impatto del funding
        """
        # Calcolo dell'impatto del funding
        funding_impact_per_period = position_value * funding_rate
        total_funding_impact = funding_impact_per_period * holding_periods
        
        return {
            "funding_impact_per_period": funding_impact_per_period,
            "total_funding_impact": total_funding_impact,
            "impact_percentage": (total_funding_impact / position_value) * 100
        }
    
    @staticmethod
    def calculate_liquidation_safety_margin(
        entry_price: float, 
        leverage: float, 
        maintenance_margin: float = 0.005
    ) -> float:
        """
        Calcola il margine di sicurezza prima della liquidazione
        
        Args:
            entry_price: Prezzo di entrata
            leverage: Leva utilizzata
            maintenance_margin: Margine di mantenimento
        
        Returns:
            Prezzo di liquidazione stimato
        """
        # Formula base per il prezzo di liquidazione
        liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin)
        
        return max(0, liquidation_price)
    
    @staticmethod
    def calculate_risk_of_ruin(
        win_rate: float, 
        risk_per_trade: float, 
        num_trades: int
    ) -> float:
        """
        Calcola la probabilità di rovina basata su Kelly Criterion
        
        Args:
            win_rate: Percentuale di trade vincenti
            risk_per_trade: Percentuale di rischio per trade
            num_trades: Numero di trade
        
        Returns:
            Probabilità di rovina
        """
        # Calcolo della probabilità di rovina
        # Basato su Kelly Criterion e sua variante
        if win_rate <= 0 or risk_per_trade <= 0:
            return 1.0
        
        # Probabilità di perdita per trade
        p_loss = 1 - win_rate
        
        # Calcolo del capitale residuo
        def capital_remaining(initial_capital, trade_count):
            remaining = initial_capital
            for _ in range(trade_count):
                # Simulazione probabilistica
                if np.random.random() < p_loss:
                    remaining *= (1 - risk_per_trade)
                else:
                    remaining *= (1 + risk_per_trade)
                
                if remaining <= 0:
                    return 0
            return remaining
        
        # Simulazione Monte Carlo
        simulations = 10000
        ruin_count = 0
        
        for _ in range(simulations):
            final_capital = capital_remaining(100.0, num_trades)
            if final_capital <= 0:
                ruin_count += 1
        
        return ruin_count / simulations
    
    @staticmethod
    def analyze_funding_opportunity(
        positive_funding_rate: float, 
        negative_funding_rate: float,
        entry_side: str = 'long'
    ) -> Dict[str, Any]:
        """
        Analizza le opportunità di arbitraggio sul funding rate
        
        Args:
            positive_funding_rate: Tasso di funding positivo
            negative_funding_rate: Tasso di funding negativo
            entry_side: Lato di entrata ('long' o 'short')
        
        Returns:
            Dizionario con dettagli dell'opportunità di arbitraggio
        """
        # Calcolo opportunità di arbitraggio
        if entry_side == 'long':
            # Long paga quando positivo, guadagna quando negativo
            payoff = -positive_funding_rate if positive_funding_rate > 0 else abs(negative_funding_rate)
        else:  # short
            # Short guadagna quando positivo, paga quando negativo
            payoff = positive_funding_rate if positive_funding_rate > 0 else -abs(negative_funding_rate)
        
        # Calcolo rendimento annualizzato
        annual_return = payoff * 3 * 365 / 100  # 3 volte al giorno, converti in decimale
        
        return {
            "entry_side": entry_side,
            "funding_payoff": payoff,
            "annual_return_percent": annual_return * 100,
            "is_opportunity": abs(payoff) > 0.001  # Soglia minima
        }
    
    @staticmethod
    def optimize_trade_timing(
        historical_data: pd.DataFrame, 
        feature_columns: List[str],
        target_column: str = 'close'
    ) -> Dict[str, Any]:
        """
        Ottimizza la tempistica dei trade basandosi su analisi storiche
        
        Args:
            historical_data: DataFrame con dati storici
            feature_columns: Colonne da utilizzare come features
            target_column: Colonna target per le predizioni
        
        Returns:
            Dizionario con suggeriementi di timing
        """
        # Preparazione dei dati
        data = historical_data.copy()
        
        # Calcolo dei rendimenti
        data['returns'] = data[target_column].pct_change()
        
        # Calcolo volatilità
        volatility = data['returns'].std()
        
        # Identificazione dei migliori periodi di trading
        best_hours = []
        if 'datetime' in data.columns:
            data['hour'] = data['datetime'].dt.hour
            hourly_returns = data.groupby('hour')['returns'].mean()
            best_hours = hourly_returns.nlargest(3).index.tolist()
        
        return {
            "optimal_volatility": volatility,
            "best_trading_hours": best_hours,
            "average_return": data['returns'].mean(),
            "max_drawdown": (data['returns'].cummax() - data['returns']).max()
        }
