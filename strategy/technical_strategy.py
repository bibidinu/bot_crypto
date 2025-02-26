"""
Strategia di trading basata su indicatori tecnici
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import ta

from api.exchange_interface import ExchangeInterface
from strategy.strategy_base import StrategyBase, Signal, SignalType, EntryType
from utils.logger import get_logger
from config.settings import STOP_LOSS_PERCENT, TAKE_PROFIT_LEVELS

logger = get_logger(__name__)

class TechnicalStrategy(StrategyBase):
    """Strategia di trading basata su indicatori tecnici"""
    
    def __init__(self, exchange: ExchangeInterface, name: str = "TechnicalStrategy"):
        """
        Inizializza la strategia tecnica
        
        Args:
            exchange: Istanza dell'exchange
            name: Nome della strategia
        """
        super().__init__(exchange, name)
        self.logger = get_logger(f"{__name__}.{name}")
        
        # Parametri indicatori
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        self.ema_short = 20
        self.ema_medium = 50
        self.ema_long = 200
        
        self.atr_period = 14
        self.atr_multiplier = 2.0
        
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        
        self.logger.info(f"Strategia tecnica {self.name} inizializzata")
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge indicatori tecnici al DataFrame
        
        Args:
            df: DataFrame con i dati OHLCV
            
        Returns:
            DataFrame con indicatori aggiunti
        """
        # Assicura che il DataFrame abbia le colonne necessarie
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"DataFrame mancante di colonne richieste {required_columns}")
            return df
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        
        # MACD
        macd = ta.trend.MACD(
            df['close'], 
            window_slow=self.macd_slow, 
            window_fast=self.macd_fast, 
            window_sign=self.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # EMA
        df['ema_short'] = ta.trend.EMAIndicator(df['close'], window=self.ema_short).ema_indicator()
        df['ema_medium'] = ta.trend.EMAIndicator(df['close'], window=self.ema_medium).ema_indicator()
        df['ema_long'] = ta.trend.EMAIndicator(df['close'], window=self.ema_long).ema_indicator()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=self.atr_period
        ).average_true_range()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            df['close'], 
            window=self.bollinger_period, 
            window_dev=self.bollinger_std
        )
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['high'],
            low=df['low'],
            window1=9,
            window2=26,
            window3=52
        )
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # OBV (On Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        return df
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Genera un segnale di trading per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di mercato
            
        Returns:
            Un oggetto Signal con il suggerimento
        """
        df = self.add_indicators(data.copy())
        
        # Evita errori con dati insufficienti
        if len(df) < self.ema_long:
            return Signal(symbol, SignalType.HOLD, price=0, strength=0, 
                         reason="Dati insufficienti per l'analisi")
        
        # Ultimi valori
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Calcolo dei punteggi per i vari indicatori
        scores = {}
        reasons = []
        
        # Ultimo prezzo
        last_price = current['close']
        
        # --- RSI ---
        rsi_score = 0
        if current['rsi'] < self.rsi_oversold:
            rsi_score = 1  # Segnale di acquisto
            reasons.append(f"RSI oversold ({current['rsi']:.2f})")
        elif current['rsi'] > self.rsi_overbought:
            rsi_score = -1  # Segnale di vendita
            reasons.append(f"RSI overbought ({current['rsi']:.2f})")
        scores['rsi'] = rsi_score
        
        # --- MACD ---
        macd_score = 0
        if current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            macd_score = 1  # Segnale di acquisto (MACD incrocia al rialzo la signal line)
            reasons.append("MACD incrocio bullish")
        elif current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            macd_score = -1  # Segnale di vendita (MACD incrocia al ribasso la signal line)
            reasons.append("MACD incrocio bearish")
        scores['macd'] = macd_score
        
        # --- EMA Trend ---
        ema_score = 0
        if (current['ema_short'] > current['ema_medium'] and 
            current['ema_medium'] > current['ema_long']):
            ema_score = 1  # Trend rialzista
            reasons.append("Trend EMA bullish")
        elif (current['ema_short'] < current['ema_medium'] and 
              current['ema_medium'] < current['ema_long']):
            ema_score = -1  # Trend ribassista
            reasons.append("Trend EMA bearish")
        scores['ema'] = ema_score
        
        # --- Bollinger Bands ---
        bb_score = 0
        if current['close'] < current['bollinger_lband']:
            bb_score = 1  # Prezzo sotto la banda inferiore (potenziale rimbalzo)
            reasons.append("Prezzo sotto Bollinger lower band")
        elif current['close'] > current['bollinger_hband']:
            bb_score = -1  # Prezzo sopra la banda superiore (potenziale inversione)
            reasons.append("Prezzo sopra Bollinger upper band")
        scores['bollinger'] = bb_score
        
        # --- Stochastic ---
        stoch_score = 0
        if current['stoch_k'] < 20 and current['stoch_d'] < 20 and current['stoch_k'] > current['stoch_d']:
            stoch_score = 1  # Stochastic oversold con crossover bullish
            reasons.append("Stochastic oversold con crossover bullish")
        elif current['stoch_k'] > 80 and current['stoch_d'] > 80 and current['stoch_k'] < current['stoch_d']:
            stoch_score = -1  # Stochastic overbought con crossover bearish
            reasons.append("Stochastic overbought con crossover bearish")
        scores['stochastic'] = stoch_score
        
        # --- ADX (forza del trend) ---
        adx_score = 0
        if current['adx'] > 25:
            if current['adx_pos'] > current['adx_neg']:
                adx_score = 0.5  # Trend forte positivo
                reasons.append(f"ADX forte trend bullish ({current['adx']:.2f})")
            elif current['adx_neg'] > current['adx_pos']:
                adx_score = -0.5  # Trend forte negativo
                reasons.append(f"ADX forte trend bearish ({current['adx']:.2f})")
        scores['adx'] = adx_score
        
        # --- Ichimoku Cloud ---
        ichimoku_score = 0
        if (current['close'] > current['ichimoku_a'] and 
            current['close'] > current['ichimoku_b']):
            ichimoku_score = 0.5  # Prezzo sopra la nuvola (bullish)
            reasons.append("Prezzo sopra Ichimoku cloud (bullish)")
        elif (current['close'] < current['ichimoku_a'] and 
              current['close'] < current['ichimoku_b']):
            ichimoku_score = -0.5  # Prezzo sotto la nuvola (bearish)
            reasons.append("Prezzo sotto Ichimoku cloud (bearish)")
        scores['ichimoku'] = ichimoku_score
        
        # Calcola il punteggio totale e normalizzalo tra -1 e 1
        total_score = sum(scores.values())
        max_possible_score = len(scores)
        normalized_score = total_score / max_possible_score
        
        # Converti il punteggio normalizzato in una forza tra 0 e 1
        signal_strength = abs(normalized_score)
        
        # Determina il tipo di segnale
        if normalized_score > 0.3:  # Soglia per segnale di acquisto
            signal_type = SignalType.BUY
            entry_type = EntryType.LONG
            
            # Calcola stop loss e take profit
            stop_loss = last_price * (1 - STOP_LOSS_PERCENT / 100)
            
            take_profits = []
            for tp in TAKE_PROFIT_LEVELS:
                take_profit_price = last_price * (1 + tp["percent"] / 100)
                take_profits.append(take_profit_price)
            
            reason = ", ".join(reasons) + f" (Score: {normalized_score:.2f})"
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_type=entry_type,
                price=last_price,
                strength=signal_strength,
                stop_loss=stop_loss,
                take_profits=take_profits,
                reason=reason,
                indicators=dict(scores)
            )
            
        elif normalized_score < -0.3:  # Soglia per segnale di vendita
            signal_type = SignalType.SELL
            entry_type = EntryType.SHORT
            
            # Calcola stop loss e take profit per short
            stop_loss = last_price * (1 + STOP_LOSS_PERCENT / 100)
            
            take_profits = []
            for tp in TAKE_PROFIT_LEVELS:
                take_profit_price = last_price * (1 - tp["percent"] / 100)
                take_profits.append(take_profit_price)
            
            reason = ", ".join(reasons) + f" (Score: {normalized_score:.2f})"
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_type=entry_type,
                price=last_price,
                strength=signal_strength,
                stop_loss=stop_loss,
                take_profits=take_profits,
                reason=reason,
                indicators=dict(scores)
            )
        else:
            # Segnale di mantenimento (HOLD)
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                price=last_price,
                strength=0,
                reason=f"Segnali misti o deboli. Score: {normalized_score:.2f}"
            )
    
    def optimize_parameters(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Ottimizza i parametri della strategia utilizzando backtest
        
        Args:
            symbol: Simbolo della coppia
            start_date: Data di inizio per il backtest
            end_date: Data di fine per il backtest
            
        Returns:
            Parametri ottimizzati
        """
        # Implementazione dell'ottimizzazione dei parametri
        # Questa è una versione semplificata, in un'implementazione reale
        # si eseguirebbero più backtest con diverse combinazioni di parametri
        
        self.logger.info(f"Ottimizzazione parametri per {symbol}")
        
        # I parametri ottimizzati (esempio)
        optimized_params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ema_short": 20,
            "ema_medium": 50,
            "ema_long": 200
        }
        
        # Imposta i parametri ottimizzati
        for param, value in optimized_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        self.logger.info(f"Parametri ottimizzati per {symbol}: {optimized_params}")
        
        return optimized_params
