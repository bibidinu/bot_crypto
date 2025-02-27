"""
Strategia di trading basata sul machine learning
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

from api.exchange_interface import ExchangeInterface
from strategy.strategy_base import StrategyBase, Signal, SignalType, EntryType
from strategy.technical_strategy import TechnicalStrategy
from utils.logger import get_logger
from config.settings import (
    STOP_LOSS_PERCENT, TAKE_PROFIT_LEVELS, MODEL_SAVE_PATH, 
    FEATURE_WINDOW, LEARNING_RATE, BATCH_SIZE, EPOCHS
)

logger = get_logger(__name__)

class MLStrategy(StrategyBase):
    """Strategia di trading basata sul machine learning"""
    
    def __init__(self, exchange: ExchangeInterface, 
                 name: str = "MLStrategy", 
                 tech_strategy: Optional[TechnicalStrategy] = None):
        """
        Inizializza la strategia basata su ML
        
        Args:
            exchange: Istanza dell'exchange
            name: Nome della strategia
            tech_strategy: Strategia tecnica per calcolare features
        """
        super().__init__(exchange, name)
        self.logger = get_logger(f"{__name__}.{name}")
        
        # Crea una strategia tecnica se non fornita
        self.tech_strategy = tech_strategy or TechnicalStrategy(exchange)
        
        # Creare cartella per salvare modelli se non esiste
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # Dizionario per memorizzare i modelli per simbolo
        self.models = {}
        
        # Parametri per l'apprendimento
        self.feature_window = FEATURE_WINDOW
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        
        # Traccia i trade predetti per l'apprendimento di rinforzo
        self.trade_history = []
        
        self.logger.info(f"Strategia ML {self.name} inizializzata")
    
def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara le features per il modello di ML
    
    Args:
        data: DataFrame con dati OHLCV
        
    Returns:
        DataFrame con features
    """
    if len(data) < self.feature_window:
        self.logger.warning(f"Dati insufficienti: {len(data)} righe < {self.feature_window} richieste")
        return pd.DataFrame()
        
    # Copia i dati per non modificare l'originale
    df = data.copy()
    
    try:
        # Aggiungi indicatori tecnici
        df = self.tech_strategy.add_indicators(df)
        
        # Gestione valori NaN iniziali dovuti al calcolo degli indicatori
        df = df.iloc[self.feature_window:]  # Rimuovi le prime n righe invece di dropna()
        
        # Aggiungi feature di variazione percentuale con gestione degli infiniti
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[f"{col}_pct_change"] = df[col].pct_change()
            # Sostituisci infinito con il massimo valore float64
            df[f"{col}_pct_change"] = df[f"{col}_pct_change"].replace([np.inf, -np.inf], 0)
        
        # Aggiungi feature di volatilità con controllo divisione per zero
        df['volatility'] = np.where(
            df['low'] > 0,
            (df['high'] - df['low']) / df['low'],
            0
        )
        
        # Aggiungi feature di range con controllo
        df['day_range'] = df['high'] - df['low']
        df['day_range_pct'] = np.where(
            df['close'] > 0,
            df['day_range'] / df['close'],
            0
        )
        
        # Calcola feature di momentum con gestione degli infiniti
        for period in [1, 3, 5, 10]:
            df[f'momentum_{period}d'] = df['close'].pct_change(period)
            df[f'momentum_{period}d'] = df[f'momentum_{period}d'].replace([np.inf, -np.inf], 0)
        
        # Feature di rapporto volume/prezzo con controllo divisione per zero
        df['volume_price_ratio'] = np.where(
            df['close'] > 0,
            df['volume'] / df['close'],
            0
        )
        
        # Feature ciclica per tempo (ora del giorno, giorno della settimana)
        if 'datetime' in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
            
            # Trasforma queste feature in coordinate circolari
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Riempi i NaN rimanenti con 0
        df = df.fillna(0)
        
        # Seleziona solo colonne numeriche
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Rimuovi colonne non rilevanti
        exclude_cols = ['timestamp']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Clip dei valori per evitare outlier estremi
        df[feature_cols] = df[feature_cols].clip(-1e6, 1e6)
        
        if df.empty:
            self.logger.warning("DataFrame vuoto dopo il preprocessing")
            return pd.DataFrame()
            
        return df[feature_cols]
        
    except Exception as e:
        self.logger.error(f"Errore nella preparazione delle features: {str(e)}")
        return pd.DataFrame()
    
    def create_target(self, data: pd.DataFrame, look_ahead: int = 12) -> pd.DataFrame:
        """
        Crea la variabile target per l'addestramento
        
        Args:
            data: DataFrame con i dati di mercato
            look_ahead: Numero di periodi futuri per calcolare il rendimento
            
        Returns:
            DataFrame con target
        """
        df = data.copy()
        
        # Calcola il rendimento futuro
        df['future_return'] = df['close'].shift(-look_ahead) / df['close'] - 1
        
        # Crea target categorico
        threshold = 0.005  # 0.5% come soglia per segnali significativi
        
        df['target'] = 0  # HOLD
        df.loc[df['future_return'] > threshold, 'target'] = 1  # BUY
        df.loc[df['future_return'] < -threshold, 'target'] = -1  # SELL
        
        return df
    
    def train_model(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Addestra il modello per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di mercato
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        
        self.logger.info(f"Addestramento modello per {symbol}")
        
        try:
            # Prepara features e target
            df = self.prepare_features(data)
            df = self.create_target(df)
            
            # Rimuovi righe con valori NaN
            df = df.dropna()
            
            if len(df) < 100:
                self.logger.warning(f"Dati insufficienti per {symbol}: {len(df)} righe")
                return
            
            # Seleziona feature e target
            X = df.drop(['target', 'future_return'], axis=1, errors='ignore')
            y = df['target']
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Crea pipeline con scaler e modello
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ])
            
            # Addestra il modello
            model.fit(X_train, y_train)
            
            # Valuta il modello
            accuracy = model.score(X_test, y_test)
            self.logger.info(f"Accuratezza modello per {symbol}: {accuracy:.4f}")
            
            # Salva il modello
            self.models[symbol] = model
            
            # Salva il modello su disco
            model_path = os.path.join(MODEL_SAVE_PATH, f"{symbol.replace('/', '_')}.joblib")
            joblib.dump(model, model_path)
            self.logger.info(f"Modello salvato in {model_path}")
            
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento del modello per {symbol}: {str(e)}")
            raise
    
    def load_model(self, symbol: str) -> Optional[Any]:
        """
        Carica il modello per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Modello caricato o None se non esiste
        """
        try:
            model_path = os.path.join(MODEL_SAVE_PATH, f"{symbol.replace('/', '_')}.joblib")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.models[symbol] = model
                self.logger.info(f"Modello caricato da {model_path}")
                return model
            else:
                self.logger.info(f"Nessun modello trovato per {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Errore nel caricamento del modello per {symbol}: {str(e)}")
            return None
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Effettua una previsione utilizzando il modello
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di mercato
            
        Returns:
            Tupla (predizione, probabilità)
        """
        try:
            # Carica il modello se non è già in memoria
            if symbol not in self.models:
                model = self.load_model(symbol)
                if model is None:
                    self.logger.warning(f"Modello non disponibile per {symbol}, addestramento necessario")
                    return 0, 0.0
            else:
                model = self.models[symbol]
            
            # Prepara le features
            features = self.prepare_features(data)
            
            # Rimuovi righe con valori NaN
            features = features.dropna()
            
            if features.empty:
                self.logger.warning(f"Features vuote per {symbol}")
                return 0, 0.0
            
            # Prendi l'ultima riga per la previsione
            latest_features = features.iloc[-1:].copy()
            
            # Effettua la previsione
            prediction = model.predict(latest_features)[0]
            
            # Ottieni la probabilità
            probabilities = model.predict_proba(latest_features)[0]
            
            # La probabilità corrisponde alla classe predetta
            if prediction == -1:
                prob = probabilities[0]  # SELL
            elif prediction == 1:
                prob = probabilities[2]  # BUY
            else:
                prob = probabilities[1]  # HOLD
            
            return int(prediction), float(prob)
            
        except Exception as e:
            self.logger.error(f"Errore nella previsione per {symbol}: {str(e)}")
            return 0, 0.0
    
    def update_model(self, symbol: str, signal: Signal, 
                    actual_profit: float, success: bool) -> None:
        """
        Aggiorna il modello con i risultati del trade
        
        Args:
            symbol: Simbolo della coppia
            signal: Segnale generato
            actual_profit: Profitto/perdita effettivo
            success: Se il trade è stato vincente
        """
        # Aggiungi il trade alla cronologia
        trade_result = {
            "symbol": symbol,
            "signal_type": signal.signal_type.value,
            "entry_price": signal.price,
            "profit": actual_profit,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.trade_history.append(trade_result)
        
        # Log del risultato
        self.logger.info(f"Trade completato per {symbol}: profit={actual_profit:.2%}, success={success}")
        
        # TODO: Implementare l'aggiornamento del modello con reinforcement learning
        # In una versione più completa, qui si potrebbe implementare l'aggiornamento
        # del modello utilizzando tecniche di reinforcement learning
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Genera un segnale di trading per il simbolo specificato
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di mercato
            
        Returns:
            Un oggetto Signal con il suggerimento
        """
        # Ottieni anche un segnale dalla strategia tecnica
        tech_signal = self.tech_strategy.generate_signal(symbol, data)
        
        try:
            # Effettua la previsione con il modello ML
            prediction, probability = self.predict(symbol, data)
            
            # Se il modello non è disponibile, usa solo il segnale tecnico
            if probability == 0.0:
                self.logger.info(f"Usando solo segnale tecnico per {symbol}")
                return tech_signal
            
            # Ultimo prezzo
            last_price = data.iloc[-1]['close']
            
            # Calcola una forza del segnale combinata
            # Combina la probabilità ML con la forza del segnale tecnico
            combined_strength = 0.7 * probability + 0.3 * tech_signal.strength
            
            # Implementa una soglia per filtrare i segnali deboli
            threshold = 0.6
            
            if prediction == 1 and combined_strength > threshold:
                # Segnale di acquisto
                signal_type = SignalType.BUY
                entry_type = EntryType.LONG
                
                # Calcola stop loss e take profit
                stop_loss = last_price * (1 - STOP_LOSS_PERCENT / 100)
                
                take_profits = []
                for tp in TAKE_PROFIT_LEVELS:
                    take_profit_price = last_price * (1 + tp["percent"] / 100)
                    take_profits.append(take_profit_price)
                
                reason = (f"ML Prediction: BUY (prob={probability:.2f}), "
                          f"Tech: {tech_signal.reason}")
                
                return Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_type=entry_type,
                    price=last_price,
                    strength=combined_strength,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    reason=reason
                )
                
            elif prediction == -1 and combined_strength > threshold:
                # Segnale di vendita
                signal_type = SignalType.SELL
                entry_type = EntryType.SHORT
                
                # Calcola stop loss e take profit per short
                stop_loss = last_price * (1 + STOP_LOSS_PERCENT / 100)
                
                take_profits = []
                for tp in TAKE_PROFIT_LEVELS:
                    take_profit_price = last_price * (1 - tp["percent"] / 100)
                    take_profits.append(take_profit_price)
                
                reason = (f"ML Prediction: SELL (prob={probability:.2f}), "
                          f"Tech: {tech_signal.reason}")
                
                return Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_type=entry_type,
                    price=last_price,
                    strength=combined_strength,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    reason=reason
                )
            else:
                # Segnale di mantenimento (HOLD) o segnale troppo debole
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    price=last_price,
                    strength=combined_strength,
                    reason=f"ML Prediction: {prediction} (prob={probability:.2f}), segnale debole o neutrale"
                )
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione del segnale ML per {symbol}: {str(e)}")
            # In caso di errore, usa il segnale tecnico come fallback
            return tech_signal
    
    def analyze_multiple_timeframes(self, symbol: str) -> Signal:
        """
        Analizza il simbolo su più timeframe e combina i segnali
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Segnale combinato
        """
        from config.settings import AVAILABLE_TIMEFRAMES
        
        self.logger.info(f"Analisi multi-timeframe per {symbol}")
        
        original_timeframe = self.timeframe
        signals = []
        
        try:
            # Analizza ogni timeframe
            for tf in AVAILABLE_TIMEFRAMES:
                self.timeframe = tf
                data = self.get_market_data(symbol)
                signal = self.generate_signal(symbol, data)
                signals.append(signal)
                self.logger.debug(f"Segnale per {symbol} su {tf}: {signal}")
                
            # Ripristina il timeframe originale
            self.timeframe = original_timeframe
            
            # Calcola un segnale combinato
            # Dai più peso ai timeframe più lunghi
            weights = {
                "1m": 0.05,
                "5m": 0.1,
                "15m": 0.15,
                "30m": 0.15,
                "1h": 0.2,
                "4h": 0.2,
                "1d": 0.15
            }
            
            # Pesa i segnali
            buy_strength = 0
            sell_strength = 0
            
            for i, signal in enumerate(signals):
                tf = AVAILABLE_TIMEFRAMES[i]
                weight = weights.get(tf, 0.1)
                
                if signal.signal_type == SignalType.BUY:
                    buy_strength += signal.strength * weight
                elif signal.signal_type == SignalType.SELL:
                    sell_strength += signal.strength * weight
            
            # Ultimo prezzo dal segnale più recente
            last_price = signals[0].price
            
            # Determina il tipo di segnale finale
            if buy_strength > 0.3 and buy_strength > sell_strength:
                # Segnale di acquisto combinato
                signal_type = SignalType.BUY
                entry_type = EntryType.LONG
                strength = buy_strength
                
                # Calcola stop loss e take profit
                stop_loss = last_price * (1 - STOP_LOSS_PERCENT / 100)
                
                take_profits = []
                for tp in TAKE_PROFIT_LEVELS:
                    take_profit_price = last_price * (1 + tp["percent"] / 100)
                    take_profits.append(take_profit_price)
                
                reason = f"Multi-timeframe BUY (strength={strength:.2f})"
                
                return Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_type=entry_type,
                    price=last_price,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    reason=reason
                )
                
            elif sell_strength > 0.3 and sell_strength > buy_strength:
                # Segnale di vendita combinato
                signal_type = SignalType.SELL
                entry_type = EntryType.SHORT
                strength = sell_strength
                
                # Calcola stop loss e take profit per short
                stop_loss = last_price * (1 + STOP_LOSS_PERCENT / 100)
                
                take_profits = []
                for tp in TAKE_PROFIT_LEVELS:
                    take_profit_price = last_price * (1 - tp["percent"] / 100)
                    take_profits.append(take_profit_price)
                
                reason = f"Multi-timeframe SELL (strength={strength:.2f})"
                
                return Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_type=entry_type,
                    price=last_price,
                    strength=strength,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    reason=reason
                )
            else:
                # Segnale di mantenimento (HOLD)
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    price=last_price,
                    strength=max(buy_strength, sell_strength),
                    reason=f"Multi-timeframe HOLD (buy={buy_strength:.2f}, sell={sell_strength:.2f})"
                )
                
        except Exception as e:
            self.logger.error(f"Errore nell'analisi multi-timeframe per {symbol}: {str(e)}")
            # In caso di errore, torna al timeframe originale e utilizza un singolo segnale
            self.timeframe = original_timeframe
            data = self.get_market_data(symbol)
            return self.generate_signal(symbol, data)
