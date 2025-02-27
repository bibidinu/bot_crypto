"""
Modelli avanzati di Machine Learning specializzati per futures
"""
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.logger import get_logger
from config.settings import MODEL_SAVE_PATH

class FuturesPredictor:
    """
    Modello predittivo specializzato per futures con apprendimento multi-fattoriale
    """
    
    def __init__(
        self, 
        symbol: str, 
        model_type: str = 'gradient_boosting'
    ):
        """
        Inizializza il predittore per futures
        
        Args:
            symbol: Simbolo della coppia
            model_type: Tipo di modello da utilizzare
        """
        self.symbol = symbol
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        # Caratteristiche specifiche per futures
        self.futures_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'funding_rate', 'open_interest', 
            'rsi', 'macd', 'adx',
            'volatility', 'leverage'
        ]
        
        self.logger = get_logger(__name__)
    
    def prepare_data(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'close', 
        future_periods: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara i dati per l'addestramento
        
        Args:
            data: DataFrame con i dati
            target_column: Colonna target per la predizione
            future_periods: Periodi futuri da predire
        
        Returns:
            Tuple di features e target
        """
        # Preprocessing dei dati
        df = data.copy()
        
        # Calcolo del target come rendimento futuro
        df['target'] = df[target_column].shift(-future_periods) / df[target_column] - 1
        
        # Rimuovi righe con valori NaN
        df = df.dropna()
        
        # Seleziona le features
        features = df[self.futures_features]
        target = df['target']
        
        return features.values, target.values
    
    def build_model(self) -> Any:
        """
        Costruisce il modello predittivo
        
        Returns:
            Modello scikit-learn
        """
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Tipo di modello non supportato: {self.model_type}")
    
    def train(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Addestra il modello sui dati futures
        
        Args:
            data: DataFrame con i dati di mercato
            test_size: Proporzione di dati per test
        
        Returns:
            Dizionario con metriche di performance
        """
        try:
            # Preparazione dei dati
            X, y = self.prepare_data(data)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            # Costruzione del pipeline
            self.model = Pipeline([
                ('scaler', self.scaler),
                ('regressor', self.build_model())
            ])
            
            # Addestramento
            self.model.fit(X_train, y_train)
            
            # Predizioni
            y_pred = self.model.predict(X_test)
            
            # Calcolo delle metriche
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'cross_val_score': np.mean(cross_val_score(
                    self.model, X, y, cv=5, scoring='neg_mean_squared_error'
                ))
            }
            
            self.logger.info(f"Modello addestrato per {self.symbol}. Metriche: {metrics}")
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento: {e}")
            raise
    
    def predict(
        self, 
        data: pd.DataFrame, 
        future_periods: int = 1
    ) -> np.ndarray:
        """
        Effettua predizioni sui dati futures
        
        Args:
            data: DataFrame con i dati di mercato
            future_periods: Periodi futuri da predire
        
        Returns:
            Array di predizioni
        """
        try:
            # Preparazione dei dati
            X, _ = self.prepare_data(data, future_periods=future_periods)
            
            # Effettua predizione
            predictions = self.model.predict(X)
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Errore nella predizione: {e}")
            raise
    
    def save_model(self) -> str:
        """
        Salva il modello
        
        Returns:
            Percorso del file salvato
        """
        try:
            import os
            
            # Crea la directory se non esiste
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            
            # Genera il percorso del file
            filename = f"{self.symbol}_{self.model_type}_futures_model.joblib"
            filepath = os.path.join(MODEL_SAVE_PATH, filename)
            
            # Salva il modello
            joblib.dump(self.model, filepath)
            
            self.logger.info(f"Modello salvato in {filepath}")
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del modello: {e}")
            raise
    
    @classmethod
    def load_model(
        cls, 
        symbol: str, 
        model_type: str = 'gradient_boosting'
    ) -> 'FuturesPredictor':
        """
        Carica un modello salvato
        
        Args:
            symbol: Simbolo della coppia
            model_type: Tipo di modello
        
        Returns:
            Istanza del predittore
        """
        try:
            import os
            
            filename = f"{symbol}_{model_type}_futures_model.joblib"
            filepath = os.path.join(MODEL_SAVE_PATH, filename)
            
            # Crea l'istanza
            predictor = cls(symbol, model_type)
            predictor.model = joblib.load(filepath)
            
            return predictor
        
        except Exception as e:
            predictor.logger.error(f"Errore nel caricamento del modello: {e}")
            raise
