"""
Modelli di machine learning per il trading
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

from utils.logger import get_logger
from config.settings import MODEL_SAVE_PATH

logger = get_logger(__name__)

class MLModel:
    """Classe base per i modelli di machine learning"""
    
    def __init__(self, model_name: str, model_type: str):
        """
        Inizializza il modello ML
        
        Args:
            model_name: Nome del modello
            model_type: Tipo di modello
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.trained = False
        self.last_training_time = None
        self.performance_metrics = {}
        
        # Crea la directory per i modelli se non esiste
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        self.logger = get_logger(f"{__name__}.{model_name}")
        
        self.logger.info(f"Modello ML {model_name} inizializzato (tipo: {model_type})")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preelabora i dati per l'addestramento o la previsione
        
        Args:
            data: DataFrame con i dati
            
        Returns:
            Tuple con features e target (se presente)
        """
        # Verifica che il dataframe non sia vuoto
        if data.empty:
            raise ValueError("Il dataframe fornito è vuoto")
        
        # Rimuovi le righe con valori mancanti
        data_clean = data.dropna()
        
        if data_clean.empty:
            raise ValueError("Il dataframe fornito contiene solo valori mancanti")
        
        # Separa features e target (se 'target' è presente)
        if 'target' in data_clean.columns:
            X = data_clean.drop('target', axis=1)
            y = data_clean['target'].values
        else:
            X = data_clean
            y = None
        
        # Memorizza i nomi delle features
        self.feature_names = X.columns.tolist()
        
        # Scala le features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def train(self, data: pd.DataFrame, target_column: str = 'target', 
             test_size: float = 0.2, optimize: bool = False) -> Dict[str, Any]:
        """
        Addestra il modello sui dati forniti
        
        Args:
            data: DataFrame con i dati di training
            target_column: Nome della colonna target
            test_size: Dimensione del set di test
            optimize: Se ottimizzare gli iperparametri
            
        Returns:
            Metriche di performance del modello
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Inizio addestramento del modello {self.model_name}")
            
            # Rinomina la colonna target se è diversa da 'target'
            if target_column in data.columns and target_column != 'target':
                data = data.copy()
                data['target'] = data[target_column]
            
            # Preelabora i dati
            X, y = self.preprocess_data(data)
            
            if y is None:
                raise ValueError("La colonna target non è presente nei dati")
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Ottimizza gli iperparametri se richiesto
            if optimize:
                self._optimize_hyperparameters(X_train, y_train)
            
            # Addestra il modello
            self.model.fit(X_train, y_train)
            
            # Valuta il modello
            y_pred = self.model.predict(X_test)
            
            # Calcola le metriche
            self.performance_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            self.trained = True
            self.last_training_time = datetime.now()
            
            training_time = time.time() - start_time
            self.logger.info(f"Addestramento completato in {training_time:.2f} secondi. "
                            f"Accuracy: {self.performance_metrics['accuracy']:.4f}")
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento del modello: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Genera previsioni con il modello addestrato
        
        Args:
            data: DataFrame con i dati di input
            
        Returns:
            Array di previsioni
        """
        if not self.trained or self.model is None:
            raise ValueError("Il modello non è stato addestrato")
        
        # Preelabora i dati
        X, _ = self.preprocess_data(data)
        
        # Genera le previsioni
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Genera probabilità di previsione con il modello addestrato
        
        Args:
            data: DataFrame con i dati di input
            
        Returns:
            Array di probabilità
        """
        if not self.trained or self.model is None:
            raise ValueError("Il modello non è stato addestrato")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Il modello non supporta previsioni probabilistiche")
        
        # Preelabora i dati
        X, _ = self.preprocess_data(data)
        
        # Genera le probabilità
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Salva il modello su disco
        
        Args:
            filepath: Percorso per il salvataggio (opzionale)
            
        Returns:
            Percorso del file salvato
        """
        if not self.trained or self.model is None:
            raise ValueError("Il modello non è stato addestrato")
        
        if filepath is None:
            # Genera un nome di file basato sul nome del modello e la data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODEL_SAVE_PATH, f"{self.model_name}_{timestamp}.joblib")
        
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Preparare i dati da salvare
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'performance_metrics': self.performance_metrics,
            'last_training_time': self.last_training_time
        }
        
        # Salva il modello
        joblib.dump(model_data, filepath)
        
        self.logger.info(f"Modello salvato in {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLModel':
        """
        Carica un modello da disco
        
        Args:
            filepath: Percorso del file
            
        Returns:
            Istanza di MLModel
        """
        try:
            # Carica i dati del modello
            model_data = joblib.load(filepath)
            
            # Crea una nuova istanza
            instance = cls(model_data['model_name'], model_data['model_type'])
            
            # Ripristina i dati del modello
            instance.model = model_data['model']
            instance.scaler = model_data['scaler']
            instance.feature_names = model_data['feature_names']
            instance.performance_metrics = model_data['performance_metrics']
            instance.last_training_time = model_data['last_training_time']
            instance.trained = True
            
            instance.logger.info(f"Modello caricato da {filepath}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello da {filepath}: {str(e)}")
            raise
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ottimizza gli iperparametri del modello
        
        Args:
            X_train: Dati di training
            y_train: Target di training
        """
        # Override in classi derivate
        pass
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Ottiene l'importanza delle features
        
        Returns:
            DataFrame con l'importanza delle features
        """
        if not self.trained or self.model is None:
            raise ValueError("Il modello non è stato addestrato")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Il modello non supporta l'importanza delle features")
        
        # Ottieni l'importanza
        importances = self.model.feature_importances_
        
        # Crea un DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Ordina per importanza
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df

class RandomForestModel(MLModel):
    """Modello RandomForest per il trading"""
    
    def __init__(self, model_name: str, n_estimators: int = 100, 
                max_depth: int = 10, random_state: int = 42):
        """
        Inizializza il modello RandomForest
        
        Args:
            model_name: Nome del modello
            n_estimators: Numero di alberi nella foresta
            max_depth: Profondità massima degli alberi
            random_state: Seed per la riproducibilità
        """
        super().__init__(model_name, "RandomForest")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Usa tutti i core disponibili
        )
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ottimizza gli iperparametri del modello RandomForest
        
        Args:
            X_train: Dati di training
            y_train: Target di training
        """
        self.logger.info("Inizio ottimizzazione iperparametri per RandomForest")
        
        # Definisci i parametri da ottimizzare
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Crea il grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Esegui la ricerca
        grid_search.fit(X_train, y_train)
        
        # Aggiorna il modello con i migliori parametri
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"Ottimizzazione completata. Migliori parametri: {grid_search.best_params_}")

class GradientBoostingModel(MLModel):
    """Modello GradientBoosting per il trading"""
    
    def __init__(self, model_name: str, n_estimators: int = 100, 
                learning_rate: float = 0.1, max_depth: int = 3, 
                random_state: int = 42):
        """
        Inizializza il modello GradientBoosting
        
        Args:
            model_name: Nome del modello
            n_estimators: Numero di estimatori
            learning_rate: Tasso di apprendimento
            max_depth: Profondità massima degli alberi
            random_state: Seed per la riproducibilità
        """
        super().__init__(model_name, "GradientBoosting")
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ottimizza gli iperparametri del modello GradientBoosting
        
        Args:
            X_train: Dati di training
            y_train: Target di training
        """
        self.logger.info("Inizio ottimizzazione iperparametri per GradientBoosting")
        
        # Definisci i parametri da ottimizzare
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Crea il grid search
        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Esegui la ricerca
        grid_search.fit(X_train, y_train)
        
        # Aggiorna il modello con i migliori parametri
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"Ottimizzazione completata. Migliori parametri: {grid_search.best_params_}")

class LogisticRegressionModel(MLModel):
    """Modello LogisticRegression per il trading"""
    
    def __init__(self, model_name: str, C: float = 1.0, 
                penalty: str = 'l2', solver: str = 'lbfgs', 
                random_state: int = 42):
        """
        Inizializza il modello LogisticRegression
        
        Args:
            model_name: Nome del modello
            C: Parametro di regolarizzazione
            penalty: Tipo di penalità
            solver: Algoritmo da utilizzare
            random_state: Seed per la riproducibilità
        """
        super().__init__(model_name, "LogisticRegression")
        
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            random_state=random_state,
            max_iter=1000,
            multi_class='auto'
        )
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Ottimizza gli iperparametri del modello LogisticRegression
        
        Args:
            X_train: Dati di training
            y_train: Target di training
        """
        self.logger.info("Inizio ottimizzazione iperparametri per LogisticRegression")
        
        # Definisci i parametri da ottimizzare
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        
        # Non tutti i solver sono compatibili con tutte le penalità
        # Rimossi alcuni parametri incompatibili
        
        # Crea il grid search
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid=param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Esegui la ricerca
        grid_search.fit(X_train, y_train)
        
        # Aggiorna il modello con i migliori parametri
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"Ottimizzazione completata. Migliori parametri: {grid_search.best_params_}")

def create_model(model_type: str, model_name: str, **kwargs) -> MLModel:
    """
    Crea un'istanza di un modello ML in base al tipo
    
    Args:
        model_type: Tipo di modello
        model_name: Nome del modello
        **kwargs: Parametri aggiuntivi per il modello
        
    Returns:
        Istanza del modello
    """
    model_type = model_type.lower()
    
    if model_type == 'randomforest':
        return RandomForestModel(model_name, **kwargs)
    elif model_type == 'gradientboosting':
        return GradientBoostingModel(model_name, **kwargs)
    elif model_type == 'logisticregression':
        return LogisticRegressionModel(model_name, **kwargs)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")

def get_available_models() -> List[str]:
    """
    Ottiene l'elenco dei modelli disponibili
    
    Returns:
        Lista dei tipi di modelli disponibili
    """
    return ['randomforest', 'gradientboosting', 'logisticregression']

def prepare_training_data(data: pd.DataFrame, target_generator: callable, 
                         future_periods: int = 10, binary: bool = True) -> pd.DataFrame:
    """
    Prepara i dati per l'addestramento
    
    Args:
        data: DataFrame con i dati (OHLCV e indicatori)
        target_generator: Funzione per generare il target
        future_periods: Numero di periodi futuri per il target
        binary: Se creare un target binario
        
    Returns:
        DataFrame con i dati e il target
    """
    if data.empty:
        return pd.DataFrame()
    
    # Crea una copia dei dati
    df = data.copy()
    
    # Genera il target
    df = target_generator(df, future_periods, binary)
    
    # Rimuovi le righe con dati mancanti
    df = df.dropna()
    
    return df

def generate_directional_target(data: pd.DataFrame, periods: int = 10, 
                               binary: bool = True) -> pd.DataFrame:
    """
    Genera un target direzionale basato sul prezzo futuro
    
    Args:
        data: DataFrame con i dati
        periods: Numero di periodi futuri
        binary: Se creare un target binario
        
    Returns:
        DataFrame con il target aggiunto
    """
    df = data.copy()
    
    # Calcola la variazione percentuale futura
    df['future_change'] = df['close'].shift(-periods) / df['close'] - 1
    
    if binary:
        # Crea un target binario (-1, 1)
        threshold = 0.005  # 0.5% come soglia
        df['target'] = 0  # Neutro di default
        df.loc[df['future_change'] > threshold, 'target'] = 1  # Rialzo
        df.loc[df['future_change'] < -threshold, 'target'] = -1  # Ribasso
    else:
        # Usa la variazione percentuale come target
        df['target'] = df['future_change']
    
    # Rimuovi la colonna temporanea
    df = df.drop('future_change', axis=1)
    
    return df

def generate_peak_trough_target(data: pd.DataFrame, periods: int = 10, 
                               binary: bool = True) -> pd.DataFrame:
    """
    Genera un target basato sui picchi e minimi
    
    Args:
        data: DataFrame con i dati
        periods: Finestra per la ricerca di picchi
        binary: Se creare un target binario
        
    Returns:
        DataFrame con il target aggiunto
    """
    df = data.copy()
    
    # Inizializza il target
    df['target'] = 0  # Neutro di default
    
    for i in range(periods, len(df) - periods):
        # Massimo locale
        if df.iloc[i]['high'] == df.iloc[i-periods:i+periods]['high'].max():
            df.loc[df.index[i], 'target'] = 1  # Picco
        
        # Minimo locale
        if df.iloc[i]['low'] == df.iloc[i-periods:i+periods]['low'].min():
            df.loc[df.index[i], 'target'] = -1  # Minimo
    
    if not binary:
        # Aggiungi un valore continuo basato sulla distanza dal picco/minimo
        df['target_value'] = 0.0
        
        for i in range(len(df)):
            if df.iloc[i]['target'] == 1:
                # Picco
                df.loc[df.index[i], 'target_value'] = 1.0
            elif df.iloc[i]['target'] == -1:
                # Minimo
                df.loc[df.index[i], 'target_value'] = -1.0
        
        # Riempi i valori mancanti con interpolazione
        df['target_value'] = df['target_value'].interpolate(method='linear')
        
        # Sostituisci il target
        df['target'] = df['target_value']
        df = df.drop('target_value', axis=1)
    
    return df

def evaluate_model(model: MLModel, test_data: pd.DataFrame, 
                  target_column: str = 'target') -> Dict[str, Any]:
    """
    Valuta le performance di un modello
    
    Args:
        model: Modello da valutare
        test_data: Dati di test
        target_column: Nome della colonna target
        
    Returns:
        Metriche di performance
    """
    if not model.trained:
        raise ValueError("Il modello non è stato addestrato")
    
    # Prepara i dati
    X = test_data.drop(target_column, axis=1)
    y_true = test_data[target_column].values
    
    # Genera le previsioni
    y_pred = model.predict(X)
    
    # Calcola le metriche
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics
