"""
Modulo per la gestione degli AI agents
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import threading
import time
import queue
import random
import uuid
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict

from models.ml_models import (
    RandomForestModel, GradientBoostingModel, LogisticRegressionModel, MLModel,
    create_model, prepare_training_data, generate_directional_target
)
from models.reinforcement import (
    DQNAgent, Environment, train_dqn_agent, evaluate_dqn_agent,
    create_trading_env, create_dqn_agent
)
from strategy.strategy_base import Signal, SignalType, EntryType
from utils.logger import get_logger
from config.settings import (
    MODEL_SAVE_PATH, FEATURE_WINDOW, LEARNING_RATE, BATCH_SIZE, EPOCHS, TRAIN_FREQUENCY
)

logger = get_logger(__name__)

class AIAgent:
    """Rappresenta un singolo agente AI"""
    
    def __init__(self, agent_id: str, agent_type: str, symbol: str):
        """
        Inizializza un agente AI
        
        Args:
            agent_id: ID dell'agente
            agent_type: Tipo di agente (ml/rl)
            symbol: Simbolo della coppia
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.symbol = symbol
        self.model = None
        self.performance = {}
        self.trades_history = []
        self.created_at = datetime.now()
        self.last_trained_at = None
        self.train_count = 0
        self.win_rate = 0.0
        self.config = {}
        
        self.logger = get_logger(f"{__name__}.{agent_id}")
        self.logger.info(f"Agente {agent_id} inizializzato per {symbol}")
    
    def predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Genera una previsione
        
        Args:
            data: DataFrame con i dati di input
            
        Returns:
            Tuple (previsione, probabilità)
        """
        if self.model is None:
            return 0, 0.0
        
        if self.agent_type == "ml":
            return self._ml_predict(data)
        elif self.agent_type == "rl":
            return self._rl_predict(data)
        else:
            return 0, 0.0
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Addestra l'agente
        
        Args:
            data: DataFrame con i dati di training
            
        Returns:
            Metriche di performance
        """
        if self.agent_type == "ml":
            return self._ml_train(data)
        elif self.agent_type == "rl":
            return self._rl_train(data)
        else:
            return {}
    
    def save(self) -> str:
        """
        Salva l'agente
        
        Returns:
            Percorso del file salvato
        """
        # Crea la directory se non esiste
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # Percorso del file
        filepath = os.path.join(MODEL_SAVE_PATH, f"{self.agent_id}.json")
        
        # Dati dell'agente
        agent_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "created_at": self.created_at.isoformat(),
            "last_trained_at": self.last_trained_at.isoformat() if self.last_trained_at else None,
            "train_count": self.train_count,
            "win_rate": self.win_rate,
            "performance": self.performance,
            "config": self.config
        }
        
        # Salva i dati dell'agente
        with open(filepath, 'w') as f:
            json.dump(agent_data, f, indent=2)
        
        # Salva il modello
        if self.model is not None:
            if self.agent_type == "ml" and isinstance(self.model, MLModel):
                model_path = os.path.join(MODEL_SAVE_PATH, f"{self.agent_id}_model.joblib")
                self.model.save_model(model_path)
            elif self.agent_type == "rl" and isinstance(self.model, DQNAgent):
                model_path = os.path.join(MODEL_SAVE_PATH, f"{self.agent_id}_model.h5")
                self.model.save(model_path)
        
        self.logger.info(f"Agente salvato in {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'AIAgent':
        """
        Carica un agente da file
        
        Args:
            filepath: Percorso del file
            
        Returns:
            Istanza di AIAgent
        """
        try:
            # Carica i dati dell'agente
            with open(filepath, 'r') as f:
                agent_data = json.load(f)
            
            # Crea una nuova istanza
            agent = cls(
                agent_id=agent_data["agent_id"],
                agent_type=agent_data["agent_type"],
                symbol=agent_data["symbol"]
            )
            
            # Ripristina i dati
            agent.created_at = datetime.fromisoformat(agent_data["created_at"])
            if agent_data["last_trained_at"]:
                agent.last_trained_at = datetime.fromisoformat(agent_data["last_trained_at"])
            agent.train_count = agent_data["train_count"]
            agent.win_rate = agent_data["win_rate"]
            agent.performance = agent_data["performance"]
            agent.config = agent_data["config"]
            
            # Carica il modello
            if agent.agent_type == "ml":
                model_path = os.path.join(MODEL_SAVE_PATH, f"{agent.agent_id}_model.joblib")
                if os.path.exists(model_path):
                    agent.model = MLModel.load_model(model_path)
            elif agent.agent_type == "rl":
                model_path = os.path.join(MODEL_SAVE_PATH, f"{agent.agent_id}_model.h5")
                if os.path.exists(model_path):
                    agent.model = DQNAgent.load(model_path)
            
            logger.info(f"Agente caricato da {filepath}")
            return agent
        
        except Exception as e:
            logger.error(f"Errore nel caricamento dell'agente da {filepath}: {str(e)}")
            raise
    
    def add_trade_result(self, trade_data: Dict[str, Any]) -> None:
        """
        Aggiunge il risultato di un trade
        
        Args:
            trade_data: Dati del trade
        """
        # Aggiungi il trade alla cronologia
        self.trades_history.append(trade_data)
        
        # Aggiorna il win rate
        win_trades = sum(1 for t in self.trades_history if t.get("profit", 0) > 0)
        self.win_rate = win_trades / len(self.trades_history) if self.trades_history else 0.0
        
        # Aggiorna le performance
        total_trades = len(self.trades_history)
        total_profit = sum(t.get("profit", 0) for t in self.trades_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0.0
        
        self.performance = {
            "total_trades": total_trades,
            "win_trades": win_trades,
            "loss_trades": total_trades - win_trades,
            "win_rate": self.win_rate,
            "total_profit": total_profit,
            "avg_profit": avg_profit
        }
        
        self.logger.info(f"Trade aggiunto: profit={trade_data.get('profit', 0):.2f}, win_rate={self.win_rate:.2f}")
    
    def should_train(self) -> bool:
        """
        Verifica se l'agente deve essere addestrato
        
        Returns:
            True se l'agente deve essere addestrato
        """
        # Se non è mai stato addestrato
        if self.last_trained_at is None:
            return True
        
        # Se sono passati abbastanza trade
        trades_since_last_train = len(self.trades_history)
        if self.last_trained_at:
            trades_since_last_train = sum(1 for t in self.trades_history 
                                       if t.get("timestamp") and datetime.fromisoformat(t["timestamp"]) > self.last_trained_at)
        
        if trades_since_last_train >= TRAIN_FREQUENCY:
            return True
        
        # Se sono passati abbastanza giorni
        days_since_last_train = (datetime.now() - self.last_trained_at).days if self.last_trained_at else 999
        if days_since_last_train >= 7:  # Almeno una volta a settimana
            return True
        
        return False
    
    def _ml_predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Genera una previsione con un modello ML
        
        Args:
            data: DataFrame con i dati di input
            
        Returns:
            Tuple (previsione, probabilità)
        """
        if not isinstance(self.model, MLModel):
            return 0, 0.0
        
        try:
            # Prepara i dati
            features = data.copy()
            
            # Rimuovi la colonna target se presente
            if 'target' in features.columns:
                features = features.drop('target', axis=1)
            
            # Effettua la previsione
            prediction = self.model.predict(features)
            
            # Ottieni la probabilità
            probability = 0.5
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features)
                
                # La probabilità corrisponde alla classe prevista
                if prediction == -1:
                    probability = probs[0][0]  # SELL
                elif prediction == 1:
                    probability = probs[0][2]  # BUY
                else:
                    probability = probs[0][1]  # HOLD
            
            return int(prediction), float(probability)
            
        except Exception as e:
            self.logger.error(f"Errore nella previsione ML: {str(e)}")
            return 0, 0.0
    
    def _rl_predict(self, data: pd.DataFrame) -> Tuple[int, float]:
        """
        Genera una previsione con un modello RL
        
        Args:
            data: DataFrame con i dati di input
            
        Returns:
            Tuple (previsione, probabilità)
        """
        if not isinstance(self.model, DQNAgent):
            return 0, 0.0
        
        try:
            # Prepara i dati
            env = Environment(data)
            state = env.reset()
            
            # Effettua la previsione
            action = self.model.act(state, training=False)
            
            # Mappa l'azione a buy/sell/hold
            if action == 1:  # Buy
                prediction = 1
            elif action == 2:  # Sell
                prediction = -1
            else:  # Hold
                prediction = 0
            
            # Probabilità fissa per ora (RL non fornisce probabilità)
            probability = 0.7
            
            return prediction, probability
            
        except Exception as e:
            self.logger.error(f"Errore nella previsione RL: {str(e)}")
            return 0, 0.0
    
    def _ml_train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Addestra un modello ML
        
        Args:
            data: DataFrame con i dati di training
            
        Returns:
            Metriche di performance
        """
        try:
            # Se il modello non esiste, crealo
            if self.model is None:
                model_type = self.config.get("model_type", "randomforest")
                self.model = create_model(model_type, f"{self.agent_id}_model")
            
            # Prepara i dati di training
            train_data = prepare_training_data(
                data, 
                generate_directional_target,
                future_periods=10,
                binary=True
            )
            
            # Addestra il modello
            metrics = self.model.train(train_data)
            
            # Aggiorna lo stato
            self.last_trained_at = datetime.now()
            self.train_count += 1
            
            self.logger.info(f"Modello ML addestrato: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento ML: {str(e)}")
            return {}
    
    def _rl_train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Addestra un modello RL
        
        Args:
            data: DataFrame con i dati di training
            
        Returns:
            Metriche di performance
        """
        try:
            # Crea l'ambiente
            env = create_trading_env(data)
            
            # Se il modello non esiste, crealo
            if self.model is None:
                state_size = len(env._get_observation())
                action_size = 3  # hold, buy, sell
                self.model = create_dqn_agent(state_size, action_size)
            
            # Addestra il modello
            train_results = train_dqn_agent(
                env, 
                self.model, 
                episodes=50,
                batch_size=32,
                render_interval=10
            )
            
            # Valuta il modello
            eval_results = evaluate_dqn_agent(env, self.model)
            
            # Aggiorna lo stato
            self.last_trained_at = datetime.now()
            self.train_count += 1
            
            # Combina i risultati
            metrics = {
                "train_results": train_results,
                "eval_results": eval_results
            }
            
            self.logger.info(f"Modello RL addestrato: win_rate={eval_results.get('win_rate', 0):.2f}%, "
                           f"profit={eval_results.get('total_profit', 0):.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento RL: {str(e)}")
            return {}

class AgentManager:
    """
    Classe per la gestione degli AI agents
    """
    
    def __init__(self):
        """Inizializza il gestore degli agenti"""
        self.logger = get_logger(__name__)
        
        # Agenti (symbol -> [agents])
        self.agents = defaultdict(list)
        
        # Thread e code
        self.training_thread = None
        self.running = False
        self.training_queue = queue.Queue()
        
        # Directory per i modelli
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        self.logger.info("AgentManager inizializzato")
    
    def start(self) -> None:
        """Avvia il gestore degli agenti"""
        if self.training_thread is not None and self.training_thread.is_alive():
            self.logger.warning("Thread di addestramento già in esecuzione")
            return
            
        self.running = True
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
        
        self.logger.info("AgentManager avviato")
    
    def stop(self) -> None:
        """Ferma il gestore degli agenti"""
        self.running = False
        
        if self.training_thread is not None:
            try:
                self.training_thread.join(timeout=5.0)
            except:
                pass
                
        self.logger.info("AgentManager fermato")
    
    def create_agent(self, symbol: str, agent_type: str = "ml",
                    config: Optional[Dict[str, Any]] = None) -> AIAgent:
        """
        Crea un nuovo agente
        
        Args:
            symbol: Simbolo della coppia
            agent_type: Tipo di agente (ml/rl)
            config: Configurazione dell'agente
            
        Returns:
            Nuova istanza di AIAgent
        """
        # Genera un ID univoco
        agent_id = f"agent_{symbol.replace('/', '_')}_{agent_type}_{uuid.uuid4().hex[:8]}"
        
        # Crea l'agente
        agent = AIAgent(agent_id, agent_type, symbol)
        
        # Imposta la configurazione
        agent.config = config or {}
        
        # Aggiungi l'agente
        self.agents[symbol].append(agent)
        
        self.logger.info(f"Agente {agent_id} creato per {symbol}")
        
        return agent
    
    def get_best_agent(self, symbol: str) -> Optional[AIAgent]:
        """
        Ottiene il miglior agente per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Miglior agente o None
        """
        agents = self.agents.get(symbol, [])
        
        if not agents:
            return None
        
        # Ordina per win rate o altro criterio
        sorted_agents = sorted(agents, key=lambda a: a.win_rate, reverse=True)
        
        return sorted_agents[0]
    
    def get_prediction(self, symbol: str, data: pd.DataFrame) -> Tuple[int, float, List[Dict[str, Any]]]:
        """
        Ottiene una previsione combinata dagli agenti
        
        Args:
            symbol: Simbolo della coppia
            data: DataFrame con i dati di input
            
        Returns:
            Tuple (previsione, probabilità, dettagli)
        """
        agents = self.agents.get(symbol, [])
        
        if not agents:
            return 0, 0.0, []
        
        # Raccogli le previsioni
        predictions = []
        
        for agent in agents:
            try:
                pred, prob = agent.predict(data)
                
                predictions.append({
                    "agent_id": agent.agent_id,
                    "prediction": pred,
                    "probability": prob,
                    "win_rate": agent.win_rate
                })
            except Exception as e:
                self.logger.error(f"Errore nella previsione dell'agente {agent.agent_id}: {str(e)}")
        
        if not predictions:
            return 0, 0.0, []
        
        # Calcola il voto ponderato
        weighted_vote = 0.0
        total_weight = 0.0
        
        for p in predictions:
            # Peso basato su win rate e probabilità
            weight = p["win_rate"] * p["probability"]
            
            weighted_vote += p["prediction"] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_vote = weighted_vote / total_weight
        else:
            final_vote = 0.0
        
        # Considera la forza del segnale
        if final_vote > 0.3:
            prediction = 1  # BUY
        elif final_vote < -0.3:
            prediction = -1  # SELL
        else:
            prediction = 0  # HOLD
        
        # Calcola la probabilità come forza del segnale
        probability = abs(final_vote)
        
        return prediction, probability, predictions
    
    def notify_trade_result(self, symbol: str, trade_data: Dict[str, Any]) -> None:
        """
        Notifica agli agenti il risultato di un trade
        
        Args:
            symbol: Simbolo della coppia
            trade_data: Dati del trade
        """
        agents = self.agents.get(symbol, [])
        
        if not agents:
            return
        
        for agent in agents:
            try:
                agent.add_trade_result(trade_data)
                
                # Verifica se l'agente deve essere addestrato
                if agent.should_train():
                    self.training_queue.put((agent, None))
            except Exception as e:
                self.logger.error(f"Errore nella notifica del trade all'agente {agent.agent_id}: {str(e)}")
    
    def train_agent(self, agent: AIAgent, data: Optional[pd.DataFrame] = None) -> None:
        """
        Addestra un agente
        
        Args:
            agent: Agente da addestrare
            data: Dati di training (opzionale)
        """
        self.training_queue.put((agent, data))
    
    def save_agents(self) -> None:
        """Salva tutti gli agenti"""
        for symbol, agents in self.agents.items():
            for agent in agents:
                try:
                    agent.save()
                except Exception as e:
                    self.logger.error(f"Errore nel salvataggio dell'agente {agent.agent_id}: {str(e)}")
    
    def load_agents(self) -> None:
        """Carica gli agenti salvati"""
        try:
            # Cerca tutti i file JSON nella directory
            for filename in os.listdir(MODEL_SAVE_PATH):
                if filename.endswith(".json") and filename.startswith("agent_"):
                    filepath = os.path.join(MODEL_SAVE_PATH, filename)
                    
                    try:
                        # Carica l'agente
                        agent = AIAgent.load(filepath)
                        
                        # Aggiungi l'agente
                        self.agents[agent.symbol].append(agent)
                        
                        self.logger.info(f"Agente {agent.agent_id} caricato")
                    except Exception as e:
                        self.logger.error(f"Errore nel caricamento dell'agente da {filepath}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Errore nel caricamento degli agenti: {str(e)}")
    
    def get_agents_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche degli agenti
        
        Returns:
            Statistiche degli agenti
        """
        stats = {
            "total_agents": 0,
            "agents_by_symbol": {},
            "agents_by_type": {
                "ml": 0,
                "rl": 0
            },
            "best_agents": {}
        }
        
        for symbol, agents in self.agents.items():
            stats["total_agents"] += len(agents)
            stats["agents_by_symbol"][symbol] = len(agents)
            
            for agent in agents:
                stats["agents_by_type"][agent.agent_type] += 1
            
            # Miglior agente per simbolo
            best_agent = self.get_best_agent(symbol)
            if best_agent:
                stats["best_agents"][symbol] = {
                    "agent_id": best_agent.agent_id,
                    "win_rate": best_agent.win_rate,
                    "total_trades": best_agent.performance.get("total_trades", 0),
                    "total_profit": best_agent.performance.get("total_profit", 0.0)
                }
        
        return stats
    
    def _training_worker(self) -> None:
        """Thread worker per l'addestramento"""
        self.logger.info("Worker di addestramento avviato")
        
        while self.running:
            try:
                # Ottieni un agente dalla coda
                try:
                    agent, data = self.training_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Addestra l'agente
                try:
                    self.logger.info(f"Addestramento dell'agente {agent.agent_id}...")
                    
                    if data is None:
                        # Usa dati simulati per ora
                        # In un'implementazione reale, si dovrebbero utilizzare dati dal market data
                        data = pd.DataFrame()
                    
                    if not data.empty:
                        # Addestra l'agente
                        agent.train(data)
                        
                        # Salva l'agente
                        agent.save()
                except Exception as e:
                    self.logger.error(f"Errore nell'addestramento dell'agente {agent.agent_id}: {str(e)}")
                
                # Segna il task come completato
                self.training_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Errore nel worker di addestramento: {str(e)}")
                time.sleep(5.0)
