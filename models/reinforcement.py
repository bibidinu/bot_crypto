"""
Modelli di reinforcement learning per il trading
"""
import os
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import time
import joblib
import pickle
from collections import deque

from utils.logger import get_logger
from config.settings import MODEL_SAVE_PATH

logger = get_logger(__name__)

class Environment:
    """Ambiente di trading per reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                transaction_fee: float = 0.001, window_size: int = 20):
        """
        Inizializza l'ambiente
        
        Args:
            data: DataFrame con i dati OHLCV
            initial_balance: Bilancio iniziale
            transaction_fee: Commissione di transazione (percentuale)
            window_size: Dimensione della finestra di osservazione
        """
        self.logger = get_logger(__name__)
        
        # Verifica che il dataframe contenga i dati necessari
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"DataFrame mancante di colonne richieste: {missing}")
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        
        # Stato corrente
        self.balance = initial_balance
        self.shares_held = 0.0
        self.current_step = 0
        self.current_price = 0.0
        self.current_value = initial_balance
        self.trade_history = []
        
        # Limiti
        self.start_step = window_size
        self.end_step = len(data) - 1
        
        self.logger.info(f"Ambiente inizializzato con {len(data)} dati e bilancio {initial_balance}")
    
    def reset(self) -> np.ndarray:
        """
        Resetta l'ambiente
        
        Returns:
            Stato iniziale
        """
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.current_step = self.start_step
        self.current_price = self.data.iloc[self.current_step]['close']
        self.current_value = self.initial_balance
        self.trade_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Ottiene l'osservazione corrente
        
        Returns:
            Vettore di osservazione
        """
        # Estrai la finestra di dati
        window_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalizza i dati nella finestra
        # Dividi ogni colonna per il suo massimo
        normalized_data = window_data.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            max_val = normalized_data[col].max()
            if max_val > 0:
                normalized_data[col] = normalized_data[col] / max_val
        
        # Normalizza il volume
        max_volume = normalized_data['volume'].max()
        if max_volume > 0:
            normalized_data['volume'] = normalized_data['volume'] / max_volume
        
        # Aggiungi informazioni sulla posizione corrente
        position = np.array([
            self.shares_held / 100,  # Normalizza le azioni detenute
            self.balance / self.initial_balance,  # Bilancio normalizzato
            1.0 if self.shares_held > 0 else 0.0,  # Flag per posizione long
            1.0 if self.shares_held < 0 else 0.0   # Flag per posizione short
        ])
        
        # Converti in array numpy
        obs_data = normalized_data[['open', 'high', 'low', 'close', 'volume']].values.flatten()
        
        # Concatena i dati OHLCV con le informazioni sulla posizione
        observation = np.concatenate([obs_data, position])
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Esegue un passo nell'ambiente
        
        Args:
            action: Azione da eseguire (0: hold, 1: buy, 2: sell)
            
        Returns:
            Tuple (next_state, reward, done, info)
        """
        # Esegui l'azione
        self._take_action(action)
        
        # Passa al prossimo step
        self.current_step += 1
        done = self.current_step >= self.end_step
        
        # Aggiorna il prezzo corrente
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # Calcola il valore totale (bilancio + valore delle azioni)
        new_value = self.balance + self.shares_held * self.current_price
        
        # Calcola la ricompensa (variazione di valore)
        reward = new_value - self.current_value
        
        # Normalizza la ricompensa
        reward = reward / self.initial_balance
        
        # Aggiorna il valore corrente
        self.current_value = new_value
        
        # Informazioni aggiuntive
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'current_value': self.current_value,
            'step': self.current_step
        }
        
        # Ottieni la nuova osservazione
        next_state = self._get_observation()
        
        return next_state, reward, done, info
    
    def _take_action(self, action: int) -> None:
        """
        Esegue un'azione nell'ambiente
        
        Args:
            action: Azione da eseguire (0: hold, 1: buy, 2: sell)
        """
        # Prezzo corrente
        price = self.data.iloc[self.current_step]['close']
        
        # Azione: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1:  # Buy
            # Calcola il massimo di azioni acquistabili
            max_shares = self.balance / (price * (1 + self.transaction_fee))
            
            # Acquista la metà delle azioni disponibili
            shares_to_buy = max_shares * 0.5
            
            # Costo totale
            cost = shares_to_buy * price * (1 + self.transaction_fee)
            
            # Verifica che ci sia abbastanza saldo
            if cost > 0 and cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
                
                # Registra il trade
                self.trade_history.append({
                    'step': self.current_step,
                    'timestamp': self.data.index[self.current_step],
                    'action': 'buy',
                    'price': price,
                    'shares': shares_to_buy,
                    'cost': cost,
                    'balance': self.balance,
                    'shares_held': self.shares_held
                })
        
        elif action == 2:  # Sell
            # Vendi tutte le azioni detenute
            if self.shares_held > 0:
                # Ricavo totale
                proceeds = self.shares_held * price * (1 - self.transaction_fee)
                
                # Aggiorna il bilancio
                self.balance += proceeds
                
                # Registra il trade
                self.trade_history.append({
                    'step': self.current_step,
                    'timestamp': self.data.index[self.current_step],
                    'action': 'sell',
                    'price': price,
                    'shares': self.shares_held,
                    'proceeds': proceeds,
                    'balance': self.balance,
                    'shares_held': 0.0
                })
                
                # Azzera le azioni detenute
                self.shares_held = 0.0
    
    def render(self) -> None:
        """
        Visualizza lo stato corrente dell'ambiente
        """
        price = self.data.iloc[self.current_step]['close']
        value = self.balance + self.shares_held * price
        profit = value - self.initial_balance
        profit_percent = (profit / self.initial_balance) * 100
        
        print(f"Step: {self.current_step}/{self.end_step}")
        print(f"Price: {price:.2f}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Shares held: {self.shares_held:.6f}")
        print(f"Current value: {value:.2f} ({profit_percent:.2f}%)")
        print(f"Trades: {len(self.trade_history)}")
        print("----------------------------")
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Ottiene le metriche di performance
        
        Returns:
            Dizionario con le metriche
        """
        if not self.trade_history:
            return {
                'total_profit': 0.0,
                'total_profit_percent': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'average_profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Ultimo prezzo
        last_price = self.data.iloc[self.current_step]['close']
        
        # Valore finale
        final_value = self.balance + self.shares_held * last_price
        
        # Profitto totale
        total_profit = final_value - self.initial_balance
        total_profit_percent = (total_profit / self.initial_balance) * 100
        
        # Calcola le metriche
        num_trades = len([t for t in self.trade_history if t['action'] in ['buy', 'sell']])
        
        # Calcola la serie dei valori
        values = []
        peak = self.initial_balance
        drawdowns = []
        
        for step in range(self.start_step, self.current_step + 1):
            price = self.data.iloc[step]['close']
            
            # Trova il trade più recente a questo step
            position = 0.0
            balance = self.initial_balance
            
            for trade in self.trade_history:
                if trade['step'] <= step:
                    if trade['action'] == 'buy':
                        position = trade.get('shares_held', 0.0)
                        balance = trade.get('balance', self.initial_balance)
                    elif trade['action'] == 'sell':
                        position = 0.0
                        balance = trade.get('balance', self.initial_balance)
            
            # Calcola il valore
            value = balance + position * price
            values.append(value)
            
            # Aggiorna il picco e calcola il drawdown
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        # Calcola il rendimento giornaliero
        daily_returns = []
        
        for i in range(1, len(values)):
            daily_return = (values[i] - values[i-1]) / values[i-1]
            daily_returns.append(daily_return)
        
        # Calcola il Sharpe ratio
        if daily_returns:
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Massimo drawdown
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        
        # Win rate (percentuale di trade vincenti)
        wins = 0
        for i, trade in enumerate(self.trade_history):
            if trade['action'] == 'sell':
                # Trova l'ultimo acquisto
                buy_trade = None
                for j in range(i-1, -1, -1):
                    if self.trade_history[j]['action'] == 'buy':
                        buy_trade = self.trade_history[j]
                        break
                
                if buy_trade:
                    buy_price = buy_trade['price']
                    sell_price = trade['price']
                    
                    if sell_price > buy_price:
                        wins += 1
        
        win_rate = wins / (num_trades / 2) * 100 if num_trades > 0 else 0.0
        
        # Profitto medio per trade
        average_profit = total_profit / (num_trades / 2) if num_trades > 0 else 0.0
        
        return {
            'total_profit': total_profit,
            'total_profit_percent': total_profit_percent,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'average_profit': average_profit,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

class DQNAgent:
    """Agente di reinforcement learning basato su Deep Q-Network"""
    
    def __init__(self, state_size: int, action_size: int,
                learning_rate: float = 0.001, gamma: float = 0.95,
                epsilon: float = 1.0, epsilon_min: float = 0.01,
                epsilon_decay: float = 0.995, memory_size: int = 2000):
        """
        Inizializza l'agente DQN
        
        Args:
            state_size: Dimensione dello stato
            action_size: Dimensione dell'azione
            learning_rate: Tasso di apprendimento
            gamma: Fattore di sconto
            epsilon: Epsilon per l'exploration
            epsilon_min: Epsilon minimo
            epsilon_decay: Decadimento di epsilon
            memory_size: Dimensione della memoria
        """
        self.logger = get_logger(__name__)
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Memoria di replay (deque è una lista a doppia estremità ottimizzata)
        self.memory = deque(maxlen=memory_size)
        
        # Modello Q
        self.model = self._build_model()
        
        # Modello target (per la stabilità dell'apprendimento)
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.logger.info(f"Agente DQN inizializzato con state_size={state_size}, action_size={action_size}")
    
    def _build_model(self):
        """
        Costruisce la rete neurale
        
        Returns:
            Modello Keras
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential()
            model.add(Dense(64, input_dim=self.state_size, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            
            return model
            
        except ImportError:
            self.logger.error("TensorFlow non disponibile. Installare TensorFlow per utilizzare DQNAgent.")
            raise
    
    def update_target_model(self) -> None:
        """
        Aggiorna il modello target copiando i pesi dal modello principale
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Memorizza un'esperienza nella memoria di replay
        
        Args:
            state: Stato corrente
            action: Azione eseguita
            reward: Ricompensa ottenuta
            next_state: Stato successivo
            done: Se l'episodio è terminato
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleziona un'azione in base alla policy
        
        Args:
            state: Stato corrente
            training: Se l'agente è in fase di training
            
        Returns:
            Azione selezionata
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploration: azione casuale
            return random.randrange(self.action_size)
        
        # Exploitation: azione con il massimo Q-value
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int) -> float:
        """
        Apprende dai campioni nella memoria di replay
        
        Args:
            batch_size: Dimensione del batch
            
        Returns:
            Loss dell'addestramento
        """
        if len(self.memory) < batch_size:
            return 0.0
        
        # Campiona un batch di esperienze
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepara i batch di dati
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Ottieni il target Q-value per azione
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                # Per stati terminali, il target è solo la ricompensa
                target[action] = reward
            else:
                # Target Q-value = r + γ * max(Q)
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.max(t)
            
            targets[i] = target
        
        # Addestra il modello
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decrementa epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, filepath: str) -> None:
        """
        Salva il modello
        
        Args:
            filepath: Percorso per il salvataggio
        """
        try:
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Salva il modello
            self.model.save(filepath)
            
            # Salva i parametri dell'agente
            params = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
            
            # Salva i parametri in un file separato
            params_path = filepath + '.params'
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            
            self.logger.info(f"Modello salvato in {filepath}")
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del modello: {str(e)}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'DQNAgent':
        """
        Carica un modello salvato
        
        Args:
            filepath: Percorso del modello
            
        Returns:
            Istanza di DQNAgent
        """
        try:
            from tensorflow.keras.models import load_model
            
            # Carica i parametri
            params_path = filepath + '.params'
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            # Crea una nuova istanza con i parametri salvati
            agent = cls(
                state_size=params['state_size'],
                action_size=params['action_size'],
                learning_rate=params['learning_rate'],
                gamma=params['gamma'],
                epsilon=params['epsilon'],
                epsilon_min=params['epsilon_min'],
                epsilon_decay=params['epsilon_decay']
            )
            
            # Carica il modello
            agent.model = load_model(filepath)
            agent.update_target_model()
            
            agent.logger.info(f"Modello caricato da {filepath}")
            
            return agent
            
        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {str(e)}")
            raise

def train_dqn_agent(env: Environment, agent: DQNAgent, episodes: int = 100, 
                   batch_size: int = 32, render_interval: int = 10, 
                   save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Addestra un agente DQN
    
    Args:
        env: Ambiente di trading
        agent: Agente DQN
        episodes: Numero di episodi
        batch_size: Dimensione del batch
        render_interval: Intervallo per la visualizzazione
        save_path: Percorso per salvare il modello
        
    Returns:
        Dizionario con i risultati dell'addestramento
    """
    logger.info(f"Inizio addestramento per {episodes} episodi")
    
    # Memorizza i risultati
    results = {
        'episode_rewards': [],
        'episode_profits': [],
        'losses': []
    }
    
    for episode in range(episodes):
        # Resetta l'ambiente
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        
        # Variabili per l'episodio
        total_reward = 0
        losses = []
        
        # Flag per l'episodio terminato
        done = False
        
        while not done:
            # Seleziona un'azione
            action = agent.act(state)
            
            # Esegui l'azione
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            # Memorizza l'esperienza
            agent.remember(state[0], action, reward, next_state[0], done)
            
            # Aggiorna lo stato
            state = next_state
            
            # Accumula la ricompensa
            total_reward += reward
            
            # Apprendi dall'esperienza
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                losses.append(loss)
        
        # Aggiorna il modello target ogni 10 episodi
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Ottieni le performance
        performance = env.get_performance()
        total_profit = performance['total_profit']
        
        # Salva i risultati
        results['episode_rewards'].append(total_reward)
        results['episode_profits'].append(total_profit)
        
        if losses:
            results['losses'].append(np.mean(losses))
        
        # Visualizza i risultati
        if episode % render_interval == 0:
            logger.info(f"Episodio: {episode}/{episodes}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Profit: {total_profit:.2f} ({performance['total_profit_percent']:.2f}%), "
                      f"Epsilon: {agent.epsilon:.2f}")
            
            if losses:
                logger.info(f"Loss media: {np.mean(losses):.6f}")
        
        # Salva il modello
        if save_path and (episode + 1) % 50 == 0:
            save_file = f"{save_path}_episode_{episode + 1}.h5"
            agent.save(save_file)
    
    # Salva il modello finale
    if save_path:
        save_file = f"{save_path}_final.h5"
        agent.save(save_file)
        logger.info(f"Modello salvato in {save_file}")
    
    logger.info("Addestramento completato")
    
    return results

def evaluate_dqn_agent(env: Environment, agent: DQNAgent) -> Dict[str, Any]:
    """
    Valuta un agente DQN
    
    Args:
        env: Ambiente di trading
        agent: Agente DQN
        
    Returns:
        Metriche di performance
    """
    logger.info("Inizio valutazione dell'agente")
    
    # Resetta l'ambiente
    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    
    # Flag per l'episodio terminato
    done = False
    
    # Disabilita l'exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    while not done:
        # Seleziona un'azione
        action = agent.act(state, training=False)
        
        # Esegui l'azione
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        
        # Aggiorna lo stato
        state = next_state
    
    # Ripristina epsilon
    agent.epsilon = original_epsilon
    
    # Ottieni le performance
    performance = env.get_performance()
    
    logger.info(f"Valutazione completata. "
               f"Profitto: {performance['total_profit']:.2f} ({performance['total_profit_percent']:.2f}%), "
               f"Trade: {performance['num_trades']}, "
               f"Win rate: {performance['win_rate']:.2f}%")
    
    return performance

def create_trading_env(data: pd.DataFrame, **kwargs) -> Environment:
    """
    Crea un ambiente di trading
    
    Args:
        data: DataFrame con i dati OHLCV
        **kwargs: Parametri aggiuntivi per l'ambiente
        
    Returns:
        Istanza di Environment
    """
    return Environment(data, **kwargs)

def create_dqn_agent(state_size: int, action_size: int, **kwargs) -> DQNAgent:
    """
    Crea un agente DQN
    
    Args:
        state_size: Dimensione dello stato
        action_size: Dimensione dell'azione
        **kwargs: Parametri aggiuntivi per l'agente
        
    Returns:
        Istanza di DQNAgent
    """
    return DQNAgent(state_size, action_size, **kwargs)
