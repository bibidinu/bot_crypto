"""
Modulo per la gestione del database del trading bot
"""
import os
import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

from utils.logger import get_logger
from config.settings import DATABASE_PATH

logger = get_logger(__name__)

class DatabaseManager:
    """Classe per la gestione del database"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Inizializza il manager del database
        
        Args:
            db_path: Percorso del database
        """
        self.logger = get_logger(__name__)
        self.db_path = db_path
        
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Inizializza il database
        self._init_db()
        
        self.logger.info(f"DatabaseManager inizializzato con database in {db_path}")
    
    def _init_db(self) -> None:
        """
        Inizializza le tabelle del database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Tabella per le configurazioni
            c.execute('''
                CREATE TABLE IF NOT EXISTS configurations (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Tabella per i trade
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    profit REAL,
                    profit_percent REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    strategy TEXT,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            # Tabella per i segnali
            c.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    signal_type TEXT,
                    entry_type TEXT,
                    price REAL,
                    strength REAL,
                    stop_loss REAL,
                    take_profits TEXT,
                    reason TEXT,
                    timestamp TEXT,
                    metadata TEXT
                )
            ''')
            
            # Tabella per le posizioni
            c.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_type TEXT,
                    entry_price REAL,
                    size REAL,
                    stop_loss REAL,
                    take_profits TEXT,
                    entry_time TEXT,
                    status TEXT,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    metadata TEXT
                )
            ''')
            
            # Tabella per i market data
            c.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            ''')
            
            # Tabella per le prestazioni
            c.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_trades INTEGER,
                    win_trades INTEGER,
                    loss_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    total_profit REAL,
                    max_drawdown REAL,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database inizializzato")
            
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione del database: {str(e)}")
            raise
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Salva un trade nel database
        
        Args:
            trade_data: Dati del trade
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Estrai i dati del trade
            trade_id = trade_data.get('id')
            symbol = trade_data.get('symbol')
            entry_type = trade_data.get('entry_type')
            entry_price = trade_data.get('entry_price')
            exit_price = trade_data.get('exit_price')
            size = trade_data.get('size')
            profit = trade_data.get('profit')
            profit_percent = trade_data.get('profit_percent')
            entry_time = trade_data.get('entry_time')
            exit_time = trade_data.get('exit_time')
            strategy = trade_data.get('strategy')
            status = trade_data.get('status')
            
            # Converti metadati in JSON
            metadata_keys = set(trade_data.keys()) - {
                'id', 'symbol', 'entry_type', 'entry_price', 'exit_price', 
                'size', 'profit', 'profit_percent', 'entry_time', 'exit_time', 
                'strategy', 'status'
            }
            
            metadata = {key: trade_data[key] for key in metadata_keys}
            metadata_json = json.dumps(metadata)
            
            # Inserisci o aggiorna il trade
            c.execute('''
                INSERT OR REPLACE INTO trades
                (id, symbol, entry_type, entry_price, exit_price, size, profit, 
                 profit_percent, entry_time, exit_time, strategy, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, symbol, entry_type, entry_price, exit_price, size, profit,
                profit_percent, entry_time, exit_time, strategy, status, metadata_json
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Trade salvato: {trade_id} - {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del trade: {str(e)}")
            return False
    
    def get_trades(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Ottiene i trade dal database
        
        Args:
            filters: Filtri da applicare (opzionale)
            limit: Numero massimo di trade da restituire
            
        Returns:
            Lista di trade
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Costruisci la query
            query = "SELECT * FROM trades"
            params = []
            
            if filters:
                conditions = []
                
                for key, value in filters.items():
                    if key in ['id', 'symbol', 'entry_type', 'strategy', 'status']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            # Esegui la query
            c.execute(query, params)
            rows = c.fetchall()
            
            # Converti i risultati in dizionari
            trades = []
            for row in rows:
                trade = dict(row)
                
                # Converti i metadati da JSON
                if 'metadata' in trade and trade['metadata']:
                    try:
                        trade['metadata'] = json.loads(trade['metadata'])
                    except:
                        trade['metadata'] = {}
                
                trades.append(trade)
            
            conn.close()
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei trade: {str(e)}")
            return []
    
    def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Salva un segnale nel database
        
        Args:
            signal_data: Dati del segnale
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Estrai i dati del segnale
            signal_id = signal_data.get('id')
            symbol = signal_data.get('symbol')
            signal_type = signal_data.get('signal_type')
            entry_type = signal_data.get('entry_type')
            price = signal_data.get('price')
            strength = signal_data.get('strength')
            stop_loss = signal_data.get('stop_loss')
            take_profits = json.dumps(signal_data.get('take_profits', []))
            reason = signal_data.get('reason')
            timestamp = signal_data.get('timestamp')
            
            # Converti metadati in JSON
            metadata_keys = set(signal_data.keys()) - {
                'id', 'symbol', 'signal_type', 'entry_type', 'price', 
                'strength', 'stop_loss', 'take_profits', 'reason', 'timestamp'
            }
            
            metadata = {key: signal_data[key] for key in metadata_keys}
            metadata_json = json.dumps(metadata)
            
            # Inserisci o aggiorna il segnale
            c.execute('''
                INSERT OR REPLACE INTO signals
                (id, symbol, signal_type, entry_type, price, strength, 
                 stop_loss, take_profits, reason, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id, symbol, signal_type, entry_type, price, strength,
                stop_loss, take_profits, reason, timestamp, metadata_json
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Segnale salvato: {signal_id} - {symbol} - {signal_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del segnale: {str(e)}")
            return False
    
    def get_signals(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Ottiene i segnali dal database
        
        Args:
            filters: Filtri da applicare (opzionale)
            limit: Numero massimo di segnali da restituire
            
        Returns:
            Lista di segnali
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Costruisci la query
            query = "SELECT * FROM signals"
            params = []
            
            if filters:
                conditions = []
                
                for key, value in filters.items():
                    if key in ['id', 'symbol', 'signal_type', 'entry_type']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Esegui la query
            c.execute(query, params)
            rows = c.fetchall()
            
            # Converti i risultati in dizionari
            signals = []
            for row in rows:
                signal = dict(row)
                
                # Converti take_profits da JSON
                if 'take_profits' in signal and signal['take_profits']:
                    try:
                        signal['take_profits'] = json.loads(signal['take_profits'])
                    except:
                        signal['take_profits'] = []
                
                # Converti i metadati da JSON
                if 'metadata' in signal and signal['metadata']:
                    try:
                        signal['metadata'] = json.loads(signal['metadata'])
                    except:
                        signal['metadata'] = {}
                
                signals.append(signal)
            
            conn.close()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei segnali: {str(e)}")
            return []
    
    def save_position(self, position_data: Dict[str, Any]) -> bool:
        """
        Salva una posizione nel database
        
        Args:
            position_data: Dati della posizione
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Estrai i dati della posizione
            position_id = position_data.get('id')
            symbol = position_data.get('symbol')
            entry_type = position_data.get('entry_type')
            entry_price = position_data.get('entry_price')
            size = position_data.get('size')
            stop_loss = position_data.get('stop_loss')
            take_profits = json.dumps(position_data.get('take_profits', []))
            entry_time = position_data.get('entry_time')
            status = position_data.get('status')
            realized_pnl = position_data.get('realized_pnl', 0.0)
            unrealized_pnl = position_data.get('unrealized_pnl', 0.0)
            
            # Converti metadati in JSON
            metadata_keys = set(position_data.keys()) - {
                'id', 'symbol', 'entry_type', 'entry_price', 'size', 
                'stop_loss', 'take_profits', 'entry_time', 'status',
                'realized_pnl', 'unrealized_pnl'
            }
            
            metadata = {key: position_data[key] for key in metadata_keys}
            metadata_json = json.dumps(metadata)
            
            # Inserisci o aggiorna la posizione
            c.execute('''
                INSERT OR REPLACE INTO positions
                (id, symbol, entry_type, entry_price, size, stop_loss, 
                 take_profits, entry_time, status, realized_pnl, unrealized_pnl, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position_id, symbol, entry_type, entry_price, size, stop_loss,
                take_profits, entry_time, status, realized_pnl, unrealized_pnl, metadata_json
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Posizione salvata: {position_id} - {symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio della posizione: {str(e)}")
            return False
    
    def get_positions(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Ottiene le posizioni dal database
        
        Args:
            filters: Filtri da applicare (opzionale)
            limit: Numero massimo di posizioni da restituire
            
        Returns:
            Lista di posizioni
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Costruisci la query
            query = "SELECT * FROM positions"
            params = []
            
            if filters:
                conditions = []
                
                for key, value in filters.items():
                    if key in ['id', 'symbol', 'entry_type', 'status']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            # Esegui la query
            c.execute(query, params)
            rows = c.fetchall()
            
            # Converti i risultati in dizionari
            positions = []
            for row in rows:
                position = dict(row)
                
                # Converti take_profits da JSON
                if 'take_profits' in position and position['take_profits']:
                    try:
                        position['take_profits'] = json.loads(position['take_profits'])
                    except:
                        position['take_profits'] = []
                
                # Converti i metadati da JSON
                if 'metadata' in position and position['metadata']:
                    try:
                        position['metadata'] = json.loads(position['metadata'])
                    except:
                        position['metadata'] = {}
                
                positions.append(position)
            
            conn.close()
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero delle posizioni: {str(e)}")
            return []
    
    def save_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Salva i dati di mercato nel database
        
        Args:
            symbol: Simbolo della coppia
            timeframe: Intervallo temporale
            data: DataFrame con i dati OHLCV
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Prepara i dati
            rows = []
            for idx, row in data.iterrows():
                # Converti l'indice in timestamp
                if isinstance(idx, pd.Timestamp):
                    timestamp = idx.isoformat()
                else:
                    timestamp = str(idx)
                
                rows.append((
                    symbol,
                    timeframe,
                    timestamp,
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))
            
            # Inserisci i dati
            c.executemany('''
                INSERT OR REPLACE INTO market_data
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', rows)
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Dati di mercato salvati: {symbol} - {timeframe} - {len(rows)} candele")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio dei dati di mercato: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str, timeframe: str, start_time: Optional[str] = None, 
                       end_time: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Ottiene i dati di mercato dal database
        
        Args:
            symbol: Simbolo della coppia
            timeframe: Intervallo temporale
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            limit: Numero massimo di candele da restituire
            
        Returns:
            DataFrame con i dati OHLCV
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Costruisci la query
            query = "SELECT * FROM market_data WHERE symbol = ? AND timeframe = ?"
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Esegui la query
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Converti i timestamp in indici datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Ordina per timestamp
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei dati di mercato: {str(e)}")
            return pd.DataFrame()
    
    def save_config(self, key: str, value: Any) -> bool:
        """
        Salva una configurazione nel database
        
        Args:
            key: Chiave della configurazione
            value: Valore della configurazione
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Converti il valore in JSON
            if not isinstance(value, str):
                value = json.dumps(value)
            
            # Timestamp corrente
            timestamp = datetime.now().isoformat()
            
            # Inserisci o aggiorna la configurazione
            c.execute('''
                INSERT OR REPLACE INTO configurations
                (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, value, timestamp))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Configurazione salvata: {key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio della configurazione: {str(e)}")
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Ottiene una configurazione dal database
        
        Args:
            key: Chiave della configurazione
            default: Valore di default se la chiave non esiste
            
        Returns:
            Valore della configurazione
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Esegui la query
            c.execute("SELECT value FROM configurations WHERE key = ?", (key,))
            row = c.fetchone()
            
            conn.close()
            
            if row is None:
                return default
            
            value = row[0]
            
            # Prova a convertire il valore da JSON
            try:
                return json.loads(value)
            except:
                return value
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero della configurazione: {str(e)}")
            return default
    
    def save_performance(self, performance_data: Dict[str, Any]) -> bool:
        """
        Salva una performance nel database
        
        Args:
            performance_data: Dati della performance
            
        Returns:
            True se il salvataggio è riuscito
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Estrai i dati della performance
            perf_id = performance_data.get('id')
            start_time = performance_data.get('start_time')
            end_time = performance_data.get('end_time')
            total_trades = performance_data.get('total_trades')
            win_trades = performance_data.get('win_trades')
            loss_trades = performance_data.get('loss_trades')
            win_rate = performance_data.get('win_rate')
            profit_factor = performance_data.get('profit_factor')
            total_profit = performance_data.get('total_profit')
            max_drawdown = performance_data.get('max_drawdown')
            
            # Converti metadati in JSON
            metadata_keys = set(performance_data.keys()) - {
                'id', 'start_time', 'end_time', 'total_trades', 'win_trades', 
                'loss_trades', 'win_rate', 'profit_factor', 'total_profit', 'max_drawdown'
            }
            
            metadata = {key: performance_data[key] for key in metadata_keys}
            metadata_json = json.dumps(metadata)
            
            # Inserisci o aggiorna la performance
            c.execute('''
                INSERT OR REPLACE INTO performance
                (id, start_time, end_time, total_trades, win_trades, loss_trades, 
                 win_rate, profit_factor, total_profit, max_drawdown, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                perf_id, start_time, end_time, total_trades, win_trades, loss_trades,
                win_rate, profit_factor, total_profit, max_drawdown, metadata_json
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.debug(f"Performance salvata: {perf_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio della performance: {str(e)}")
            return False
    
    def get_performances(self, start_time: Optional[str] = None, 
                        end_time: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ottiene le performance dal database
        
        Args:
            start_time: Timestamp di inizio (opzionale)
            end_time: Timestamp di fine (opzionale)
            limit: Numero massimo di performance da restituire
            
        Returns:
            Lista di performance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # Costruisci la query
            query = "SELECT * FROM performance"
            params = []
            
            conditions = []
            
            if start_time:
                conditions.append("start_time >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("end_time <= ?")
                params.append(end_time)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY end_time DESC LIMIT ?"
            params.append(limit)
            
            # Esegui la query
            c.execute(query, params)
            rows = c.fetchall()
            
            # Converti i risultati in dizionari
            performances = []
            for row in rows:
                perf = dict(row)
                
                # Converti i metadati da JSON
                if 'metadata' in perf and perf['metadata']:
                    try:
                        perf['metadata'] = json.loads(perf['metadata'])
                    except:
                        perf['metadata'] = {}
                
                performances.append(perf)
            
            conn.close()
            
            return performances
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero delle performance: {str(e)}")
            return []
    
    def clear_table(self, table_name: str) -> bool:
        """
        Cancella tutti i dati da una tabella
        
        Args:
            table_name: Nome della tabella
            
        Returns:
            True se la cancellazione è riuscita
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Verifica che la tabella esista
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not c.fetchone():
                self.logger.warning(f"Tabella non trovata: {table_name}")
                return False
            
            # Cancella i dati
            c.execute(f"DELETE FROM {table_name}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Tabella cancellata: {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella cancellazione della tabella: {str(e)}")
            return False
    
    def get_db_stats(self) -> Dict[str, int]:
        """
        Ottiene le statistiche del database
        
        Returns:
            Dizionario con le statistiche
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Ottieni il conteggio delle righe per ogni tabella
            tables = ['configurations', 'trades', 'signals', 'positions', 'market_data', 'performance']
            stats = {}
            
            for table in tables:
                c.execute(f"SELECT COUNT(*) FROM {table}")
                count = c.fetchone()[0]
                stats[table] = count
            
            conn.close()
            
            # Aggiungi dimensione del database
            if os.path.exists(self.db_path):
                stats['db_size'] = os.path.getsize(self.db_path)
            else:
                stats['db_size'] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero delle statistiche del database: {str(e)}")
            return {}
