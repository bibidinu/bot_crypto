"""
Modulo per la gestione dei log del trading bot
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
import traceback
import threading
from typing import Optional

# Importazione delle configurazioni
try:
    from config.settings import LOG_LEVEL, LOG_FILE, BOT_NAME
except ImportError:
    # Valori di default se il modulo config non è disponibile
    LOG_LEVEL = "INFO"
    BOT_NAME = "TradingBot"
    LOG_FILE = os.path.join("logs", f"{BOT_NAME.lower()}_{datetime.now().strftime('%Y%m%d')}.log")

# Thread-safe singleton per il logger
_logger_lock = threading.Lock()
_logger_instances = {}

def get_logger(name: str = None) -> logging.Logger:
    """
    Ottiene un'istanza del logger
    
    Args:
        name: Nome del logger (opzionale)
        
    Returns:
        Istanza del logger
    """
    global _logger_instances
    
    # Usa il nome del bot come default
    if name is None:
        name = BOT_NAME
    
    # Thread-safe singleton pattern
    with _logger_lock:
        if name in _logger_instances:
            return _logger_instances[name]
        
        # Crea una nuova istanza del logger
        logger = _create_logger(name)
        _logger_instances[name] = logger
        
        return logger

def _create_logger(name: str) -> logging.Logger:
    """
    Crea una nuova istanza del logger
    
    Args:
        name: Nome del logger
        
    Returns:
        Istanza del logger configurata
    """
    # Crea la directory dei log se non esiste
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Crea il logger
    logger = logging.getLogger(name)
    
    # Livello di log da impostare
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(LOG_LEVEL, logging.INFO))
    
    # Formattatore per i log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler per i file di log con rotazione
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level_map.get(LOG_LEVEL, logging.INFO))
    
    # Handler per la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level_map.get(LOG_LEVEL, logging.INFO))
    
    # Rimuovi tutti gli handler esistenti
    if logger.handlers:
        logger.handlers = []
    
    # Aggiungi gli handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Disabilita la propagazione per evitare duplicati
    logger.propagate = False
    
    return logger

def log_exception(logger: logging.Logger, exc_info=None) -> None:
    """
    Logga un'eccezione con il traceback completo
    
    Args:
        logger: Istanza del logger
        exc_info: Informazioni sull'eccezione (opzionale)
    """
    if exc_info is None:
        exc_info = sys.exc_info()
    
    if exc_info[0] is not None:
        logger.error(
            f"Exception: {exc_info[1]}",
            exc_info=exc_info
        )

class LogCapture:
    """Classe per catturare i log in una lista"""
    
    def __init__(self, max_logs: int = 1000):
        """
        Inizializza il catturatore di log
        
        Args:
            max_logs: Numero massimo di log da memorizzare
        """
        self.logs = []
        self.max_logs = max_logs
        self.lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Aggiunge un record di log alla lista
        
        Args:
            record: Record di log
        """
        with self.lock:
            # Formatta il log
            log_entry = f"{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')} - {record.levelname} - {record.getMessage()}"
            
            # Aggiungi alla lista
            self.logs.append(log_entry)
            
            # Mantieni la dimensione massima
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)
    
    def get_logs(self, limit: int = None) -> list:
        """
        Ottiene i log memorizzati
        
        Args:
            limit: Numero massimo di log da restituire (opzionale)
            
        Returns:
            Lista dei log
        """
        with self.lock:
            if limit is None or limit > len(self.logs):
                return self.logs.copy()
            
            return self.logs[-limit:]

# Crea un'istanza globale del catturatore di log
log_capture = LogCapture()

def setup_log_capture(logger_name: str = None) -> None:
    """
    Configura il catturatore di log per un logger specifico
    
    Args:
        logger_name: Nome del logger (opzionale)
    """
    logger = get_logger(logger_name)
    
    # Crea un handler personalizzato
    class CaptureHandler(logging.Handler):
        def emit(self, record):
            log_capture.emit(record)
    
    # Aggiungi l'handler al logger
    handler = CaptureHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
