"""
Configurazioni globali per il trading bot
"""
import os
from enum import Enum
from datetime import datetime

# Modalità di esecuzione
class BotMode(Enum):
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"

# Status bot
class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

# Configurazioni generali
BOT_NAME = "CryptoTradingBot"
BOT_VERSION = "1.0.0"
BOT_MODE = BotMode.DEMO  # Inizia in modalità demo
BOT_STATUS = BotStatus.STOPPED

# Configurazioni delle performance per passare a mainnet
PROFIT_THRESHOLD_PERCENT = 85.0  # Percentuale di profitto per passare a mainnet
MIN_TRADES_FOR_EVALUATION = 100  # Numero minimo di trade per valutare le performance

# Configurazioni di trading
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 5
DEFAULT_MARGIN_MODE = "isolated"  # isolated o cross
MAX_CONCURRENT_TRADES = 3
MAX_POSITION_SIZE_PERCENT = 5.0  # Percentuale massima del capitale per trade

# Gestione del rischio
RISK_PER_TRADE_PERCENT = 1.0
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_LEVELS = [
    {"percent": 1.5, "size_percent": 40},  # 40% della posizione al +1.5%
    {"percent": 3.0, "size_percent": 30},  # 30% della posizione al +3.0%
    {"percent": 5.0, "size_percent": 30}   # 30% della posizione al +5.0%
]
TRAILING_STOP_ACTIVATION = 2.0  # % di profitto per attivare trailing stop
TRAILING_STOP_DISTANCE = 1.0    # Distanza del trailing stop in %

# Intervalli di tempo
DEFAULT_TIMEFRAME = "15m"  # Intervallo predefinito per l'analisi
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
DATA_LOOKBACK_PERIODS = 500  # Numero di candle da recuperare per l'analisi

# Coppie trading predefinite
DEFAULT_TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", 
    "BNB/USDT", "XRP/USDT", "ADA/USDT"
]

# Parametri AI
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_FREQUENCY = 50  # Ogni quanti trade riaddestra il modello
MODEL_SAVE_PATH = os.path.join("models", "saved")
FEATURE_WINDOW = 30  # Numero di candle da utilizzare come features

# Sentiment analysis
SENTIMENT_UPDATE_INTERVAL = 3600  # Aggiornamento sentiment ogni ora (in secondi)
SENTIMENT_WEIGHT = 0.3  # Peso del sentiment nel processo decisionale

# Economic calendar
ECONOMIC_CALENDAR_UPDATE_INTERVAL = 86400  # Aggiornamento dati economici ogni giorno

# Configurazioni di logging
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join("logs", f"{BOT_NAME.lower()}_{datetime.now().strftime('%Y%m%d')}.log")

# Configurazioni dei database
DATABASE_PATH = os.path.join("database", "trading_bot.db")
USE_DATABASE = True

# Configurazioni comunicazione
ENABLE_DISCORD = True
ENABLE_TELEGRAM = True
NOTIFICATION_TRADES = True
NOTIFICATION_ERRORS = True
NOTIFICATION_PERFORMANCE = True
PERFORMANCE_REPORT_INTERVAL = 86400  # Report quotidiano (in secondi)

# Configurazioni per backtest
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE = datetime.now().strftime("%Y-%m-%d")