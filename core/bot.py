"""
Classe principale del Trading Bot
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import threading
import time
import os
import signal
import queue
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

from api.bybit_api import BybitAPI
from strategy.technical_strategy import TechnicalStrategy
from strategy.ml_strategy import MLStrategy
from strategy.strategy_base import Signal, SignalType, EntryType
from data.market_data import MarketData
from data.sentiment_analyzer import SentimentAnalyzer
from data.economic_calendar import EconomicCalendar
from risk_management.position_manager import PositionManager
from risk_management.risk_calculator import RiskCalculator
from communication.telegram_client import TelegramClient
from communication.discord_client import DiscordClient
from stats.performance import PerformanceCalculator
from stats.reporting import ReportGenerator
from database.db_manager import DatabaseManager
from database.models import Trade, Position, Signal as SignalModel
from models.ml_models import RandomForestModel
from utils.logger import get_logger, setup_log_capture, log_capture
from utils.helpers import normalize_symbol
from config.settings import (
    BOT_NAME, BOT_MODE, BOT_STATUS, BotMode, BotStatus,
    DEFAULT_TRADING_PAIRS, DEFAULT_TIMEFRAME, AVAILABLE_TIMEFRAMES,
    RISK_PER_TRADE_PERCENT, NOTIFICATION_TRADES, NOTIFICATION_ERRORS,
    NOTIFICATION_PERFORMANCE, PERFORMANCE_REPORT_INTERVAL,
    ENABLE_DISCORD, ENABLE_TELEGRAM, USE_DATABASE
)
from commands.cmd_processor import CommandProcessor
from commands.cmd_handlers import CommandHandlers

logger = get_logger(__name__)

class TradingBot:
    """Classe principale del Trading Bot"""
    
    def __init__(self):
        """Inizializza il Trading Bot"""
        self.logger = get_logger(__name__)
        setup_log_capture()
        
        self.logger.info(f"Inizializzazione {BOT_NAME}...")
        
        # Stato del bot
        self.mode = BOT_MODE
        self.status = BotStatus.STOPPED
        self.start_time = None
        self.stop_event = threading.Event()
        
        # Coppie di trading
        self.trading_pairs = DEFAULT_TRADING_PAIRS.copy()
        
        # API dell'exchange
        self.exchange = BybitAPI(mode=self.mode)
        
        # Moduli principali
        self.market_data = MarketData(self.exchange)
        self.sentiment = SentimentAnalyzer()
        self.economic_calendar = EconomicCalendar()
        self.position_manager = PositionManager(self.exchange)
        self.risk_calculator = RiskCalculator(self.exchange)
        self.performance = PerformanceCalculator()
        self.report_generator = ReportGenerator(self.performance)
        
        # Database
        if USE_DATABASE:
            self.db = DatabaseManager()
        else:
            self.db = None
        
        # Strategie
        self.technical_strategy = TechnicalStrategy(self.exchange, "MainTechnicalStrategy")
        self.ml_strategy = MLStrategy(self.exchange, "MainMLStrategy", self.technical_strategy)
        
        # Comunicazione
        if ENABLE_TELEGRAM:
            self.telegram = TelegramClient()
        else:
            self.telegram = None
            
        if ENABLE_DISCORD:
            self.discord = DiscordClient()
        else:
            self.discord = None
        
        # Gestione dei comandi
        self.cmd_handlers = CommandHandlers(self)
        
        # Thread di elaborazione
        self.main_thread = None
        self.market_thread = None
        self.signals_thread = None
        self.positions_thread = None
        self.report_thread = None
        
        # Code per la comunicazione tra thread
        self.signals_queue = queue.Queue()
        self.orders_queue = queue.Queue()
        
        # Ultimo aggiornamento dei report
        self.last_report_time = datetime.now() - timedelta(days=1)
        
        self.logger.info(f"{BOT_NAME} inizializzato")
    
    def start(self) -> bool:
        """
        Avvia il bot
        
        Returns:
            True se l'avvio è riuscito
        """
        if self.status == BotStatus.RUNNING:
            self.logger.warning("Il bot è già in esecuzione")
            return False
        
        self.logger.info("Avvio del bot...")
        
        try:
            # Imposta lo stato
            self.status = BotStatus.RUNNING
            self.start_time = datetime.now()
            self.stop_event.clear()
            
            # Avvia i thread di elaborazione
            self._start_threads()
            
            # Avvia i servizi di comunicazione
            self._start_communication_services()
            
            # Avvia l'ascolto dei comandi
            if self.telegram:
                self.telegram.register_command_handler("help", lambda args: self.cmd_handlers.process_command("/help " + args))
                self.telegram.register_command_handler("status", lambda args: self.cmd_handlers.process_command("/status " + args))
                self.telegram.register_command_handler("stats", lambda args: self.cmd_handlers.process_command("/stats " + args))
                self.telegram.register_command_handler("positions", lambda args: self.cmd_handlers.process_command("/positions " + args))
                self.telegram.register_command_handler("trades", lambda args: self.cmd_handlers.process_command("/trades " + args))
                self.telegram.register_command_handler("start", lambda args: self.cmd_handlers.process_command("/start " + args))
                self.telegram.register_command_handler("stop", lambda args: self.cmd_handlers.process_command("/stop " + args))
                self.telegram.register_command_handler("scan", lambda args: self.cmd_handlers.process_command("/scan " + args))
                self.telegram.register_command_handler("analyze", lambda args: self.cmd_handlers.process_command("/analyze " + args))
                self.telegram.start_listening()
            
            # Invia notifica di avvio
            self._send_notification(f"🤖 {BOT_NAME} avviato in modalità {self.mode.value.upper()}")
            
            self.logger.info(f"Bot avviato in modalità {self.mode.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'avvio del bot: {str(e)}")
            self.status = BotStatus.ERROR
            self._send_error_notification(f"Errore nell'avvio del bot: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Ferma il bot
        
        Returns:
            True se l'arresto è riuscito
        """
        if self.status == BotStatus.STOPPED:
            self.logger.warning("Il bot è già fermo")
            return False
        
        self.logger.info("Arresto del bot...")
        
        try:
            # Imposta lo stato
            self.status = BotStatus.STOPPED
            self.stop_event.set()
            
            # Ferma i thread di elaborazione
            self._stop_threads()
            
            # Ferma i servizi di comunicazione
            if self.telegram:
                self.telegram.stop_listening()
            
            # Invia notifica di arresto
            self._send_notification(f"🛑 {BOT_NAME} fermato")
            
            self.logger.info("Bot fermato")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'arresto del bot: {str(e)}")
            self.status = BotStatus.ERROR
            self._send_error_notification(f"Errore nell'arresto del bot: {str(e)}")
            return False
    
    def pause(self) -> bool:
        """
        Mette in pausa il bot
        
        Returns:
            True se la pausa è riuscita
        """
        if self.status != BotStatus.RUNNING:
            self.logger.warning(f"Impossibile mettere in pausa il bot: stato corrente {self.status.value}")
            return False
        
        self.logger.info("Messa in pausa del bot...")
        
        try:
            # Imposta lo stato
            self.status = BotStatus.PAUSED
            
            # Invia notifica di pausa
            self._send_notification(f"⏸️ {BOT_NAME} in pausa")
            
            self.logger.info("Bot in pausa")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella messa in pausa del bot: {str(e)}")
            self.status = BotStatus.ERROR
            self._send_error_notification(f"Errore nella messa in pausa del bot: {str(e)}")
            return False
    
    def resume(self) -> bool:
        """
        Riprende l'esecuzione del bot
        
        Returns:
            True se la ripresa è riuscita
        """
        if self.status != BotStatus.PAUSED:
            self.logger.warning(f"Impossibile riprendere il bot: stato corrente {self.status.value}")
            return False
        
        self.logger.info("Ripresa dell'esecuzione del bot...")
        
        try:
            # Imposta lo stato
            self.status = BotStatus.RUNNING
            
            # Invia notifica di ripresa
            self._send_notification(f"▶️ {BOT_NAME} ripreso")
            
            self.logger.info("Bot ripreso")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella ripresa del bot: {str(e)}")
            self.status = BotStatus.ERROR
            self._send_error_notification(f"Errore nella ripresa del bot: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """
        Verifica se il bot è in esecuzione
        
        Returns:
            True se il bot è in esecuzione
        """
        return self.status == BotStatus.RUNNING
    
    def is_paused(self) -> bool:
        """
        Verifica se il bot è in pausa
        
        Returns:
            True se il bot è in pausa
        """
        return self.status == BotStatus.PAUSED
    
    def get_uptime(self) -> str:
        """
        Ottiene l'uptime del bot
        
        Returns:
            Uptime formattato
        """
        if self.start_time is None:
            return "0s"
        
        uptime = datetime.now() - self.start_time
        
        # Formatta l'uptime
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def set_mode(self, mode: BotMode) -> bool:
        """
        Imposta la modalità del bot
        
        Args:
            mode: Nuova modalità
            
        Returns:
            True se l'impostazione è riuscita
        """
        # Verifica se è necessario riavviare il bot
        restart_required = self.is_running()
        
        if restart_required:
            self.stop()
        
        self.mode = mode
        
        # Aggiorna l'API dell'exchange
        self.exchange = BybitAPI(mode=self.mode)
        
        # Aggiorna i moduli che utilizzano l'exchange
        self.market_data = MarketData(self.exchange)
        self.position_manager = PositionManager(self.exchange)
        self.risk_calculator = RiskCalculator(self.exchange)
        self.technical_strategy = TechnicalStrategy(self.exchange, "MainTechnicalStrategy")
        self.ml_strategy = MLStrategy(self.exchange, "MainMLStrategy", self.technical_strategy)
        
        # Salva la configurazione nel database
        if self.db:
            self.db.save_config("bot_mode", mode.value)
        
        self.logger.info(f"Modalità impostata a {mode.value}")
        
        # Riavvia il bot se necessario
        if restart_required:
            self.start()
        
        return True
    
    def add_trading_pair(self, symbol: str) -> bool:
        """
        Aggiunge una coppia di trading
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            True se l'aggiunta è riuscita
        """
        symbol = normalize_symbol(symbol)
        
        if symbol in self.trading_pairs:
            self.logger.warning(f"Coppia {symbol} già presente")
            return False
        
        # Verifica che la coppia esista
        try:
            ticker = self.exchange.get_ticker(symbol)
            if not ticker:
                self.logger.warning(f"Coppia {symbol} non trovata su {self.exchange.__class__.__name__}")
                return False
        except Exception as e:
            self.logger.error(f"Errore nella verifica della coppia {symbol}: {str(e)}")
            return False
        
        # Aggiungi la coppia
        self.trading_pairs.append(symbol)
        
        # Salva la configurazione nel database
        if self.db:
            self.db.save_config("trading_pairs", self.trading_pairs)
        
        self.logger.info(f"Coppia {symbol} aggiunta")
        
        return True
    
    def remove_trading_pair(self, symbol: str) -> bool:
        """
        Rimuove una coppia di trading
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            True se la rimozione è riuscita
        """
        symbol = normalize_symbol(symbol)
        
        if symbol not in self.trading_pairs:
            self.logger.warning(f"Coppia {symbol} non presente")
            return False
        
        # Verifica che non ci siano posizioni aperte
        open_positions = self.get_open_positions()
        for pos in open_positions:
            if pos.get("symbol") == symbol:
                self.logger.warning(f"Impossibile rimuovere la coppia {symbol}: posizione aperta")
                return False
        
        # Rimuovi la coppia
        self.trading_pairs.remove(symbol)
        
        # Salva la configurazione nel database
        if self.db:
            self.db.save_config("trading_pairs", self.trading_pairs)
        
        self.logger.info(f"Coppia {symbol} rimossa")
        
        return True
    
    def set_risk_per_trade(self, risk_percent: float) -> bool:
        """
        Imposta la percentuale di rischio per trade
        
        Args:
            risk_percent: Percentuale di rischio
            
        Returns:
            True se l'impostazione è riuscita
        """
        # Limita il rischio a valori ragionevoli
        risk_percent = max(0.1, min(risk_percent, 5.0))
        
        # Imposta il rischio nel calcolatore
        self.risk_calculator.risk_per_trade_percent = risk_percent
        
        # Salva la configurazione nel database
        if self.db:
            self.db.save_config("risk_per_trade_percent", risk_percent)
        
        self.logger.info(f"Rischio per trade impostato al {risk_percent}%")
        
        return True
    
    def get_wallet_balance(self) -> float:
        """
        Ottiene il saldo del wallet
        
        Returns:
            Saldo del wallet
        """
        try:
            balance = self.exchange.get_wallet_balance()
            return float(balance["totalEquity"])
        except Exception as e:
            self.logger.error(f"Errore nel recupero del saldo: {str(e)}")
            return 0.0
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Ottiene le posizioni aperte
        
        Returns:
            Lista delle posizioni aperte
        """
        try:
            positions = self.position_manager.get_open_positions()
            return [pos.to_dict() for pos in positions]
        except Exception as e:
            self.logger.error(f"Errore nel recupero delle posizioni aperte: {str(e)}")
            return []
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Ottiene gli ordini aperti
        
        Returns:
            Lista degli ordini aperti
        """
        try:
            return self.exchange.get_open_orders()
        except Exception as e:
            self.logger.error(f"Errore nel recupero degli ordini aperti: {str(e)}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """
        Ottiene il prezzo corrente di un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Prezzo corrente
        """
        try:
            ticker = self.exchange.get_ticker(symbol)
            return float(ticker["lastPrice"])
        except Exception as e:
            self.logger.error(f"Errore nel recupero del prezzo di {symbol}: {str(e)}")
            return 0.0
    
    def close_position(self, symbol: str) -> bool:
        """
        Chiude una posizione
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            True se la chiusura è riuscita
        """
        symbol = normalize_symbol(symbol)
        
        try:
            # Cerca la posizione
            open_positions = self.get_open_positions()
            position = None
            
            for pos in open_positions:
                if pos.get("symbol") == symbol:
                    position = pos
                    break
            
            if not position:
                self.logger.warning(f"Nessuna posizione aperta per {symbol}")
                return False
            
            # Ottieni il prezzo corrente
            current_price = self.get_current_price(symbol)
            
            if current_price <= 0:
                self.logger.error(f"Prezzo non valido per {symbol}: {current_price}")
                return False
            
            # Chiudi la posizione
            result = self.position_manager.close_position(position["id"], current_price)
            
            if result:
                self.logger.info(f"Posizione {symbol} chiusa a {current_price}")
                
                # Invia notifica
                self._send_trade_notification({
                    "symbol": symbol,
                    "action": "close",
                    "price": current_price,
                    "size": position.get("size", 0.0),
                    "pnl": position.get("unrealized_pnl", 0.0)
                })
                
                return True
            else:
                self.logger.error(f"Errore nella chiusura della posizione {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nella chiusura della posizione {symbol}: {str(e)}")
            self._send_error_notification(f"Errore nella chiusura della posizione {symbol}: {str(e)}")
            return False
    
    def execute_manual_trade(self, symbol: str, side: str, size: float, 
                           price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Esegue un trade manuale
        
        Args:
            symbol: Simbolo della coppia
            side: Lato del trade (buy/sell/long/short)
            size: Dimensione del trade
            price: Prezzo (opzionale, se None usa il prezzo di mercato)
            
        Returns:
            Risultato del trade o None in caso di errore
        """
        symbol = normalize_symbol(symbol)
        side = side.lower()
        
        try:
            # Converti side in formato compatibile con l'exchange
            if side in ["buy", "long"]:
                exchange_side = "Buy"
                entry_type = "long"
            elif side in ["sell", "short"]:
                exchange_side = "Sell"
                entry_type = "short"
            else:
                self.logger.error(f"Lato non valido: {side}")
                return None
            
            # Se il prezzo non è specificato, ottieni il prezzo corrente
            if price is None:
                price = self.get_current_price(symbol)
                
                if price <= 0:
                    self.logger.error(f"Prezzo non valido per {symbol}: {price}")
                    return None
            
            # Tipo di ordine
            order_type = "Limit" if price else "Market"
            
            # Calcola lo stop loss
            stop_loss = self.risk_calculator.calculate_stop_loss(
                symbol, price, entry_type
            )
            
            # Calcola i take profit
            take_profits = self.risk_calculator.calculate_take_profits(
                symbol, price, entry_type, stop_loss
            )
            
            # Esegui l'ordine
            result = self.exchange.place_order(
                symbol=symbol,
                side=exchange_side,
                order_type=order_type,
                qty=size,
                price=price if order_type == "Limit" else None,
                stop_loss=stop_loss,
                take_profit=take_profits[0] if take_profits else None
            )
            
            if result:
                self.logger.info(f"Trade manuale eseguito: {symbol} {side} {size} a {price}")
                
                # Crea una posizione
                position = Position(
                    symbol=symbol,
                    entry_type=entry_type,
                    entry_price=price,
                    size=size,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    entry_time=datetime.now().isoformat(),
                    status="open"
                )
                
                # Aggiungi la posizione al gestore
                self.position_manager.positions[position.id] = position
                
                # Salva la posizione nel database
                if self.db:
                    self.db.save_position(position.to_dict())
                
                # Invia notifica
                self._send_trade_notification({
                    "symbol": symbol,
                    "action": side,
                    "price": price,
                    "size": size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profits
                })
                
                return result
            else:
                self.logger.error(f"Errore nell'esecuzione del trade manuale: {symbol} {side} {size}")
                return None
                
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione del trade manuale: {str(e)}")
            self._send_error_notification(f"Errore nell'esecuzione del trade manuale: {str(e)}")
            return None
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Analizza un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Risultato dell'analisi
        """
        symbol = normalize_symbol(symbol)
        
        try:
            # Ottieni i dati di mercato
            data = self.market_data.get_data_with_indicators(symbol)
            
            if data.empty:
                self.logger.warning(f"Nessun dato disponibile per {symbol}")
                return {}
            
            # Analizza con la strategia ML
            signal = self.ml_strategy.analyze(symbol)
            
            # Ottieni il sentiment
            sentiment = self.sentiment.get_sentiment(symbol)
            
            # Ottieni il prezzo corrente
            current_price = self.get_current_price(symbol)
            
            # Componi il risultato
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "signal": signal.signal_type.value,
                "signal_strength": signal.strength,
                "entry_type": signal.entry_type.value if signal.entry_type else None,
                "stop_loss": signal.stop_loss,
                "take_profits": signal.take_profits,
                "reason": signal.reason,
                "timestamp": datetime.now().isoformat()
            }
            
            # Aggiungi gli indicatori tecnici
            if not data.empty:
                last_row = data.iloc[-1]
                
                result.update({
                    "rsi": last_row.get("rsi", 0.0),
                    "macd": last_row.get("macd", 0.0),
                    "macd_signal": last_row.get("macd_signal", 0.0),
                    "ema_20": last_row.get("ema_20", 0.0),
                    "ema_50": last_row.get("ema_50", 0.0),
                    "ema_200": last_row.get("ema_200", 0.0),
                    "bb_upper": last_row.get("bb_upper", 0.0),
                    "bb_lower": last_row.get("bb_lower", 0.0),
                    "atr": last_row.get("atr", 0.0)
                })
            
            # Aggiungi il sentiment
            if sentiment:
                result.update({
                    "sentiment": sentiment.get("sentiment", "neutral"),
                    "sentiment_score": sentiment.get("score", 0.0)
                })
            
            # Aggiungi la variazione di prezzo
            try:
                if len(data) > 1:
                    prev_close = data.iloc[-2]["close"]
                    price_change = (current_price - prev_close) / prev_close * 100
                    result["price_change_24h"] = price_change
            except:
                pass
            
            # Calcola supporti e resistenze
            try:
                window = 50
                if len(data) > window:
                    data_window = data.iloc[-window:]
                    
                    # Trova i minimi locali (supporti)
                    supports = []
                    for i in range(1, len(data_window) - 1):
                        if (data_window.iloc[i]["low"] <= data_window.iloc[i-1]["low"] and 
                            data_window.iloc[i]["low"] <= data_window.iloc[i+1]["low"]):
                            supports.append(data_window.iloc[i]["low"])
                    
                    # Trova i massimi locali (resistenze)
                    resistances = []
                    for i in range(1, len(data_window) - 1):
                        if (data_window.iloc[i]["high"] >= data_window.iloc[i-1]["high"] and 
                            data_window.iloc[i]["high"] >= data_window.iloc[i+1]["high"]):
                            resistances.append(data_window.iloc[i]["high"])
                    
                    # Trova il supporto e la resistenza più vicini
                    if supports:
                        supports = [s for s in supports if s < current_price]
                        if supports:
                            result["next_support"] = max(supports)
                    
                    if resistances:
                        resistances = [r for r in resistances if r > current_price]
                        if resistances:
                            result["next_resistance"] = min(resistances)
            except:
                pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi di {symbol}: {str(e)}")
            return {}
    
    def scan_for_opportunities(self) -> List[Dict[str, Any]]:
        """
        Scansiona tutte le coppie per opportunità
        
        Returns:
            Lista di opportunità
        """
        opportunities = []
        
        try:
            for symbol in self.trading_pairs:
                try:
                    # Analizza il simbolo
                    analysis = self.analyze_symbol(symbol)
                    
                    if not analysis:
                        continue
                    
                    # Verifica se c'è un segnale forte
                    signal = analysis.get("signal", "hold")
                    strength = analysis.get("signal_strength", 0.0)
                    
                    if signal.lower() != "hold" and strength > 0.5:
                        opportunities.append(analysis)
                        
                except Exception as e:
                    self.logger.error(f"Errore nell'analisi di {symbol}: {str(e)}")
            
            # Ordina per forza del segnale
            opportunities.sort(key=lambda x: x.get("signal_strength", 0.0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Errore nella scansione delle opportunità: {str(e)}")
            return []
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Genera un report completo
        
        Returns:
            Risultato della generazione del report
        """
        try:
            # Genera il report
            report = self.report_generator.generate_periodic_report()
            
            # Invia notifica
            if report.get("chart_paths"):
                for chart_path in report["chart_paths"]:
                    self._send_chart(chart_path)
            
            # Invia un riassunto via messaggio
            stats = report.get("report", {}).get("overall_stats", {})
            if stats:
                self._send_performance_notification(stats)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del report: {str(e)}")
            self._send_error_notification(f"Errore nella generazione del report: {str(e)}")
            return {}
    
    def generate_chart(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> Optional[str]:
        """
        Genera un grafico per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            timeframe: Intervallo temporale
            
        Returns:
            Percorso del grafico generato o None in caso di errore
        """
        symbol = normalize_symbol(symbol)
        
        try:
            # Verifica che il timeframe sia valido
            if timeframe not in AVAILABLE_TIMEFRAMES:
                self.logger.error(f"Timeframe non valido: {timeframe}")
                return None
            
            # Ottieni i dati di mercato
            data = self.market_data.get_data_with_indicators(symbol, timeframe)
            
            if data.empty:
                self.logger.warning(f"Nessun dato disponibile per {symbol} {timeframe}")
                return None
            
            # Genera il grafico
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter
            import mplfinance as mpf
            
            # Crea la directory per i grafici se non esiste
            os.makedirs("reports/charts", exist_ok=True)
            
            # Nome del file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"reports/charts/{symbol.replace('/', '_')}_{timeframe}_{timestamp}.png"
            
            # Converti i dati in formato OHLCV per mplfinance
            ohlc_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Aggiorna l'indice se necessario
            if not isinstance(ohlc_data.index, pd.DatetimeIndex):
                ohlc_data.index = pd.to_datetime(ohlc_data.index)
            
            # Crea il grafico
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Grafico OHLC
            mpf.plot(ohlc_data, type='candle', style='charles',
                    title=f"{symbol} - {timeframe}",
                    ylabel='Price',
                    ax=ax1, volume=False)
            
            # Aggiungi le medie mobili
            if 'ema_20' in data.columns:
                ax1.plot(ohlc_data.index, data['ema_20'], label='EMA 20', color='blue')
            if 'ema_50' in data.columns:
                ax1.plot(ohlc_data.index, data['ema_50'], label='EMA 50', color='green')
            if 'ema_200' in data.columns:
                ax1.plot(ohlc_data.index, data['ema_200'], label='EMA 200', color='red')
            
            # Aggiungi le bande di Bollinger
            if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
                ax1.plot(ohlc_data.index, data['bb_upper'], label='BB Upper', color='purple', alpha=0.7)
                ax1.plot(ohlc_data.index, data['bb_lower'], label='BB Lower', color='purple', alpha=0.7)
            
            # Aggiungi la legenda
            ax1.legend()
            
            # Grafico del volume
            mpf.plot(ohlc_data, type='candle', style='charles',
                    ylabel='Volume',
                    ax=ax2, volume=True, show_nontrading=False)
            
            # Aggiungi l'RSI se disponibile
            if 'rsi' in data.columns:
                ax3 = ax2.twinx()
                ax3.plot(ohlc_data.index, data['rsi'], label='RSI', color='orange')
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                ax3.set_ylabel('RSI')
                ax3.legend(loc='upper right')
            
            # Formatta l'asse x
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # Ruota le etichette dell'asse x
            plt.xticks(rotation=45)
            
            # Aggiungi il titolo generale
            fig.suptitle(f"{symbol} - {timeframe} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=16)
            
            # Aggiungi padding
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            # Salva il grafico
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Grafico generato: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione del grafico per {symbol}: {str(e)}")
            return None
    
    def get_logs(self, limit: int = 100) -> List[str]:
        """
        Ottiene gli ultimi log
        
        Args:
            limit: Numero massimo di log da restituire
            
        Returns:
            Lista di log
        """
        return log_capture.get_logs(limit)
    
    def _start_threads(self) -> None:
        """Avvia i thread di elaborazione"""
        # Thread principale
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        # Thread per i dati di mercato
        self.market_thread = threading.Thread(target=self._market_data_loop)
        self.market_thread.daemon = True
        self.market_thread.start()
        
        # Thread per i segnali
        self.signals_thread = threading.Thread(target=self._signals_loop)
        self.signals_thread.daemon = True
        self.signals_thread.start()
        
        # Thread per le posizioni
        self.positions_thread = threading.Thread(target=self._positions_loop)
        self.positions_thread.daemon = True
        self.positions_thread.start()
        
        # Thread per i report
        self.report_thread = threading.Thread(target=self._report_loop)
        self.report_thread.daemon = True
        self.report_thread.start()
        
        self.logger.info("Thread di elaborazione avviati")
    
    def _stop_threads(self) -> None:
        """Ferma i thread di elaborazione"""
        self.stop_event.set()
        
        # Attendi che i thread terminino
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
            
        if self.market_thread and self.market_thread.is_alive():
            self.market_thread.join(timeout=5.0)
            
        if self.signals_thread and self.signals_thread.is_alive():
            self.signals_thread.join(timeout=5.0)
            
        if self.positions_thread and self.positions_thread.is_alive():
            self.positions_thread.join(timeout=5.0)
            
        if self.report_thread and self.report_thread.is_alive():
            self.report_thread.join(timeout=5.0)
            
        self.logger.info("Thread di elaborazione fermati")
    
    def _start_communication_services(self) -> None:
        """Avvia i servizi di comunicazione"""
        if self.telegram:
            self.telegram.test_connection()
            
        if self.discord:
            self.discord.test_connection()
    
    def _send_notification(self, message: str) -> None:
        """
        Invia una notifica
        
        Args:
            message: Messaggio da inviare
        """
        if self.telegram:
            self.telegram.send_message(message)
            
        if self.discord:
            self.discord.send_message(message, f"{BOT_NAME} Notification")
            
        self.logger.info(f"Notifica inviata: {message}")
    
    def _send_error_notification(self, error_message: str) -> None:
        """
        Invia una notifica di errore
        
        Args:
            error_message: Messaggio di errore
        """
        if not NOTIFICATION_ERRORS:
            return
            
        if self.telegram:
            self.telegram.send_error_notification(error_message)
            
        if self.discord:
            self.discord.send_error_notification(error_message)
            
        self.logger.info(f"Notifica di errore inviata: {error_message}")
    
    def _send_trade_notification(self, trade_data: Dict[str, Any]) -> None:
        """
        Invia una notifica di trade
        
        Args:
            trade_data: Dati del trade
        """
        if not NOTIFICATION_TRADES:
            return
            
        if self.telegram:
            self.telegram.send_trade_notification(trade_data)
            
        if self.discord:
            self.discord.send_trade_notification(trade_data)
            
        self.logger.info(f"Notifica di trade inviata: {trade_data.get('symbol')} {trade_data.get('action')}")
    
    def _send_performance_notification(self, performance_data: Dict[str, Any]) -> None:
        """
        Invia una notifica di performance
        
        Args:
            performance_data: Dati di performance
        """
        if not NOTIFICATION_PERFORMANCE:
            return
            
        if self.telegram:
            self.telegram.send_performance_report(performance_data)
            
        if self.discord:
            self.discord.send_performance_report(performance_data)
            
        self.logger.info("Notifica di performance inviata")
    
    def _send_chart(self, chart_path: str, caption: str = "") -> None:
        """
        Invia un grafico
        
        Args:
            chart_path: Percorso del grafico
            caption: Didascalia del grafico
        """
        if not os.path.exists(chart_path):
            self.logger.error(f"Grafico non trovato: {chart_path}")
            return
            
        if self.telegram:
            self.telegram.send_chart(chart_path, caption)
            
        if self.discord:
            self.discord.send_chart(chart_path, caption)
            
        self.logger.info(f"Grafico inviato: {chart_path}")
    
    def _main_loop(self) -> None:
        """Loop principale del bot"""
        self.logger.info("Loop principale avviato")
        
        while not self.stop_event.is_set():
            try:
                # Verifica lo stato del bot
                if self.status != BotStatus.RUNNING:
                    time.sleep(1.0)
                    continue
                
                # Verifica se è necessario passare da demo a live
                if self.mode == BotMode.DEMO and self.performance.demo_to_live_ready:
                    self.logger.info("Condizioni soddisfatte per passare a live trading")
                    self._send_notification("🚀 Condizioni soddisfatte per passare a live trading! "
                                          f"Win rate: {self.performance.get_overall_stats()['win_rate'] * 100:.2f}%")
                
                # Sleep per non sovraccaricare la CPU
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop principale: {str(e)}")
                self._send_error_notification(f"Errore nel loop principale: {str(e)}")
                time.sleep(5.0)
    
    def _market_data_loop(self) -> None:
        """Loop per l'aggiornamento dei dati di mercato"""
        self.logger.info("Loop dei dati di mercato avviato")
        
        # Avvia gli aggiornamenti in background
        self.market_data.start_background_updates()
        
        while not self.stop_event.is_set():
            try:
                # Verifica lo stato del bot
                if self.status != BotStatus.RUNNING:
                    time.sleep(1.0)
                    continue
                
                # Sleep per non sovraccaricare l'API
                time.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop dei dati di mercato: {str(e)}")
                time.sleep(5.0)
        
        # Ferma gli aggiornamenti in background
        self.market_data.stop_background_updates()
    
    def _signals_loop(self) -> None:
        """Loop per la generazione dei segnali"""
        self.logger.info("Loop dei segnali avviato")
        
        while not self.stop_event.is_set():
            try:
                # Verifica lo stato del bot
                if self.status != BotStatus.RUNNING:
                    time.sleep(1.0)
                    continue
                
                # Genera segnali per tutte le coppie
                for symbol in self.trading_pairs:
                    try:
                        # Ottieni i dati di mercato
                        data = self.market_data.get_data_with_indicators(symbol)
                        
                        if data.empty:
                            continue
                        
                        # Genera un segnale con la strategia ML
                        signal = self.ml_strategy.analyze(symbol)
                        
                        # Verifica se il segnale è abbastanza forte
                        if signal.signal_type != SignalType.HOLD and signal.strength > 0.7:
                            # Verifica se il trade è consentito
                            open_positions = self.get_open_positions()
                            allowed, reason = self.risk_calculator.is_trade_allowed(
                                symbol, signal.entry_type.value if signal.entry_type else "", open_positions
                            )
                            
                            if allowed:
                                # Metti il segnale nella coda
                                self.signals_queue.put(signal)
                                
                                # Registra il segnale nel database
                                if self.db:
                                    signal_model = SignalModel(
                                        symbol=signal.symbol,
                                        signal_type=signal.signal_type.value,
                                        price=signal.price,
                                        strength=signal.strength,
                                        entry_type=signal.entry_type.value if signal.entry_type else None,
                                        stop_loss=signal.stop_loss,
                                        take_profits=signal.take_profits,
                                        reason=signal.reason,
                                        timestamp=datetime.now().isoformat()
                                    )
                                    self.db.save_signal(signal_model.to_dict())
                                
                                self.logger.info(f"Segnale generato: {symbol} {signal.signal_type.value} "
                                               f"(forza: {signal.strength:.2f})")
                            else:
                                self.logger.info(f"Segnale ignorato: {symbol} {signal.signal_type.value} - {reason}")
                        
                    except Exception as e:
                        self.logger.error(f"Errore nella generazione del segnale per {symbol}: {str(e)}")
                
                # Sleep per non sovraccaricare la CPU
                time.sleep(60.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop dei segnali: {str(e)}")
                time.sleep(5.0)
    
    def _positions_loop(self) -> None:
        """Loop per la gestione delle posizioni"""
        self.logger.info("Loop delle posizioni avviato")
        
        while not self.stop_event.is_set():
            try:
                # Verifica lo stato del bot
                if self.status != BotStatus.RUNNING:
                    time.sleep(1.0)
                    continue
                
                # Elabora i segnali dalla coda
                while not self.signals_queue.empty():
                    signal = self.signals_queue.get()
                    
                    # Apri una posizione in base al segnale
                    position = self.position_manager.open_position(signal)
                    
                    if position:
                        # Registra la posizione nel database
                        if self.db:
                            self.db.save_position(position.to_dict())
                        
                        # Invia notifica
                        self._send_trade_notification({
                            "symbol": position.symbol,
                            "action": position.entry_type,
                            "price": position.entry_price,
                            "size": position.size,
                            "stop_loss": position.stop_loss,
                            "take_profit": position.take_profits
                        })
                        
                        self.logger.info(f"Posizione aperta: {position.symbol} {position.entry_type} "
                                       f"a {position.entry_price}")
                
                # Aggiorna le posizioni esistenti
                self.position_manager.update_positions()
                
                # Sleep per non sovraccaricare la CPU
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop delle posizioni: {str(e)}")
                time.sleep(5.0)
    
    def _report_loop(self) -> None:
        """Loop per la generazione dei report"""
        self.logger.info("Loop dei report avviato")
        
        while not self.stop_event.is_set():
            try:
                # Verifica se è il momento di generare un report
                now = datetime.now()
                
                if (now - self.last_report_time).total_seconds() > PERFORMANCE_REPORT_INTERVAL:
                    self.logger.info("Generazione del report periodico...")
                    
                    # Genera il report
                    self.generate_report()
                    
                    # Aggiorna il timestamp
                    self.last_report_time = now
                
                # Sleep per non sovraccaricare la CPU
                time.sleep(60.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop dei report: {str(e)}")
                time.sleep(5.0)
