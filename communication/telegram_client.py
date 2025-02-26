"""
Modulo per l'integrazione con Telegram
"""
from typing import Dict, List, Optional, Any, Callable
import time
import threading
import requests
import json
import re
from datetime import datetime

from utils.logger import get_logger
from config.credentials import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from config.settings import (
    BOT_NAME, NOTIFICATION_TRADES, NOTIFICATION_ERRORS, 
    NOTIFICATION_PERFORMANCE
)

logger = get_logger(__name__)

class TelegramClient:
    """Classe per l'integrazione con Telegram"""
    
    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        """
        Inizializza il client Telegram
        
        Args:
            token: Token del bot Telegram
            chat_id: ID della chat a cui inviare i messaggi
        """
        self.logger = get_logger(__name__)
        
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        
        # Thread per l'ascolto dei comandi
        self.listen_thread = None
        self.running = False
        
        # Ultima data di aggiornamento ricevuta
        self.last_update_id = 0
        
        # Callback per i comandi
        self.command_handlers: Dict[str, Callable] = {}
        
        # Prefisso per i log
        self.log_prefix = f"[{BOT_NAME}] "
        
        self.logger.info("TelegramClient inizializzato")
    
    def test_connection(self) -> bool:
        """
        Testa la connessione a Telegram
        
        Returns:
            True se la connessione è riuscita
        """
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url)
            data = response.json()
            
            if data.get("ok"):
                self.logger.info(f"Connessione a Telegram riuscita: {data.get('result', {}).get('username')}")
                return True
            else:
                self.logger.error(f"Errore nella connessione a Telegram: {data.get('description')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nel test di connessione a Telegram: {str(e)}")
            return False
    
    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Invia un messaggio a Telegram
        
        Args:
            text: Testo del messaggio
            parse_mode: Modalità di parsing ("Markdown" o "HTML")
            
        Returns:
            True se l'invio è riuscito
        """
        if not self.token or not self.chat_id:
            self.logger.warning("Token o chat_id non configurati")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            
            # Aggiungi il prefisso al testo
            text = self.log_prefix + text
            
            # Limita la lunghezza del messaggio a 4096 caratteri
            if len(text) > 4096:
                text = text[:4093] + "..."
            
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload)
            data = response.json()
            
            if data.get("ok"):
                return True
            else:
                self.logger.error(f"Errore nell'invio del messaggio a Telegram: {data.get('description')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio del messaggio a Telegram: {str(e)}")
            return False
    
    def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Invia una notifica di trade a Telegram
        
        Args:
            trade_data: Dati del trade
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_TRADES:
            return False
            
        try:
            symbol = trade_data.get("symbol", "Unknown")
            action = trade_data.get("action", "Unknown")
            price = trade_data.get("price", 0.0)
            size = trade_data.get("size", 0.0)
            pnl = trade_data.get("pnl", 0.0)
            
            emoji = "🟢" if action.lower() in ["buy", "long"] else "🔴" if action.lower() in ["sell", "short"] else "⚪"
            
            # Componi il messaggio
            message = f"{emoji} *{action.upper()} {symbol}*\n"
            message += f"Price: `{price:.5f}`\n"
            message += f"Size: `{size:.5f}`\n"
            
            if "stop_loss" in trade_data:
                message += f"Stop Loss: `{trade_data['stop_loss']:.5f}`\n"
                
            if "take_profit" in trade_data:
                if isinstance(trade_data["take_profit"], list):
                    for i, tp in enumerate(trade_data["take_profit"]):
                        message += f"TP{i+1}: `{tp:.5f}`\n"
                else:
                    message += f"Take Profit: `{trade_data['take_profit']:.5f}`\n"
            
            if pnl != 0.0:
                emoji_pnl = "✅" if pnl > 0 else "❌"
                message += f"{emoji_pnl} PnL: `{pnl:.2f}` ({trade_data.get('pnl_percent', 0.0):.2f}%)\n"
            
            if "reason" in trade_data:
                message += f"\nReason: _{trade_data['reason']}_"
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio della notifica di trade: {str(e)}")
            return False
    
    def send_error_notification(self, error_message: str) -> bool:
        """
        Invia una notifica di errore a Telegram
        
        Args:
            error_message: Messaggio di errore
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_ERRORS:
            return False
            
        try:
            message = f"⚠️ *ERROR*\n"
            message += f"{error_message}\n"
            message += f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio della notifica di errore: {str(e)}")
            return False
    
    def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Invia un report di performance a Telegram
        
        Args:
            performance_data: Dati di performance
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_PERFORMANCE:
            return False
            
        try:
            total_profit = performance_data.get("total_profit", 0.0)
            win_rate = performance_data.get("win_rate", 0.0) * 100
            total_trades = performance_data.get("total_trades", 0)
            
            emoji = "📈" if total_profit > 0 else "📉"
            
            # Componi il messaggio
            message = f"{emoji} *PERFORMANCE REPORT*\n\n"
            message += f"Total Profit: `{total_profit:.2f} USDT`\n"
            message += f"Win Rate: `{win_rate:.2f}%`\n"
            message += f"Total Trades: `{total_trades}`\n"
            
            if "winning_trades" in performance_data and "losing_trades" in performance_data:
                message += f"Winning/Losing: `{performance_data['winning_trades']}/{performance_data['losing_trades']}`\n"
            
            if "profit_factor" in performance_data:
                message += f"Profit Factor: `{performance_data['profit_factor']:.2f}`\n"
            
            if "avg_win" in performance_data and "avg_loss" in performance_data:
                message += f"Avg Win/Loss: `{performance_data['avg_win']:.2f}/{performance_data['avg_loss']:.2f}`\n"
            
            if "open_positions" in performance_data:
                message += f"Open Positions: `{performance_data['open_positions']}`\n"
            
            message += f"\nTime: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio del report di performance: {str(e)}")
            return False
    
    def register_command_handler(self, command: str, handler: Callable) -> None:
        """
        Registra un handler per un comando
        
        Args:
            command: Comando (senza /)
            handler: Funzione da chiamare quando il comando viene ricevuto
        """
        self.command_handlers[command.lower()] = handler
        self.logger.info(f"Handler registrato per il comando /{command}")
    
    def start_listening(self) -> None:
        """
        Avvia il thread di ascolto dei comandi
        """
        if self.listen_thread is not None and self.listen_thread.is_alive():
            self.logger.warning("Thread di ascolto già in esecuzione")
            return
            
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_for_commands)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        self.logger.info("Thread di ascolto avviato")
    
    def stop_listening(self) -> None:
        """
        Ferma il thread di ascolto
        """
        self.running = False
        
        if self.listen_thread is not None:
            try:
                self.listen_thread.join(timeout=5.0)
            except:
                pass
                
        self.logger.info("Thread di ascolto fermato")
    
    def _listen_for_commands(self) -> None:
        """Thread worker per l'ascolto dei comandi"""
        self.logger.info("Worker di ascolto avviato")
        
        # Invia un messaggio di avvio
        self.send_message("🤖 *Bot avviato*\nIn ascolto per comandi...")
        
        while self.running:
            try:
                # Ottieni gli aggiornamenti
                updates = self._get_updates()
                
                # Elabora gli aggiornamenti
                for update in updates:
                    # Aggiorna l'ID dell'ultimo aggiornamento
                    if update.get("update_id", 0) > self.last_update_id:
                        self.last_update_id = update.get("update_id", 0)
                    
                    # Elabora il messaggio
                    message = update.get("message", {})
                    if not message:
                        continue
                        
                    text = message.get("text", "")
                    chat_id = message.get("chat", {}).get("id")
                    
                    # Se il messaggio non è per la chat configurata, ignoralo
                    if str(chat_id) != str(self.chat_id):
                        continue
                    
                    # Verifica se è un comando
                    if text.startswith("/"):
                        # Estrai il comando
                        match = re.match(r"^/([a-zA-Z0-9_]+)(?:@\w+)?(?:\s+(.*))?$", text)
                        if match:
                            command = match.group(1).lower()
                            args = match.group(2) if match.group(2) else ""
                            
                            self.logger.info(f"Comando ricevuto: /{command} {args}")
                            
                            # Chiama l'handler appropriato
                            if command in self.command_handlers:
                                try:
                                    self.command_handlers[command](args)
                                except Exception as e:
                                    self.logger.error(f"Errore nell'handler del comando /{command}: {str(e)}")
                                    self.send_message(f"⚠️ Errore nell'esecuzione del comando /{command}: {str(e)}")
                            else:
                                self.send_message(f"❓ Comando non riconosciuto: /{command}")
                
                # Sleep per non sovraccaricare l'API
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel thread di ascolto: {str(e)}")
                time.sleep(30.0)  # Sleep più lungo in caso di errore
    
    def _get_updates(self) -> List[Dict[str, Any]]:
        """
        Ottiene gli aggiornamenti da Telegram
        
        Returns:
            Lista di aggiornamenti
        """
        try:
            url = f"{self.base_url}/getUpdates"
            
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("ok"):
                return data.get("result", [])
            else:
                self.logger.error(f"Errore nel recupero degli aggiornamenti: {data.get('description')}")
                return []
                
        except Exception as e:
            self.logger.error(f"Errore nel recupero degli aggiornamenti: {str(e)}")
            return []
            
    def send_chart(self, chart_path: str, caption: str = "") -> bool:
        """
        Invia un'immagine del grafico a Telegram
        
        Args:
            chart_path: Percorso dell'immagine
            caption: Didascalia dell'immagine
            
        Returns:
            True se l'invio è riuscito
        """
        if not self.token or not self.chat_id:
            self.logger.warning("Token o chat_id non configurati")
            return False
            
        try:
            url = f"{self.base_url}/sendPhoto"
            
            # Aggiungi il prefisso alla didascalia
            caption = self.log_prefix + caption
            
            # Limita la lunghezza della didascalia a 1024 caratteri
            if len(caption) > 1024:
                caption = caption[:1021] + "..."
            
            files = {"photo": open(chart_path, "rb")}
            payload = {
                "chat_id": self.chat_id,
                "caption": caption,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=payload, files=files)
            data = response.json()
            
            if data.get("ok"):
                return True
            else:
                self.logger.error(f"Errore nell'invio dell'immagine a Telegram: {data.get('description')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio dell'immagine a Telegram: {str(e)}")
            return False
    
    def send_market_update(self, market_data: Dict[str, Any]) -> bool:
        """
        Invia un aggiornamento di mercato a Telegram
        
        Args:
            market_data: Dati di mercato
            
        Returns:
            True se l'invio è riuscito
        """
        try:
            symbol = market_data.get("symbol", "Unknown")
            price = market_data.get("last_price", 0.0)
            change = market_data.get("daily_change_pct", 0.0)
            
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            # Componi il messaggio
            message = f"{emoji} *{symbol} Market Update*\n\n"
            message += f"Price: `{price:.5f}`\n"
            message += f"24h Change: `{change:.2f}%`\n"
            
            if "volume" in market_data:
                message += f"Volume: `{market_data['volume']:.2f}`\n"
            
            if "trend" in market_data:
                trend = market_data["trend"]
                trend_emoji = "📈" if trend == "bullish" else "📉" if trend == "bearish" else "➡️"
                message += f"Trend: {trend_emoji} `{trend}`\n"
            
            if "rsi" in market_data:
                rsi = market_data["rsi"]
                message += f"RSI: `{rsi:.2f}`\n"
            
            if "next_support" in market_data and "next_resistance" in market_data:
                message += f"Support: `{market_data['next_support']:.5f}`\n"
                message += f"Resistance: `{market_data['next_resistance']:.5f}`\n"
            
            message += f"\nTime: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio dell'aggiornamento di mercato: {str(e)}")
            return False
