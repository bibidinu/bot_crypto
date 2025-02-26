"""
Modulo per l'integrazione con Discord
"""
from typing import Dict, List, Optional, Any
import time
import threading
import requests
import json
from datetime import datetime
import io
import os

from utils.logger import get_logger
from config.credentials import DISCORD_WEBHOOK_URL
from config.settings import (
    BOT_NAME, NOTIFICATION_TRADES, NOTIFICATION_ERRORS, 
    NOTIFICATION_PERFORMANCE
)

logger = get_logger(__name__)

class DiscordClient:
    """Classe per l'integrazione con Discord tramite webhook"""
    
    def __init__(self, webhook_url: str = DISCORD_WEBHOOK_URL):
        """
        Inizializza il client Discord
        
        Args:
            webhook_url: URL del webhook Discord
        """
        self.logger = get_logger(__name__)
        
        self.webhook_url = webhook_url
        
        # Prefisso per i messaggi
        self.bot_name = BOT_NAME
        
        # Cache dei messaggi per evitare duplicati
        self.message_cache = set()
        self.max_cache_size = 100
        
        self.logger.info("DiscordClient inizializzato")
    
    def test_connection(self) -> bool:
        """
        Testa la connessione a Discord
        
        Returns:
            True se la connessione è riuscita
        """
        if not self.webhook_url:
            self.logger.warning("URL del webhook non configurato")
            return False
            
        try:
            # Invia un messaggio di test
            payload = {
                "content": "",
                "embeds": [{
                    "title": "Test di connessione",
                    "description": "Se vedi questo messaggio, la connessione è riuscita.",
                    "color": 3066993,  # Verde
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                self.logger.info("Connessione a Discord riuscita")
                return True
            else:
                self.logger.error(f"Errore nella connessione a Discord: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nel test di connessione a Discord: {str(e)}")
            return False
    
    def send_message(self, content: str, title: str = None, color: int = None) -> bool:
        """
        Invia un messaggio a Discord
        
        Args:
            content: Contenuto del messaggio
            title: Titolo dell'embed (opzionale)
            color: Colore dell'embed (opzionale)
            
        Returns:
            True se l'invio è riuscito
        """
        if not self.webhook_url:
            self.logger.warning("URL del webhook non configurato")
            return False
            
        try:
            # Calcola un hash del messaggio per la cache
            message_hash = hash(f"{title}:{content}")
            
            # Verifica se il messaggio è già stato inviato di recente
            if message_hash in self.message_cache:
                return True
            
            # Aggiungi alla cache
            self.message_cache.add(message_hash)
            
            # Limita la dimensione della cache
            if len(self.message_cache) > self.max_cache_size:
                self.message_cache.pop()
            
            # Prepara il payload
            payload = {"content": ""}
            
            if title:
                # Usa un embed
                embed = {
                    "title": title,
                    "description": content,
                    "color": color or 3447003,  # Blu di default
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }
                payload["embeds"] = [embed]
            else:
                # Messaggio semplice
                payload["content"] = content
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio del messaggio a Discord: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio del messaggio a Discord: {str(e)}")
            return False
    
    def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """
        Invia una notifica di trade a Discord
        
        Args:
            trade_data: Dati del trade
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_TRADES or not self.webhook_url:
            return False
            
        try:
            symbol = trade_data.get("symbol", "Unknown")
            action = trade_data.get("action", "Unknown")
            price = trade_data.get("price", 0.0)
            size = trade_data.get("size", 0.0)
            pnl = trade_data.get("pnl", 0.0)
            
            # Determina il colore dell'embed
            color = 5763719 if action.lower() in ["buy", "long"] else 15548997 if action.lower() in ["sell", "short"] else 10197915
            
            # Costruisci la descrizione
            description = []
            description.append(f"**Price:** {price:.5f}")
            description.append(f"**Size:** {size:.5f}")
            
            if "stop_loss" in trade_data:
                description.append(f"**Stop Loss:** {trade_data['stop_loss']:.5f}")
                
            if "take_profit" in trade_data:
                if isinstance(trade_data["take_profit"], list):
                    for i, tp in enumerate(trade_data["take_profit"]):
                        description.append(f"**TP{i+1}:** {tp:.5f}")
                else:
                    description.append(f"**Take Profit:** {trade_data['take_profit']:.5f}")
            
            if pnl != 0.0:
                description.append(f"**PnL:** {pnl:.2f} ({trade_data.get('pnl_percent', 0.0):.2f}%)")
            
            if "reason" in trade_data:
                description.append(f"**Reason:** {trade_data['reason']}")
            
            # Crea il payload
            payload = {
                "content": "",
                "embeds": [{
                    "title": f"{action.upper()} {symbol}",
                    "description": "\n".join(description),
                    "color": color,
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio della notifica di trade: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio della notifica di trade: {str(e)}")
            return False
    
    def send_error_notification(self, error_message: str) -> bool:
        """
        Invia una notifica di errore a Discord
        
        Args:
            error_message: Messaggio di errore
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_ERRORS or not self.webhook_url:
            return False
            
        try:
            # Crea il payload
            payload = {
                "content": "",
                "embeds": [{
                    "title": "ERROR",
                    "description": error_message,
                    "color": 15158332,  # Rosso
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio della notifica di errore: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio della notifica di errore: {str(e)}")
            return False
    
    def send_performance_report(self, performance_data: Dict[str, Any]) -> bool:
        """
        Invia un report di performance a Discord
        
        Args:
            performance_data: Dati di performance
            
        Returns:
            True se l'invio è riuscito
        """
        if not NOTIFICATION_PERFORMANCE or not self.webhook_url:
            return False
            
        try:
            total_profit = performance_data.get("total_profit", 0.0)
            win_rate = performance_data.get("win_rate", 0.0) * 100
            total_trades = performance_data.get("total_trades", 0)
            
            # Determina il colore dell'embed
            color = 5763719 if total_profit > 0 else 15548997  # Verde se profit positivo, rosso altrimenti
            
            # Costruisci la descrizione
            description = []
            description.append(f"**Total Profit:** {total_profit:.2f} USDT")
            description.append(f"**Win Rate:** {win_rate:.2f}%")
            description.append(f"**Total Trades:** {total_trades}")
            
            if "winning_trades" in performance_data and "losing_trades" in performance_data:
                description.append(f"**Winning/Losing:** {performance_data['winning_trades']}/{performance_data['losing_trades']}")
            
            if "profit_factor" in performance_data:
                description.append(f"**Profit Factor:** {performance_data['profit_factor']:.2f}")
            
            if "avg_win" in performance_data and "avg_loss" in performance_data:
                description.append(f"**Avg Win/Loss:** {performance_data['avg_win']:.2f}/{performance_data['avg_loss']:.2f}")
            
            if "open_positions" in performance_data:
                description.append(f"**Open Positions:** {performance_data['open_positions']}")
            
            # Crea il payload
            payload = {
                "content": "",
                "embeds": [{
                    "title": "PERFORMANCE REPORT",
                    "description": "\n".join(description),
                    "color": color,
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio del report di performance: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio del report di performance: {str(e)}")
            return False
    
    def send_chart(self, chart_path: str, title: str = "") -> bool:
        """
        Invia un'immagine del grafico a Discord
        
        Args:
            chart_path: Percorso dell'immagine
            title: Titolo dell'immagine
            
        Returns:
            True se l'invio è riuscito
        """
        if not self.webhook_url:
            self.logger.warning("URL del webhook non configurato")
            return False
            
        try:
            # Prepara i dati multipart
            with open(chart_path, 'rb') as f:
                image_data = f.read()
                
            # Costruisci il payload con l'immagine
            payload = {
                "content": title if title else "",
            }
            
            # Prepara il file per l'upload
            files = {
                'file': (os.path.basename(chart_path), image_data)
            }
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, data=payload, files=files)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio dell'immagine: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio dell'immagine: {str(e)}")
            return False
    
    def send_market_update(self, market_data: Dict[str, Any]) -> bool:
        """
        Invia un aggiornamento di mercato a Discord
        
        Args:
            market_data: Dati di mercato
            
        Returns:
            True se l'invio è riuscito
        """
        try:
            symbol = market_data.get("symbol", "Unknown")
            price = market_data.get("last_price", 0.0)
            change = market_data.get("daily_change_pct", 0.0)
            
            # Determina il colore dell'embed
            color = 5763719 if change > 0 else 15548997 if change < 0 else 10197915  # Verde se positivo, rosso se negativo, grigio se neutro
            
            # Costruisci la descrizione
            description = []
            description.append(f"**Price:** {price:.5f}")
            description.append(f"**24h Change:** {change:.2f}%")
            
            if "volume" in market_data:
                description.append(f"**Volume:** {market_data['volume']:.2f}")
            
            if "trend" in market_data:
                description.append(f"**Trend:** {market_data['trend']}")
            
            if "rsi" in market_data:
                description.append(f"**RSI:** {market_data['rsi']:.2f}")
            
            if "next_support" in market_data and "next_resistance" in market_data:
                description.append(f"**Support:** {market_data['next_support']:.5f}")
                description.append(f"**Resistance:** {market_data['next_resistance']:.5f}")
            
            # Crea il payload
            payload = {
                "content": "",
                "embeds": [{
                    "title": f"{symbol} Market Update",
                    "description": "\n".join(description),
                    "color": color,
                    "footer": {
                        "text": f"{self.bot_name} • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                }]
            }
            
            # Invia il messaggio
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                self.logger.error(f"Errore nell'invio dell'aggiornamento di mercato: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Errore nell'invio dell'aggiornamento di mercato: {str(e)}")
            return False
