"""
Modulo per gli handler dei comandi del bot di trading
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import os

from utils.logger import get_logger
from commands.cmd_processor import CommandProcessor
from config.settings import (
    BOT_NAME, BOT_MODE, BOT_STATUS, BotMode, BotStatus,
    DEFAULT_TRADING_PAIRS
)

logger = get_logger(__name__)

class CommandHandlers:
    """Classe per gli handler dei comandi del bot di trading"""
    
    def __init__(self, bot):
        """
        Inizializza gli handler dei comandi
        
        Args:
            bot: Istanza del bot di trading
        """
        self.logger = get_logger(__name__)
        self.bot = bot
        
        # Crea il processore dei comandi
        self.cmd_processor = CommandProcessor()
        
        # Registra i comandi
        self._register_commands()
        
        self.logger.info("CommandHandlers inizializzato")
    
    def _register_commands(self) -> None:
        """Registra tutti i comandi disponibili"""
        # Comandi generali
        self.cmd_processor.register_command(
            "status", self.cmd_status, 
            "Mostra lo stato corrente del bot", 
            group="generali", aliases=["stat", "info"]
        )
        
        self.cmd_processor.register_command(
            "start", self.cmd_start, 
            "Avvia il bot", 
            group="generali"
        )
        
        self.cmd_processor.register_command(
            "stop", self.cmd_stop, 
            "Ferma il bot", 
            group="generali"
        )
        
        self.cmd_processor.register_command(
            "pause", self.cmd_pause, 
            "Mette in pausa il bot", 
            group="generali"
        )
        
        self.cmd_processor.register_command(
            "resume", self.cmd_resume, 
            "Riprende l'esecuzione del bot", 
            group="generali"
        )
        
        # Comandi di trading
        self.cmd_processor.register_command(
            "positions", self.cmd_positions, 
            "Mostra le posizioni aperte", 
            group="trading", aliases=["pos"]
        )
        
        self.cmd_processor.register_command(
            "orders", self.cmd_orders, 
            "Mostra gli ordini aperti", 
            group="trading", aliases=["ord"]
        )
        
        self.cmd_processor.register_command(
            "close", self.cmd_close, 
            "Chiude una posizione. Uso: /close <symbol>", 
            group="trading"
        )
        
        self.cmd_processor.register_command(
            "closeall", self.cmd_closeall, 
            "Chiude tutte le posizioni aperte", 
            group="trading"
        )
        
        self.cmd_processor.register_command(
            "balance", self.cmd_balance, 
            "Mostra il saldo del wallet", 
            group="trading", aliases=["bal"]
        )
        
        self.cmd_processor.register_command(
            "trade", self.cmd_trade, 
            "Esegue un trade manuale. Uso: /trade <symbol> <side> <size> [price]", 
            group="trading"
        )
        
        # Comandi di configurazione
        self.cmd_processor.register_command(
            "setmode", self.cmd_setmode, 
            "Imposta la modalità del bot (demo/live). Uso: /setmode <mode>", 
            group="configurazione"
        )
        
        self.cmd_processor.register_command(
            "addpair", self.cmd_addpair, 
            "Aggiunge una coppia di trading. Uso: /addpair <symbol>", 
            group="configurazione"
        )
        
        self.cmd_processor.register_command(
            "removepair", self.cmd_removepair, 
            "Rimuove una coppia di trading. Uso: /removepair <symbol>", 
            group="configurazione"
        )
        
        self.cmd_processor.register_command(
            "pairs", self.cmd_pairs, 
            "Mostra le coppie di trading monitorate", 
            group="configurazione"
        )
        
        self.cmd_processor.register_command(
            "setrisk", self.cmd_setrisk, 
            "Imposta il livello di rischio. Uso: /setrisk <percent>", 
            group="configurazione"
        )
        
        # Comandi di analisi
        self.cmd_processor.register_command(
            "stats", self.cmd_stats, 
            "Mostra le statistiche di trading", 
            group="analisi"
        )
        
        self.cmd_processor.register_command(
            "performance", self.cmd_performance, 
            "Mostra le performance dettagliate. Uso: /performance [symbol] [period]", 
            group="analisi", aliases=["perf"]
        )
        
        self.cmd_processor.register_command(
            "trades", self.cmd_trades, 
            "Mostra gli ultimi trade eseguiti. Uso: /trades [limit]", 
            group="analisi"
        )
        
        self.cmd_processor.register_command(
            "analyze", self.cmd_analyze, 
            "Analizza una coppia di trading. Uso: /analyze <symbol>", 
            group="analisi", aliases=["a"]
        )
        
        self.cmd_processor.register_command(
            "scan", self.cmd_scan, 
            "Scansiona tutte le coppie per opportunità", 
            group="analisi"
        )
        
        # Comandi di reportistica
        self.cmd_processor.register_command(
            "report", self.cmd_report, 
            "Genera un report delle performance", 
            group="reportistica"
        )
        
        self.cmd_processor.register_command(
            "chart", self.cmd_chart, 
            "Genera un grafico. Uso: /chart <symbol> [timeframe]", 
            group="reportistica"
        )
        
        # Comandi di sistema
        self.cmd_processor.register_command(
            "version", self.cmd_version, 
            "Mostra la versione del bot", 
            group="sistema"
        )
        
        self.cmd_processor.register_command(
            "logs", self.cmd_logs, 
            "Mostra gli ultimi log. Uso: /logs [limit]", 
            group="sistema"
        )
    
    def process_command(self, command_text: str) -> str:
        """
        Elabora un comando
        
        Args:
            command_text: Testo del comando
            
        Returns:
            Risultato dell'elaborazione
        """
        self.logger.info(f"Elaborazione comando: {command_text}")
        
        # Passa il comando al processore
        result = self.cmd_processor.process_command(command_text)
        
        self.logger.info(f"Risultato comando: {result[:100]}...")
        
        return result
    
    # --- Handler dei comandi ---
    
    def cmd_status(self, args: str) -> str:
        """
        Mostra lo stato corrente del bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Stato del bot
        """
        # Ottieni lo stato del bot
        status = {
            "name": BOT_NAME,
            "mode": BOT_MODE.value,
            "status": BOT_STATUS.value,
            "uptime": self.bot.get_uptime(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trading_pairs": len(self.bot.trading_pairs),
            "open_positions": len(self.bot.get_open_positions()),
            "wallet_balance": self.bot.get_wallet_balance(),
            "total_trades": self.bot.performance.total_trades,
            "profitable_trades": self.bot.performance.winning_trades
        }
        
        # Formatta il messaggio
        message = f"📊 **Stato del Bot**\n\n"
        message += f"Nome: {status['name']}\n"
        message += f"Modalità: {status['mode']}\n"
        message += f"Stato: {status['status']}\n"
        message += f"Uptime: {status['uptime']}\n"
        message += f"Data/Ora: {status['current_time']}\n\n"
        
        message += f"Coppie monitorate: {status['trading_pairs']}\n"
        message += f"Posizioni aperte: {status['open_positions']}\n"
        message += f"Saldo wallet: {status['wallet_balance']:.2f} USDT\n\n"
        
        message += f"Trade totali: {status['total_trades']}\n"
        message += f"Trade profittevoli: {status['profitable_trades']}\n"
        
        if status['total_trades'] > 0:
            win_rate = (status['profitable_trades'] / status['total_trades']) * 100
            message += f"Win rate: {win_rate:.2f}%\n"
        
        return message
    
    def cmd_start(self, args: str) -> str:
        """
        Avvia il bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        if self.bot.is_running():
            return "Il bot è già in esecuzione."
        
        self.bot.start()
        return "✅ Bot avviato con successo!"
    
    def cmd_stop(self, args: str) -> str:
        """
        Ferma il bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        if not self.bot.is_running():
            return "Il bot è già fermo."
        
        self.bot.stop()
        return "🛑 Bot fermato con successo!"
    
    def cmd_pause(self, args: str) -> str:
        """
        Mette in pausa il bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        if not self.bot.is_running():
            return "Il bot è già fermo."
        
        if self.bot.is_paused():
            return "Il bot è già in pausa."
        
        self.bot.pause()
        return "⏸️ Bot messo in pausa. Usa /resume per riprendere."
    
    def cmd_resume(self, args: str) -> str:
        """
        Riprende l'esecuzione del bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        if not self.bot.is_paused():
            return "Il bot non è in pausa."
        
        self.bot.resume()
        return "▶️ Bot ripreso con successo!"
    
    def cmd_positions(self, args: str) -> str:
        """
        Mostra le posizioni aperte
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Elenco delle posizioni
        """
        # Ottieni le posizioni aperte
        positions = self.bot.get_open_positions()
        
        if not positions:
            return "Nessuna posizione aperta."
        
        # Formatta il messaggio
        message = f"📋 **Posizioni Aperte ({len(positions)})**\n\n"
        
        for i, pos in enumerate(positions):
            symbol = pos.get("symbol", "Unknown")
            entry_type = pos.get("entry_type", "unknown").upper()
            entry_price = pos.get("entry_price", 0.0)
            current_price = pos.get("current_price", 0.0)
            size = pos.get("size", 0.0)
            
            # Calcola PnL
            if entry_type.lower() == "long":
                pnl_percent = ((current_price / entry_price) - 1) * 100
            else:
                pnl_percent = ((entry_price / current_price) - 1) * 100
                
            pnl = pnl_percent * size * entry_price / 100
            
            # Emoji in base al PnL
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            message += f"{i+1}. {emoji} {symbol} ({entry_type})\n"
            message += f"   Prezzo entrata: {entry_price:.5f}\n"
            message += f"   Prezzo attuale: {current_price:.5f}\n"
            message += f"   Size: {size:.5f}\n"
            message += f"   PnL: {pnl:.2f} USDT ({pnl_percent:.2f}%)\n"
            
            if pos.get("stop_loss"):
                message += f"   Stop Loss: {pos['stop_loss']:.5f}\n"
                
            if pos.get("take_profits"):
                tps = pos["take_profits"]
                if isinstance(tps, list):
                    for j, tp in enumerate(tps):
                        message += f"   TP{j+1}: {tp:.5f}\n"
                else:
                    message += f"   Take Profit: {tps:.5f}\n"
            
            message += "\n"
        
        return message
    
    def cmd_orders(self, args: str) -> str:
        """
        Mostra gli ordini aperti
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Elenco degli ordini
        """
        # Ottieni gli ordini aperti
        orders = self.bot.get_open_orders()
        
        if not orders:
            return "Nessun ordine aperto."
        
        # Formatta il messaggio
        message = f"📋 **Ordini Aperti ({len(orders)})**\n\n"
        
        for i, order in enumerate(orders):
            symbol = order.get("symbol", "Unknown")
            side = order.get("side", "unknown").upper()
            order_type = order.get("orderType", "unknown").upper()
            price = order.get("price", 0.0)
            qty = order.get("qty", 0.0)
            
            message += f"{i+1}. {symbol} {side} ({order_type})\n"
            
            if order_type != "MARKET":
                message += f"   Prezzo: {price:.5f}\n"
                
            message += f"   Quantità: {qty:.5f}\n"
            message += f"   Stato: {order.get('orderStatus', 'unknown')}\n"
            message += f"   ID: {order.get('orderId', 'N/A')}\n\n"
        
        return message
    
    def cmd_close(self, args: str) -> str:
        """
        Chiude una posizione
        
        Args:
            args: Argomenti del comando (symbol)
            
        Returns:
            Messaggio di conferma
        """
        if not args:
            return "Simbolo non specificato. Uso: /close <symbol>"
        
        symbol = args.strip().upper()
        
        # Chiudi la posizione
        result = self.bot.close_position(symbol)
        
        if result:
            return f"✅ Posizione {symbol} chiusa con successo!"
        else:
            return f"❌ Impossibile chiudere la posizione {symbol}. Verificare che sia aperta."
    
    def cmd_closeall(self, args: str) -> str:
        """
        Chiude tutte le posizioni aperte
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        # Ottieni le posizioni aperte
        positions = self.bot.get_open_positions()
        
        if not positions:
            return "Nessuna posizione aperta da chiudere."
        
        # Chiudi tutte le posizioni
        closed = 0
        failed = 0
        
        for pos in positions:
            symbol = pos.get("symbol", "")
            if symbol:
                result = self.bot.close_position(symbol)
                if result:
                    closed += 1
                else:
                    failed += 1
        
        return f"✅ Chiuse {closed} posizioni. Fallite: {failed}."
    
    def cmd_balance(self, args: str) -> str:
        """
        Mostra il saldo del wallet
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Saldo del wallet
        """
        # Ottieni il saldo
        balance = self.bot.get_wallet_balance()
        
        # Ottieni anche le posizioni aperte
        positions = self.bot.get_open_positions()
        
        # Calcola il valore totale delle posizioni
        positions_value = 0.0
        for pos in positions:
            entry_price = pos.get("entry_price", 0.0)
            size = pos.get("size", 0.0)
            positions_value += entry_price * size
        
        # Calcola il PnL non realizzato
        unrealized_pnl = 0.0
        for pos in positions:
            symbol = pos.get("symbol", "")
            entry_type = pos.get("entry_type", "").lower()
            entry_price = pos.get("entry_price", 0.0)
            size = pos.get("size", 0.0)
            
            if symbol:
                current_price = self.bot.get_current_price(symbol)
                
                if entry_type == "long":
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                    
                unrealized_pnl += pnl
        
        # Formatta il messaggio
        message = f"💰 **Bilancio Wallet**\n\n"
        message += f"Saldo: {balance:.2f} USDT\n"
        message += f"Valore posizioni: {positions_value:.2f} USDT\n"
        message += f"PnL non realizzato: {unrealized_pnl:.2f} USDT\n"
        message += f"Equity totale: {(balance + unrealized_pnl):.2f} USDT\n"
        
        return message
    
    def cmd_trade(self, args: str) -> str:
        """
        Esegue un trade manuale
        
        Args:
            args: Argomenti del comando (symbol side size [price])
            
        Returns:
            Messaggio di conferma
        """
        # Analizza gli argomenti
        args_parts = args.strip().split()
        
        if len(args_parts) < 3:
            return "Argomenti insufficienti. Uso: /trade <symbol> <side> <size> [price]"
        
        symbol = args_parts[0].upper()
        side = args_parts[1].lower()
        
        try:
            size = float(args_parts[2])
        except ValueError:
            return f"Quantità non valida: {args_parts[2]}"
        
        price = None
        if len(args_parts) > 3:
            try:
                price = float(args_parts[3])
            except ValueError:
                return f"Prezzo non valido: {args_parts[3]}"
        
        # Verifica il lato
        if side not in ["buy", "sell", "long", "short"]:
            return f"Lato non valido: {side}. Usa 'buy', 'sell', 'long' o 'short'."
        
        # Esegui il trade
        result = self.bot.execute_manual_trade(symbol, side, size, price)
        
        if result:
            return f"✅ Trade eseguito con successo! ID: {result.get('orderId', 'N/A')}"
        else:
            return f"❌ Impossibile eseguire il trade. Verifica i parametri."
    
    def cmd_setmode(self, args: str) -> str:
        """
        Imposta la modalità del bot
        
        Args:
            args: Argomenti del comando (mode)
            
        Returns:
            Messaggio di conferma
        """
        if not args:
            return "Modalità non specificata. Uso: /setmode <mode> (demo/live)"
        
        mode = args.strip().lower()
        
        if mode == "demo":
            self.bot.set_mode(BotMode.DEMO)
            return "✅ Modalità del bot impostata su DEMO."
        elif mode == "live":
            # Verifica se è pronto per il live trading
            if not self.bot.performance.demo_to_live_ready:
                return ("❌ Il bot non è ancora pronto per il live trading. "
                       f"Win rate: {self.bot.performance.get_overall_stats()['win_rate'] * 100:.2f}%, "
                       f"richiesto: {self.bot.config.PROFIT_THRESHOLD_PERCENT}%")
            
            self.bot.set_mode(BotMode.LIVE)
            return "✅ Modalità del bot impostata su LIVE. Trading reale attivato!"
        else:
            return f"❌ Modalità non valida: {mode}. Usa 'demo' o 'live'."
    
    def cmd_addpair(self, args: str) -> str:
        """
        Aggiunge una coppia di trading
        
        Args:
            args: Argomenti del comando (symbol)
            
        Returns:
            Messaggio di conferma
        """
        if not args:
            return "Simbolo non specificato. Uso: /addpair <symbol>"
        
        symbol = args.strip().upper()
        
        # Aggiungi la coppia
        result = self.bot.add_trading_pair(symbol)
        
        if result:
            return f"✅ Coppia {symbol} aggiunta con successo!"
        else:
            return f"❌ Impossibile aggiungere la coppia {symbol}. Verificare il formato o se esiste già."
    
    def cmd_removepair(self, args: str) -> str:
        """
        Rimuove una coppia di trading
        
        Args:
            args: Argomenti del comando (symbol)
            
        Returns:
            Messaggio di conferma
        """
        if not args:
            return "Simbolo non specificato. Uso: /removepair <symbol>"
        
        symbol = args.strip().upper()
        
        # Rimuovi la coppia
        result = self.bot.remove_trading_pair(symbol)
        
        if result:
            return f"✅ Coppia {symbol} rimossa con successo!"
        else:
            return f"❌ Impossibile rimuovere la coppia {symbol}. Verificare se è monitorata."
    
    def cmd_pairs(self, args: str) -> str:
        """
        Mostra le coppie di trading monitorate
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Elenco delle coppie
        """
        # Ottieni le coppie
        pairs = self.bot.trading_pairs
        
        if not pairs:
            return "Nessuna coppia monitorata."
        
        # Formatta il messaggio
        message = f"📋 **Coppie Monitorate ({len(pairs)})**\n\n"
        
        for i, symbol in enumerate(sorted(pairs)):
            message += f"{i+1}. {symbol}\n"
        
        return message
    
    def cmd_setrisk(self, args: str) -> str:
        """
        Imposta il livello di rischio
        
        Args:
            args: Argomenti del comando (percent)
            
        Returns:
            Messaggio di conferma
        """
        if not args:
            return "Percentuale non specificata. Uso: /setrisk <percent>"
        
        try:
            percent = float(args.strip())
        except ValueError:
            return f"Percentuale non valida: {args}"
        
        # Limita il rischio
        if percent < 0.1:
            percent = 0.1
        elif percent > 5.0:
            percent = 5.0
        
        # Imposta il rischio
        self.bot.set_risk_per_trade(percent)
        
        return f"✅ Rischio per trade impostato al {percent:.1f}% del capitale."
    
    def cmd_stats(self, args: str) -> str:
        """
        Mostra le statistiche di trading
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Statistiche di trading
        """
        # Ottieni le statistiche globali
        stats = self.bot.performance.get_overall_stats()
        
        # Formatta il messaggio
        message = f"📊 **Statistiche di Trading**\n\n"
        
        message += f"Trade totali: {stats['total_trades']}\n"
        message += f"Trade profittevoli: {stats['winning_trades']}\n"
        message += f"Trade in perdita: {stats['losing_trades']}\n"
        
        if stats['total_trades'] > 0:
            message += f"Win rate: {stats['win_rate'] * 100:.2f}%\n"
            
        message += f"Profitto totale: {stats['total_profit']:.2f} USDT\n"
        
        if stats.get('avg_win') and stats.get('avg_loss'):
            message += f"Profitto medio: {stats['avg_win']:.2f} USDT\n"
            message += f"Perdita media: {stats['avg_loss']:.2f} USDT\n"
            
        if stats.get('profit_factor'):
            message += f"Profit factor: {stats['profit_factor']:.2f}\n"
            
        if stats.get('max_drawdown'):
            message += f"Drawdown massimo: {stats['max_drawdown']:.2f} USDT "
            message += f"({stats.get('drawdown_percent', 0.0):.2f}%)\n"
        
        # Aggiungi lo stato di passaggio a live
        demo_to_live = stats.get('demo_to_live', {})
        if demo_to_live:
            message += f"\n📈 **Passaggio a Live Trading**\n"
            message += f"Stato: {'✅ Pronto' if demo_to_live.get('ready') else '⏳ In attesa'}\n"
            message += f"Win rate attuale: {demo_to_live.get('win_rate', 0.0):.2f}%\n"
            message += f"Win rate richiesto: {demo_to_live.get('threshold', 0.0):.2f}%\n"
            message += f"Trade eseguiti: {demo_to_live.get('trades', 0)}/{demo_to_live.get('required_trades', 0)}\n"
        
        return message
    
    def cmd_performance(self, args: str) -> str:
        """
        Mostra le performance dettagliate
        
        Args:
            args: Argomenti del comando (symbol) (period)
            
        Returns:
            Performance dettagliate
        """
        # Analizza gli argomenti
        args_parts = args.strip().split()
        
        symbol = None
        period = "daily"
        
        if args_parts:
            if args_parts[0].lower() in ["daily", "weekly", "monthly", "yearly"]:
                period = args_parts[0].lower()
            else:
                symbol = args_parts[0].upper()
                
                if len(args_parts) > 1:
                    if args_parts[1].lower() in ["daily", "weekly", "monthly", "yearly"]:
                        period = args_parts[1].lower()
        
        if symbol:
            # Performance per simbolo
            stats = self.bot.performance.get_stats_for_symbol(symbol)
            
            if not stats:
                return f"Nessun dato disponibile per {symbol}."
            
            # Formatta il messaggio
            message = f"📊 **Performance {symbol}**\n\n"
            
            message += f"Trade totali: {stats.get('total_trades', 0)}\n"
            message += f"Trade profittevoli: {stats.get('winning_trades', 0)}\n"
            message += f"Trade in perdita: {stats.get('losing_trades', 0)}\n"
            
            if stats.get('total_trades', 0) > 0:
                message += f"Win rate: {stats.get('win_rate', 0.0) * 100:.2f}%\n"
                
            message += f"Profitto totale: {stats.get('total_profit', 0.0):.2f} USDT\n"
            
            if stats.get('avg_win') and stats.get('avg_loss'):
                message += f"Profitto medio: {stats.get('avg_win', 0.0):.2f} USDT\n"
                message += f"Perdita media: {stats.get('avg_loss', 0.0):.2f} USDT\n"
                
            if stats.get('profit_factor'):
                message += f"Profit factor: {stats.get('profit_factor', 0.0):.2f}\n"
                
            if stats.get('max_consecutive_wins'):
                message += f"Vincite consecutive max: {stats.get('max_consecutive_wins', 0)}\n"
                
            if stats.get('max_consecutive_losses'):
                message += f"Perdite consecutive max: {stats.get('max_consecutive_losses', 0)}\n"
                
            if stats.get('avg_holding_time'):
                message += f"Tempo medio di detenzione: {stats.get('avg_holding_time', 0.0):.2f} ore\n"
            
            return message
            
        else:
            # Performance globali per periodo
            if period == "daily":
                stats = self.bot.performance.get_daily_stats(7)
                period_name = "Giornaliere"
            elif period == "weekly":
                stats = self.bot.performance.get_weekly_stats(4)
                period_name = "Settimanali"
            elif period == "monthly":
                stats = self.bot.performance.get_monthly_stats(6)
                period_name = "Mensili"
            elif period == "yearly":
                stats = self.bot.performance.get_yearly_stats()
                period_name = "Annuali"
            else:
                stats = self.bot.performance.get_daily_stats(7)
                period_name = "Giornaliere"
            
            if not stats:
                return f"Nessun dato disponibile per performance {period_name.lower()}."
            
            # Formatta il messaggio
            message = f"📊 **Performance {period_name}**\n\n"
            
            # Ordina i periodi
            sorted_keys = sorted(stats.keys(), reverse=True)
            
            for key in sorted_keys:
                stat = stats[key]
                
                profit = stat.get('total_profit', 0.0)
                trades = stat.get('total_trades', 0)
                win_rate = stat.get('winning_trades', 0) / trades if trades > 0 else 0
                
                emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
                
                message += f"{emoji} **{key}**\n"
                message += f"   Profitto: {profit:.2f} USDT\n"
                message += f"   Trade: {trades}\n"
                message += f"   Win rate: {win_rate * 100:.2f}%\n\n"
            
            return message
    
    def cmd_trades(self, args: str) -> str:
        """
        Mostra gli ultimi trade eseguiti
        
        Args:
            args: Argomenti del comando (limit)
            
        Returns:
            Elenco dei trade
        """
        # Analizza il limite
        limit = 10
        
        if args:
            try:
                limit = int(args.strip())
            except ValueError:
                pass
        
        # Ottieni i trade recenti
        trades = self.bot.performance.get_recent_trades(limit)
        
        if not trades:
            return "Nessun trade eseguito."
        
        # Formatta il messaggio
        message = f"📋 **Ultimi Trade ({len(trades)})**\n\n"
        
        for i, trade in enumerate(trades):
            symbol = trade.get('symbol', 'Unknown')
            entry_type = trade.get('entry_type', 'unknown').upper()
            profit = trade.get('profit', 0.0)
            
            # Formatta la data di uscita
            exit_time_str = trade.get('exit_time', '')
            if isinstance(exit_time_str, str) and exit_time_str:
                try:
                    exit_time = datetime.fromisoformat(exit_time_str)
                    exit_time_str = exit_time.strftime("%d/%m/%Y %H:%M")
                except:
                    pass
            
            emoji = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            
            message += f"{i+1}. {emoji} {symbol} ({entry_type})\n"
            message += f"   Profitto: {profit:.2f} USDT\n"
            message += f"   Data: {exit_time_str}\n"
        
        return message
    
    def cmd_analyze(self, args: str) -> str:
        """
        Analizza una coppia di trading
        
        Args:
            args: Argomenti del comando (symbol)
            
        Returns:
            Analisi della coppia
        """
        if not args:
            return "Simbolo non specificato. Uso: /analyze <symbol>"
        
        symbol = args.strip().upper()
        
        # Esegui l'analisi
        analysis = self.bot.analyze_symbol(symbol)
        
        if not analysis:
            return f"Impossibile analizzare {symbol}."
        
        # Formatta il messaggio
        message = f"📊 **Analisi {symbol}**\n\n"
        
        # Prezzo e variazione
        price = analysis.get('current_price', 0.0)
        change = analysis.get('price_change_24h', 0.0)
        
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
        
        message += f"{emoji} Prezzo: {price:.5f} USDT\n"
        message += f"Variazione 24h: {change:.2f}%\n\n"
        
        # Indicatori tecnici
        message += "📈 **Indicatori Tecnici**\n"
        
        rsi = analysis.get('rsi', 0.0)
        message += f"RSI: {rsi:.2f}"
        if rsi > 70:
            message += " (Ipercomprato)"
        elif rsi < 30:
            message += " (Ipervenduto)"
        message += "\n"
        
        if 'macd' in analysis and 'macd_signal' in analysis:
            macd = analysis.get('macd', 0.0)
            macd_signal = analysis.get('macd_signal', 0.0)
            
            message += f"MACD: {macd:.5f}, Signal: {macd_signal:.5f} "
            if macd > macd_signal:
                message += "🟢"
            else:
                message += "🔴"
            message += "\n"
        
        # Supporti e resistenze
        if 'next_support' in analysis and 'next_resistance' in analysis:
            message += "\n🔍 **Livelli Chiave**\n"
            message += f"Supporto: {analysis.get('next_support', 0.0):.5f}\n"
            message += f"Resistenza: {analysis.get('next_resistance', 0.0):.5f}\n"
        
        # Sentiment
        if 'sentiment' in analysis:
            message += "\n💭 **Sentiment**\n"
            sentiment = analysis.get('sentiment', 'neutral')
            
            emoji = "🟢" if sentiment == "bullish" else "🔴" if sentiment == "bearish" else "⚪"
            
            message += f"{emoji} Sentiment: {sentiment.capitalize()}\n"
            message += f"Score: {analysis.get('sentiment_score', 0.0):.2f}\n"
        
        # Segnale
        if 'signal' in analysis:
            message += "\n🎯 **Segnale di Trading**\n"
            signal = analysis.get('signal', 'hold').upper()
            
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
            
            message += f"{emoji} {signal}\n"
            message += f"Forza: {analysis.get('signal_strength', 0.0) * 100:.2f}%\n"
            
            if 'reason' in analysis:
                message += f"Motivo: {analysis.get('reason')}\n"
        
        return message
    
    def cmd_scan(self, args: str) -> str:
        """
        Scansiona tutte le coppie per opportunità
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Risultati della scansione
        """
        # Esegui la scansione
        results = self.bot.scan_for_opportunities()
        
        if not results:
            return "Nessuna opportunità trovata."
        
        # Formatta il messaggio
        message = f"🔍 **Opportunità di Trading ({len(results)})**\n\n"
        
        for i, result in enumerate(results):
            symbol = result.get('symbol', 'Unknown')
            signal = result.get('signal', 'HOLD').upper()
            strength = result.get('strength', 0.0) * 100
            
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
            
            message += f"{i+1}. {emoji} {symbol} ({signal})\n"
            message += f"   Forza: {strength:.2f}%\n"
            message += f"   Prezzo: {result.get('price', 0.0):.5f}\n"
            
            if 'reason' in result:
                reason = result.get('reason', '')
                if len(reason) > 100:
                    reason = reason[:97] + "..."
                message += f"   Motivo: {reason}\n"
                
            message += "\n"
        
        return message
    
    def cmd_report(self, args: str) -> str:
        """
        Genera un report delle performance
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Messaggio di conferma
        """
        # Genera il report
        result = self.bot.generate_report()
        
        if not result:
            return "Impossibile generare il report."
        
        # Notifica delle operazioni in corso
        charts_count = len(result.get('chart_paths', []))
        
        message = f"✅ Report generato con successo!\n\n"
        message += f"Report salvato in: {result.get('report_path', 'N/A')}\n"
        message += f"Grafici generati: {charts_count}\n"
        
        # Aggiungi i dati principali
        report_data = result.get('report', {})
        overall_stats = report_data.get('overall_stats', {})
        
        if overall_stats:
            message += f"\n📊 **Statistiche Principali**\n"
            message += f"Trade totali: {overall_stats.get('total_trades', 0)}\n"
            message += f"Win rate: {overall_stats.get('win_rate', 0.0) * 100:.2f}%\n"
            message += f"Profitto totale: {overall_stats.get('total_profit', 0.0):.2f} USDT\n"
        
        return message
    
    def cmd_chart(self, args: str) -> str:
        """
        Genera un grafico
        
        Args:
            args: Argomenti del comando (symbol) (timeframe)
            
        Returns:
            Messaggio di conferma
        """
        # Analizza gli argomenti
        args_parts = args.strip().split()
        
        if not args_parts:
            return "Simbolo non specificato. Uso: /chart <symbol> [timeframe]"
        
        symbol = args_parts[0].upper()
        timeframe = args_parts[1].lower() if len(args_parts) > 1 else "15m"
        
        # Genera il grafico
        chart_path = self.bot.generate_chart(symbol, timeframe)
        
        if not chart_path:
            return f"Impossibile generare il grafico per {symbol}."
        
        # Il bot invierà il grafico separatamente
        return f"✅ Grafico di {symbol} ({timeframe}) generato."
    
    def cmd_version(self, args: str) -> str:
        """
        Mostra la versione del bot
        
        Args:
            args: Argomenti del comando
            
        Returns:
            Versione del bot
        """
        from config.settings import BOT_NAME, BOT_VERSION
        
        message = f"🤖 **{BOT_NAME}**\n"
        message += f"Versione: {BOT_VERSION}\n"
        message += f"Data build: {datetime.now().strftime('%Y-%m-%d')}\n"
        
        return message
    
    def cmd_logs(self, args: str) -> str:
        """
        Mostra gli ultimi log
        
        Args:
            args: Argomenti del comando (limit)
            
        Returns:
            Ultimi log
        """
        # Analizza il limite
        limit = 10
        
        if args:
            try:
                limit = int(args.strip())
            except ValueError:
                pass
        
        # Ottieni i log
        logs = self.bot.get_logs(limit)
        
        if not logs:
            return "Nessun log disponibile."
        
        # Formatta il messaggio
        message = f"📋 **Ultimi Log ({len(logs)})**\n\n"
        
        for i, log in enumerate(logs):
            message += f"{i+1}. {log}\n"
        
        return message
