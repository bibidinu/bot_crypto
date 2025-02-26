"""
Modulo per il calcolo e l'analisi delle performance di trading
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import statistics
import math
from collections import defaultdict
import json
import os

from utils.logger import get_logger
from config.settings import (
    PROFIT_THRESHOLD_PERCENT, MIN_TRADES_FOR_EVALUATION,
    BOT_MODE, BotMode
)

logger = get_logger(__name__)

class PerformanceCalculator:
    """Classe per il calcolo e l'analisi delle performance di trading"""
    
    def __init__(self):
        """Inizializza il calcolatore di performance"""
        self.logger = get_logger(__name__)
        
        # Storico dei trade completati
        self.trade_history = []
        
        # Statistiche globali
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Statistiche per simbolo
        self.symbol_stats = defaultdict(lambda: {
            "total_profit": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_holding_time": 0.0
        })
        
        # Statistiche temporali (giornaliere, settimanali, mensili, annuali)
        self.daily_stats = {}
        self.weekly_stats = {}
        self.monthly_stats = {}
        self.yearly_stats = {}
        
        # Flag per il cambio da demo a live
        self.demo_to_live_ready = False
        
        # Verifica se esiste già uno storico dei trade
        self._load_trade_history()
        
        self.logger.info("PerformanceCalculator inizializzato")
    
    def _load_trade_history(self, filepath: str = "data/trade_history.json") -> None:
        """
        Carica lo storico dei trade da file
        
        Args:
            filepath: Percorso del file
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.trade_history = json.load(f)
                
                # Ricalcola le statistiche
                self._recalculate_stats()
                
                self.logger.info(f"Storico dei trade caricato: {len(self.trade_history)} trade")
            else:
                self.logger.info("Nessuno storico dei trade trovato")
        except Exception as e:
            self.logger.error(f"Errore nel caricamento dello storico dei trade: {str(e)}")
    
    def _save_trade_history(self, filepath: str = "data/trade_history.json") -> None:
        """
        Salva lo storico dei trade su file
        
        Args:
            filepath: Percorso del file
        """
        try:
            # Crea la directory se non esiste
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
                
            self.logger.info(f"Storico dei trade salvato: {len(self.trade_history)} trade")
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio dello storico dei trade: {str(e)}")
    
    def _recalculate_stats(self) -> None:
        """
        Ricalcola tutte le statistiche
        """
        # Resetta le statistiche
        self.total_profit = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.symbol_stats = defaultdict(lambda: {
            "total_profit": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "avg_holding_time": 0.0
        })
        self.daily_stats = {}
        self.weekly_stats = {}
        self.monthly_stats = {}
        self.yearly_stats = {}
        
        # Ricalcola tutte le statistiche per ogni trade
        for trade in self.trade_history:
            self.total_trades += 1
            profit = trade.get("profit", 0.0)
            self.total_profit += profit
            
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Aggiorna statistiche per simbolo
            symbol = trade.get("symbol", "Unknown")
            self._update_symbol_stats(symbol, trade)
            
            # Aggiorna statistiche temporali
            self._update_time_stats(trade)
        
        # Calcola statistiche derivate
        for symbol, stats in self.symbol_stats.items():
            if stats["total_trades"] > 0:
                stats["win_rate"] = stats["winning_trades"] / stats["total_trades"]
                
                total_win = stats["winning_trades"] * stats["avg_win"] if stats["winning_trades"] > 0 else 0
                total_loss = abs(stats["losing_trades"] * stats["avg_loss"]) if stats["losing_trades"] > 0 else 0
                
                if total_loss > 0:
                    stats["profit_factor"] = total_win / total_loss
                else:
                    stats["profit_factor"] = float('inf') if total_win > 0 else 0.0
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Aggiunge un trade completato allo storico
        
        Args:
            trade_data: Dati del trade completato
        """
        try:
            # Assicurati che il trade abbia tutti i campi necessari
            required_fields = ["symbol", "entry_type", "entry_price", "exit_price", "size", "profit", "entry_time", "exit_time"]
            
            for field in required_fields:
                if field not in trade_data:
                    self.logger.warning(f"Campo mancante nei dati del trade: {field}")
                    if field in ["entry_time", "exit_time"]:
                        trade_data[field] = datetime.now().isoformat()
                    else:
                        trade_data[field] = None
            
            # Calcola la durata del trade
            if isinstance(trade_data["entry_time"], str):
                entry_time = datetime.fromisoformat(trade_data["entry_time"])
            else:
                entry_time = trade_data["entry_time"]
                
            if isinstance(trade_data["exit_time"], str):
                exit_time = datetime.fromisoformat(trade_data["exit_time"])
            else:
                exit_time = trade_data["exit_time"]
                
            duration = (exit_time - entry_time).total_seconds() / 3600  # in ore
            trade_data["duration"] = duration
            
            # Aggiungi un ID univoco
            trade_data["id"] = f"{len(self.trade_history) + 1}_{int(datetime.now().timestamp())}"
            
            # Aggiungi il trade allo storico
            self.trade_history.append(trade_data)
            
            # Aggiorna le statistiche
            self.total_trades += 1
            profit = trade_data.get("profit", 0.0)
            self.total_profit += profit
            
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Aggiorna statistiche per simbolo
            symbol = trade_data.get("symbol", "Unknown")
            self._update_symbol_stats(symbol, trade_data)
            
            # Aggiorna statistiche temporali
            self._update_time_stats(trade_data)
            
            # Salva lo storico aggiornato
            self._save_trade_history()
            
            self.logger.info(f"Trade aggiunto: {symbol}, profit={profit:.2f}")
            
            # Verifica se è possibile passare da demo a live
            self._check_demo_to_live()
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta del trade: {str(e)}")
    
    def _update_symbol_stats(self, symbol: str, trade_data: Dict[str, Any]) -> None:
        """
        Aggiorna le statistiche per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            trade_data: Dati del trade
        """
        stats = self.symbol_stats[symbol]
        
        # Aggiorna le statistiche base
        stats["total_trades"] += 1
        profit = trade_data.get("profit", 0.0)
        stats["total_profit"] += profit
        
        if profit > 0:
            stats["winning_trades"] += 1
            stats["max_profit"] = max(stats["max_profit"], profit)
            
            # Aggiorna media dei profitti
            if stats["winning_trades"] == 1:
                stats["avg_win"] = profit
            else:
                stats["avg_win"] = ((stats["avg_win"] * (stats["winning_trades"] - 1)) + profit) / stats["winning_trades"]
                
        else:
            stats["losing_trades"] += 1
            stats["max_loss"] = min(stats["max_loss"], profit)
            
            # Aggiorna media delle perdite
            if stats["losing_trades"] == 1:
                stats["avg_loss"] = profit
            else:
                stats["avg_loss"] = ((stats["avg_loss"] * (stats["losing_trades"] - 1)) + profit) / stats["losing_trades"]
        
        # Calcola win rate
        stats["win_rate"] = stats["winning_trades"] / stats["total_trades"]
        
        # Calcola profit factor
        total_win = stats["winning_trades"] * stats["avg_win"] if stats["winning_trades"] > 0 else 0
        total_loss = abs(stats["losing_trades"] * stats["avg_loss"]) if stats["losing_trades"] > 0 else 0
        
        if total_loss > 0:
            stats["profit_factor"] = total_win / total_loss
        else:
            stats["profit_factor"] = float('inf') if total_win > 0 else 0.0
        
        # Calcola massime sequenze vincenti e perdenti
        # Per questo serve analizzare la sequenza completa dei trade
        symbol_trades = [t for t in self.trade_history if t.get("symbol") == symbol]
        symbol_trades.sort(key=lambda t: t.get("exit_time"))
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for t in symbol_trades:
            if t.get("profit", 0.0) > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            else:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
        
        stats["max_consecutive_wins"] = max_consecutive_wins
        stats["max_consecutive_losses"] = max_consecutive_losses
        
        # Calcola tempo medio di detenzione
        durations = [t.get("duration", 0.0) for t in symbol_trades]
        if durations:
            stats["avg_holding_time"] = sum(durations) / len(durations)
    
    def _update_time_stats(self, trade_data: Dict[str, Any]) -> None:
        """
        Aggiorna le statistiche temporali
        
        Args:
            trade_data: Dati del trade
        """
        # Estrai la data di uscita
        if isinstance(trade_data.get("exit_time"), str):
            exit_time = datetime.fromisoformat(trade_data["exit_time"])
        else:
            exit_time = trade_data.get("exit_time", datetime.now())
        
        # Chiavi per le statistiche temporali
        day_key = exit_time.strftime("%Y-%m-%d")
        week_key = f"{exit_time.isocalendar()[0]}-W{exit_time.isocalendar()[1]}"
        month_key = exit_time.strftime("%Y-%m")
        year_key = exit_time.strftime("%Y")
        
        # Aggiorna statistiche giornaliere
        if day_key not in self.daily_stats:
            self.daily_stats[day_key] = {
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        
        self.daily_stats[day_key]["total_trades"] += 1
        profit = trade_data.get("profit", 0.0)
        self.daily_stats[day_key]["total_profit"] += profit
        
        if profit > 0:
            self.daily_stats[day_key]["winning_trades"] += 1
        else:
            self.daily_stats[day_key]["losing_trades"] += 1
        
        # Aggiorna statistiche settimanali
        if week_key not in self.weekly_stats:
            self.weekly_stats[week_key] = {
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        
        self.weekly_stats[week_key]["total_trades"] += 1
        self.weekly_stats[week_key]["total_profit"] += profit
        
        if profit > 0:
            self.weekly_stats[week_key]["winning_trades"] += 1
        else:
            self.weekly_stats[week_key]["losing_trades"] += 1
        
        # Aggiorna statistiche mensili
        if month_key not in self.monthly_stats:
            self.monthly_stats[month_key] = {
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        
        self.monthly_stats[month_key]["total_trades"] += 1
        self.monthly_stats[month_key]["total_profit"] += profit
        
        if profit > 0:
            self.monthly_stats[month_key]["winning_trades"] += 1
        else:
            self.monthly_stats[month_key]["losing_trades"] += 1
        
        # Aggiorna statistiche annuali
        if year_key not in self.yearly_stats:
            self.yearly_stats[year_key] = {
                "total_profit": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0
            }
        
        self.yearly_stats[year_key]["total_trades"] += 1
        self.yearly_stats[year_key]["total_profit"] += profit
        
        if profit > 0:
            self.yearly_stats[year_key]["winning_trades"] += 1
        else:
            self.yearly_stats[year_key]["losing_trades"] += 1
    
    def _check_demo_to_live(self) -> None:
        """
        Verifica se è possibile passare da demo a live
        """
        if BOT_MODE != BotMode.DEMO or self.demo_to_live_ready:
            return
            
        # Verifica se ci sono abbastanza trade
        if self.total_trades < MIN_TRADES_FOR_EVALUATION:
            return
            
        # Calcola il win rate globale
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        win_rate_percent = win_rate * 100
        
        # Verifica se il win rate è sufficientemente alto
        if win_rate_percent >= PROFIT_THRESHOLD_PERCENT:
            self.demo_to_live_ready = True
            self.logger.info(f"Il bot è pronto per passare a live trading! Win rate: {win_rate_percent:.2f}%")
    
    def get_stats_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene le statistiche per un simbolo specifico
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Statistiche per il simbolo
        """
        return dict(self.symbol_stats.get(symbol, {}))
    
    def get_daily_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Ottiene le statistiche giornaliere
        
        Args:
            days: Numero di giorni da includere
            
        Returns:
            Statistiche giornaliere
        """
        # Ordina le chiavi per data
        sorted_keys = sorted(self.daily_stats.keys(), reverse=True)
        
        # Prendi solo i giorni richiesti
        recent_keys = sorted_keys[:days]
        
        # Crea un dizionario con le statistiche recenti
        result = {}
        for key in recent_keys:
            result[key] = self.daily_stats[key]
            
        return result
    
    def get_weekly_stats(self, weeks: int = 4) -> Dict[str, Any]:
        """
        Ottiene le statistiche settimanali
        
        Args:
            weeks: Numero di settimane da includere
            
        Returns:
            Statistiche settimanali
        """
        # Ordina le chiavi per data
        sorted_keys = sorted(self.weekly_stats.keys(), reverse=True)
        
        # Prendi solo le settimane richieste
        recent_keys = sorted_keys[:weeks]
        
        # Crea un dizionario con le statistiche recenti
        result = {}
        for key in recent_keys:
            result[key] = self.weekly_stats[key]
            
        return result
    
    def get_monthly_stats(self, months: int = 6) -> Dict[str, Any]:
        """
        Ottiene le statistiche mensili
        
        Args:
            months: Numero di mesi da includere
            
        Returns:
            Statistiche mensili
        """
        # Ordina le chiavi per data
        sorted_keys = sorted(self.monthly_stats.keys(), reverse=True)
        
        # Prendi solo i mesi richiesti
        recent_keys = sorted_keys[:months]
        
        # Crea un dizionario con le statistiche recenti
        result = {}
        for key in recent_keys:
            result[key] = self.monthly_stats[key]
            
        return result
    
    def get_yearly_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche annuali
        
        Returns:
            Statistiche annuali
        """
        # Ordina le chiavi per data
        sorted_keys = sorted(self.yearly_stats.keys(), reverse=True)
        
        # Crea un dizionario con le statistiche
        result = {}
        for key in sorted_keys:
            result[key] = self.yearly_stats[key]
            
        return result
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche globali
        
        Returns:
            Statistiche globali
        """
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calcola profit factor
        winning_profits = [t.get("profit", 0.0) for t in self.trade_history if t.get("profit", 0.0) > 0]
        losing_profits = [t.get("profit", 0.0) for t in self.trade_history if t.get("profit", 0.0) <= 0]
        
        total_win = sum(winning_profits)
        total_loss = abs(sum(losing_profits))
        
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf') if total_win > 0 else 0.0
        
        # Calcola media profitti e perdite
        avg_win = statistics.mean(winning_profits) if winning_profits else 0.0
        avg_loss = statistics.mean(losing_profits) if losing_profits else 0.0
        
        # Calcola drawdown massimo
        cumulative = 0.0
        peak = 0.0
        drawdown = 0.0
        max_drawdown = 0.0
        
        # Ordina i trade per data di uscita
        sorted_trades = sorted(self.trade_history, key=lambda t: t.get("exit_time"))
        
        for trade in sorted_trades:
            profit = trade.get("profit", 0.0)
            cumulative += profit
            
            if cumulative > peak:
                peak = cumulative
                drawdown = 0.0
            else:
                drawdown = peak - cumulative
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calcola Sharpe ratio
        if self.total_trades > 0:
            returns = [t.get("profit", 0.0) for t in self.trade_history]
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0
            
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Verifica lo stato di passaggio a live
        demo_to_live_status = {
            "ready": self.demo_to_live_ready,
            "win_rate": win_rate * 100,
            "threshold": PROFIT_THRESHOLD_PERCENT,
            "trades": self.total_trades,
            "required_trades": MIN_TRADES_FOR_EVALUATION
        }
        
        return {
            "total_profit": self.total_profit,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "drawdown_percent": (max_drawdown / peak) * 100 if peak > 0 else 0.0,
            "sharpe_ratio": sharpe_ratio,
            "demo_to_live": demo_to_live_status
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ottiene i trade più recenti
        
        Args:
            limit: Numero massimo di trade da restituire
            
        Returns:
            Lista dei trade più recenti
        """
        # Ordina i trade per data di uscita (più recenti prima)
        sorted_trades = sorted(self.trade_history, key=lambda t: t.get("exit_time"), reverse=True)
        
        return sorted_trades[:limit]
    
    def get_best_performing_symbols(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Ottiene i simboli con le migliori performance
        
        Args:
            limit: Numero massimo di simboli da restituire
            
        Returns:
            Lista dei simboli con le migliori performance
        """
        # Calcola le performance per ogni simbolo
        symbol_performance = []
        
        for symbol, stats in self.symbol_stats.items():
            if stats["total_trades"] > 0:
                symbol_performance.append({
                    "symbol": symbol,
                    "total_profit": stats["total_profit"],
                    "win_rate": stats["win_rate"],
                    "profit_factor": stats["profit_factor"],
                    "total_trades": stats["total_trades"]
                })
        
        # Ordina per profitto totale (più alto prima)
        symbol_performance.sort(key=lambda s: s["total_profit"], reverse=True)
        
        return symbol_performance[:limit]
    
    def get_performance_graph_data(self, period: str = "daily") -> Dict[str, List]:
        """
        Ottiene i dati per il grafico delle performance
        
        Args:
            period: Periodo per il grafico ('daily', 'weekly', 'monthly', 'yearly')
            
        Returns:
            Dati per il grafico
        """
        if period == "daily":
            stats = self.daily_stats
        elif period == "weekly":
            stats = self.weekly_stats
        elif period == "monthly":
            stats = self.monthly_stats
        elif period == "yearly":
            stats = self.yearly_stats
        else:
            stats = self.daily_stats
        
        # Ordina le chiavi per data
        sorted_keys = sorted(stats.keys())
        
        # Prepara i dati per il grafico
        dates = []
        profits = []
        trade_counts = []
        win_rates = []
        
        # Aggiungi i dati cumulativi
        cumulative_profit = 0.0
        
        for key in sorted_keys:
            stat = stats[key]
            
            dates.append(key)
            
            # Profitto del periodo
            profit = stat["total_profit"]
            profits.append(profit)
            
            # Profitto cumulativo
            cumulative_profit += profit
            
            # Numero di trade
            trade_counts.append(stat["total_trades"])
            
            # Win rate
            win_rate = stat["winning_trades"] / stat["total_trades"] if stat["total_trades"] > 0 else 0
            win_rates.append(win_rate * 100)
        
        return {
            "dates": dates,
            "profits": profits,
            "cumulative_profits": [sum(profits[:i+1]) for i in range(len(profits))],
            "trade_counts": trade_counts,
            "win_rates": win_rates
        }
    
    def calculate_optimal_position_size(self, symbol: str, risk_per_trade: float = 1.0) -> float:
        """
        Calcola la dimensione ottimale della posizione in base alle performance passate
        
        Args:
            symbol: Simbolo della coppia
            risk_per_trade: Percentuale di rischio per trade
            
        Returns:
            Dimensione ottimale della posizione
        """
        # Ottieni le statistiche per il simbolo
        stats = self.get_stats_for_symbol(symbol)
        
        if not stats or stats.get("total_trades", 0) < 10:
            # Non abbastanza dati, usa il rischio base
            return risk_per_trade
        
        # Calcola il Kelly Criterion
        win_rate = stats.get("win_rate", 0.0)
        
        if win_rate <= 0:
            return risk_per_trade
        
        avg_win = stats.get("avg_win", 0.0)
        avg_loss = abs(stats.get("avg_loss", 0.0))
        
        if avg_loss <= 0:
            return risk_per_trade
        
        # Kelly Criterion = Win Rate - ((1 - Win Rate) / (Win/Loss Ratio))
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Limita il Kelly fraction a un massimo (di solito si usa 1/2 o 1/4 di Kelly)
        half_kelly = max(0, kelly / 2)
        
        # Converte il Kelly fraction in percentuale di rischio
        optimal_risk = min(half_kelly * 100, risk_per_trade * 2)
        
        return optimal_risk
    
    def should_trade_symbol(self, symbol: str, min_trades: int = 10) -> Tuple[bool, str]:
        """
        Determina se è consigliabile tradare un simbolo in base alle performance passate
        
        Args:
            symbol: Simbolo della coppia
            min_trades: Numero minimo di trade per una valutazione affidabile
            
        Returns:
            Tupla (tradare, motivo)
        """
        # Ottieni le statistiche per il simbolo
        stats = self.get_stats_for_symbol(symbol)
        
        if not stats or stats.get("total_trades", 0) < min_trades:
            return True, "Dati insufficienti per una valutazione"
        
        # Verifica il win rate
        win_rate = stats.get("win_rate", 0.0)
        
        if win_rate < 0.4:
            return False, f"Win rate troppo basso: {win_rate:.2%}"
        
        # Verifica il profit factor
        profit_factor = stats.get("profit_factor", 0.0)
        
        if profit_factor < 1.0:
            return False, f"Profit factor insufficiente: {profit_factor:.2f}"
        
        # Verifica le sequenze di perdite
        max_losses = stats.get("max_consecutive_losses", 0)
        
        if max_losses > 5:
            return False, f"Troppe perdite consecutive: {max_losses}"
        
        return True, "Simbolo performante"
