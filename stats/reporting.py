"""
Modulo per la generazione di report sulle performance di trading
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from utils.logger import get_logger
from stats.performance import PerformanceCalculator
from config.settings import (
    BOT_NAME, BOT_MODE, BotMode, 
    PERFORMANCE_REPORT_INTERVAL
)

logger = get_logger(__name__)

class ReportGenerator:
    """Classe per la generazione di report sulle performance di trading"""
    
    def __init__(self, performance_calculator: PerformanceCalculator):
        """
        Inizializza il generatore di report
        
        Args:
            performance_calculator: Istanza del calcolatore di performance
        """
        self.logger = get_logger(__name__)
        
        self.performance = performance_calculator
        
        # Directory per i report
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Directory per i grafici
        self.charts_dir = os.path.join(self.reports_dir, "charts")
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Tema per i grafici
        plt.style.use('seaborn-v0_8-darkgrid')
        
        self.logger.info("ReportGenerator inizializzato")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Genera un report di riepilogo delle performance
        
        Returns:
            Dizionario con il report di riepilogo
        """
        # Ottieni le statistiche globali
        overall_stats = self.performance.get_overall_stats()
        
        # Ottieni i simboli con le migliori performance
        best_symbols = self.performance.get_best_performing_symbols(5)
        
        # Ottieni le statistiche giornaliere recenti
        daily_stats = self.performance.get_daily_stats(7)
        
        # Ottieni i trade più recenti
        recent_trades = self.performance.get_recent_trades(10)
        
        # Crea il report di riepilogo
        report = {
            "bot_name": BOT_NAME,
            "bot_mode": BOT_MODE.value,
            "generated_at": datetime.now().isoformat(),
            "overall_stats": overall_stats,
            "best_symbols": best_symbols,
            "daily_stats": daily_stats,
            "recent_trades": recent_trades
        }
        
        return report
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        Genera un report dettagliato delle performance
        
        Returns:
            Dizionario con il report dettagliato
        """
        # Ottieni le statistiche globali
        overall_stats = self.performance.get_overall_stats()
        
        # Ottieni le statistiche per tutti i simboli
        symbol_stats = {}
        for symbol in self.performance.symbol_stats.keys():
            symbol_stats[symbol] = self.performance.get_stats_for_symbol(symbol)
        
        # Ottieni le statistiche temporali
        daily_stats = self.performance.get_daily_stats(30)
        weekly_stats = self.performance.get_weekly_stats(12)
        monthly_stats = self.performance.get_monthly_stats(12)
        yearly_stats = self.performance.get_yearly_stats()
        
        # Ottieni tutti i trade
        all_trades = self.performance.trade_history
        
        # Crea il report dettagliato
        report = {
            "bot_name": BOT_NAME,
            "bot_mode": BOT_MODE.value,
            "generated_at": datetime.now().isoformat(),
            "overall_stats": overall_stats,
            "symbol_stats": symbol_stats,
            "daily_stats": daily_stats,
            "weekly_stats": weekly_stats,
            "monthly_stats": monthly_stats,
            "yearly_stats": yearly_stats,
            "all_trades": all_trades
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], report_type: str = "summary") -> str:
        """
        Salva un report su file
        
        Args:
            report: Dizionario con il report
            report_type: Tipo di report ("summary" o "detailed")
            
        Returns:
            Percorso del file salvato
        """
        try:
            # Genera il nome del file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_type}_report_{timestamp}.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            # Salva il report in formato JSON
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.logger.info(f"Report salvato in {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio del report: {str(e)}")
            return ""
    
    def generate_performance_chart(self, period: str = "daily", save: bool = True) -> Optional[str]:
        """
        Genera un grafico delle performance
        
        Args:
            period: Periodo per il grafico ('daily', 'weekly', 'monthly', 'yearly')
            save: Se salvare il grafico su file
            
        Returns:
            Percorso del file salvato o None
        """
        try:
            # Ottieni i dati per il grafico
            data = self.performance.get_performance_graph_data(period)
            
            if not data or not data["dates"]:
                self.logger.warning(f"Nessun dato disponibile per il grafico {period}")
                return None
            
            # Converti le date in oggetti datetime
            if period == "daily":
                dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["dates"]]
                date_format = "%d %b"
                title_period = "Daily"
            elif period == "weekly":
                # Formato: "2023-W01"
                dates = []
                for week_str in data["dates"]:
                    year, week = week_str.split("-W")
                    # Usa il primo giorno della settimana
                    date = datetime.strptime(f"{year}-{week}-1", "%Y-%W-%w")
                    dates.append(date)
                date_format = "%d %b"
                title_period = "Weekly"
            elif period == "monthly":
                dates = [datetime.strptime(d, "%Y-%m") for d in data["dates"]]
                date_format = "%b %Y"
                title_period = "Monthly"
            elif period == "yearly":
                dates = [datetime.strptime(d, "%Y") for d in data["dates"]]
                date_format = "%Y"
                title_period = "Yearly"
            else:
                dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["dates"]]
                date_format = "%d %b"
                title_period = "Daily"
            
            # Crea il grafico
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Grafico 1: Profitto cumulativo
            ax1.plot(dates, data["cumulative_profits"], 'b-', linewidth=2)
            ax1.set_title(f"{title_period} Cumulative Profit")
            ax1.set_ylabel("Profit (USDT)")
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax1.grid(True)
            
            # Aggiungi una linea orizzontale a 0
            ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Grafico 2: Profitto per periodo
            ax2.bar(dates, data["profits"], color=['g' if p > 0 else 'r' for p in data["profits"]])
            ax2.set_title(f"{title_period} Profit")
            ax2.set_ylabel("Profit (USDT)")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax2.grid(True)
            
            # Grafico 3: Win rate
            ax3.plot(dates, data["win_rates"], 'g-', linewidth=2)
            ax3.set_title(f"{title_period} Win Rate")
            ax3.set_ylabel("Win Rate (%)")
            ax3.set_ylim(0, 100)
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax3.grid(True)
            
            # Formatta l'asse x
            plt.xticks(rotation=45)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
            
            # Aggiungi informazioni al titolo
            overall_stats = self.performance.get_overall_stats()
            fig.suptitle(f"{BOT_NAME} Performance Report\n"
                        f"Total Profit: {overall_stats['total_profit']:.2f} USDT, "
                        f"Win Rate: {overall_stats['win_rate'] * 100:.2f}%, "
                        f"Trades: {overall_stats['total_trades']}", 
                        fontsize=16)
            
            # Aggiungi padding
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            if save:
                # Salva il grafico
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"performance_{period}_{timestamp}.png"
                filepath = os.path.join(self.charts_dir, filename)
                
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Grafico salvato in {filepath}")
                
                return filepath
            else:
                plt.show()
                plt.close(fig)
                return None
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione del grafico: {str(e)}")
            return None
    
    def generate_symbol_performance_chart(self, symbol: str, save: bool = True) -> Optional[str]:
        """
        Genera un grafico delle performance per un simbolo specifico
        
        Args:
            symbol: Simbolo della coppia
            save: Se salvare il grafico su file
            
        Returns:
            Percorso del file salvato o None
        """
        try:
            # Ottieni i trade per il simbolo
            symbol_trades = [t for t in self.performance.trade_history if t.get("symbol") == symbol]
            
            if not symbol_trades:
                self.logger.warning(f"Nessun trade disponibile per il simbolo {symbol}")
                return None
            
            # Ottieni le statistiche per il simbolo
            stats = self.performance.get_stats_for_symbol(symbol)
            
            # Ordina i trade per data di uscita
            symbol_trades.sort(key=lambda t: t.get("exit_time"))
            
            # Preparare i dati per il grafico
            dates = []
            profits = []
            cumulative_profits = []
            entry_prices = []
            exit_prices = []
            trade_types = []
            
            cumulative = 0.0
            
            for trade in symbol_trades:
                if isinstance(trade.get("exit_time"), str):
                    exit_time = datetime.fromisoformat(trade["exit_time"])
                else:
                    exit_time = trade.get("exit_time", datetime.now())
                    
                dates.append(exit_time)
                
                profit = trade.get("profit", 0.0)
                profits.append(profit)
                
                cumulative += profit
                cumulative_profits.append(cumulative)
                
                entry_prices.append(trade.get("entry_price", 0.0))
                exit_prices.append(trade.get("exit_price", 0.0))
                
                trade_types.append(trade.get("entry_type", "unknown"))
            
            # Crea il grafico
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Grafico 1: Profitto cumulativo
            ax1.plot(dates, cumulative_profits, 'b-', linewidth=2)
            ax1.set_title(f"{symbol} Cumulative Profit")
            ax1.set_ylabel("Profit (USDT)")
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax1.grid(True)
            
            # Aggiungi una linea orizzontale a 0
            ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            # Grafico 2: Profitto per trade
            colors = ['g' if p > 0 else 'r' for p in profits]
            ax2.bar(dates, profits, color=colors)
            ax2.set_title(f"{symbol} Trade Profits")
            ax2.set_ylabel("Profit (USDT)")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))
            ax2.grid(True)
            
            # Grafico 3: Prezzi
            ax3.plot(dates, entry_prices, 'b-', label='Entry', linewidth=1)
            ax3.plot(dates, exit_prices, 'g-', label='Exit', linewidth=1)
            ax3.set_title(f"{symbol} Prices")
            ax3.set_ylabel("Price (USDT)")
            ax3.legend()
            ax3.grid(True)
            
            # Formatta l'asse x
            plt.xticks(rotation=45)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
            
            # Aggiungi informazioni al titolo
            fig.suptitle(f"{symbol} Performance Report\n"
                        f"Total Profit: {stats.get('total_profit', 0.0):.2f} USDT, "
                        f"Win Rate: {stats.get('win_rate', 0.0) * 100:.2f}%, "
                        f"Trades: {stats.get('total_trades', 0)}", 
                        fontsize=16)
            
            # Aggiungi padding
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            if save:
                # Salva il grafico
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol.replace('/', '_')}_{timestamp}.png"
                filepath = os.path.join(self.charts_dir, filename)
                
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Grafico salvato in {filepath}")
                
                return filepath
            else:
                plt.show()
                plt.close(fig)
                return None
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione del grafico per {symbol}: {str(e)}")
            return None
    
    def generate_win_loss_distribution_chart(self, save: bool = True) -> Optional[str]:
        """
        Genera un grafico della distribuzione di profitti e perdite
        
        Args:
            save: Se salvare il grafico su file
            
        Returns:
            Percorso del file salvato o None
        """
        try:
            # Ottieni tutti i profitti
            profits = [t.get("profit", 0.0) for t in self.performance.trade_history]
            
            if not profits:
                self.logger.warning("Nessun profitto disponibile per il grafico")
                return None
            
            # Crea il grafico
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Grafico della distribuzione
            sns.histplot(profits, bins=30, kde=True, ax=ax)
            ax.set_title("Profit/Loss Distribution")
            ax.set_xlabel("Profit/Loss (USDT)")
            ax.set_ylabel("Frequency")
            
            # Aggiungi una linea verticale a 0
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            
            # Aggiungi statistiche
            overall_stats = self.performance.get_overall_stats()
            textstr = (f"Mean: {np.mean(profits):.2f} USDT\n"
                      f"Median: {np.median(profits):.2f} USDT\n"
                      f"Std Dev: {np.std(profits):.2f} USDT\n"
                      f"Win Rate: {overall_stats['win_rate'] * 100:.2f}%\n"
                      f"Profit Factor: {overall_stats['profit_factor']:.2f}")
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            if save:
                # Salva il grafico
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"win_loss_distribution_{timestamp}.png"
                filepath = os.path.join(self.charts_dir, filename)
                
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                self.logger.info(f"Grafico salvato in {filepath}")
                
                return filepath
            else:
                plt.show()
                plt.close(fig)
                return None
                
        except Exception as e:
            self.logger.error(f"Errore nella generazione del grafico di distribuzione: {str(e)}")
            return None
    
    def generate_all_charts(self) -> List[str]:
        """
        Genera tutti i grafici
        
        Returns:
            Lista dei percorsi dei file salvati
        """
        charts = []
        
        # Genera i grafici delle performance
        for period in ["daily", "weekly", "monthly", "yearly"]:
            chart = self.generate_performance_chart(period)
            if chart:
                charts.append(chart)
        
        # Genera i grafici per i simboli con le migliori performance
        best_symbols = self.performance.get_best_performing_symbols(5)
        for symbol_data in best_symbols:
            symbol = symbol_data.get("symbol")
            chart = self.generate_symbol_performance_chart(symbol)
            if chart:
                charts.append(chart)
        
        # Genera il grafico della distribuzione di profitti e perdite
        chart = self.generate_win_loss_distribution_chart()
        if chart:
            charts.append(chart)
        
        return charts
    
    def generate_periodic_report(self) -> Dict[str, Any]:
        """
        Genera un report periodico completo
        
        Returns:
            Dizionario con il report e i percorsi dei grafici
        """
        # Genera il report dettagliato
        report = self.generate_detailed_report()
        
        # Salva il report
        report_path = self.save_report(report, "detailed")
        
        # Genera tutti i grafici
        chart_paths = self.generate_all_charts()
        
        return {
            "report": report,
            "report_path": report_path,
            "chart_paths": chart_paths
        }
