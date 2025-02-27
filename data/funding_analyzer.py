"""
Modulo per l'analisi dei tassi di funding nei futures perpetui
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
from collections import defaultdict

from api.bybit_api import BybitAPI
from utils.logger import get_logger
from utils.helpers import format_number

logger = get_logger(__name__)

class FundingAnalyzer:
    """
    Classe per l'analisi dei tassi di funding nei futures perpetui.
    I futures perpetui hanno un tasso di funding periodico (solitamente ogni 8 ore)
    che viene pagato o ricevuto dai trader in posizione. Questo meccanismo
    mantiene il prezzo del future allineato con quello spot.
    """
    
    def __init__(self, exchange: BybitAPI):
        """
        Inizializza l'analizzatore dei tassi di funding
        
        Args:
            exchange: Istanza dell'exchange API
        """
        self.exchange = exchange
        self.logger = get_logger(__name__)
        
        # Cache dei dati di funding
        self.funding_data_cache = {}
        self.last_update_time = {}
        
        # Cache dei prossimi pagamenti di funding
        self.next_funding_cache = {}
        
        # Thread per l'aggiornamento asincrono
        self.update_thread = None
        self.running = False
        
        # Coda di aggiornamento
        self.update_queue = queue.Queue()
        
        # Registrazione delle statistiche di funding
        self.funding_stats = defaultdict(lambda: {
            "avg_rate": 0.0,
            "avg_rate_positive": 0.0,  # Media dei tassi positivi
            "avg_rate_negative": 0.0,  # Media dei tassi negativi
            "max_rate": 0.0,
            "min_rate": 0.0,
            "volatility": 0.0,  # Deviazione standard dei tassi
            "samples": 0
        })
        
        self.logger.info("FundingAnalyzer inizializzato")
    
    def start_background_updates(self) -> None:
        """
        Avvia il thread di aggiornamento in background
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            self.logger.warning("Thread di aggiornamento funding già in esecuzione")
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._background_update_worker)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Thread di aggiornamento funding avviato")
    
    def stop_background_updates(self) -> None:
        """
        Ferma il thread di aggiornamento in background
        """
        self.running = False
        
        if self.update_thread is not None:
            try:
                self.update_thread.join(timeout=5.0)
            except Exception:
                pass
                
        self.logger.info("Thread di aggiornamento funding fermato")
    
    def _background_update_worker(self) -> None:
        """Thread worker per aggiornamenti in background"""
        self.logger.info("Worker di aggiornamento funding avviato")
        
        while self.running:
            try:
                # Elabora le richieste in coda
                try:
                    symbol = self.update_queue.get(timeout=0.1)
                    self.get_funding_history(symbol, force_update=True)
                    self.update_queue.task_done()
                except queue.Empty:
                    pass
                
                # Aggiorna le informazioni sui prossimi funding ogni 5 minuti
                for symbol in list(self.next_funding_cache.keys()):
                    cache_entry = self.next_funding_cache.get(symbol, {})
                    if not cache_entry or datetime.now() - cache_entry.get("timestamp", datetime.min) > timedelta(minutes=5):
                        try:
                            self.get_next_funding(symbol, True)
                        except Exception as e:
                            self.logger.error(f"Errore nell'aggiornamento del funding per {symbol}: {str(e)}")
                
                # Dormi un po' per non sovraccaricare
                time.sleep(5.0)
            
            except Exception as e:
                self.logger.error(f"Errore nel thread di aggiornamento funding: {str(e)}")
                time.sleep(10.0)
    
    def get_funding_history(self, symbol: str, limit: int = 200, 
                          force_update: bool = False) -> pd.DataFrame:
        """
        Ottiene la cronologia dei tassi di funding per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            limit: Numero massimo di record da recuperare
            force_update: Se forzare un aggiornamento dalla cache
            
        Returns:
            DataFrame con la cronologia dei tassi di funding
        """
        # Verifica la cache
        if not force_update and symbol in self.funding_data_cache:
            last_update = self.last_update_time.get(symbol, datetime.min)
            if datetime.now() - last_update < timedelta(hours=1):
                return self.funding_data_cache[symbol]
        
        try:
            # Ottieni i dati di funding dall'exchange
            funding_data = self.exchange.get_funding_rate(symbol, limit=limit)
            
            if not funding_data:
                self.logger.warning(f"Nessun dato di funding disponibile per {symbol}")
                return pd.DataFrame()
            
            # Converti in DataFrame
            rates = []
            for item in funding_data:
                # Converti il timestamp da millisecondi a datetime
                timestamp = datetime.fromtimestamp(int(item.get("fundingRateTimestamp", 0)) / 1000)
                
                # Ottieni il tasso di funding (in percentuale)
                rate = float(item.get("fundingRate", 0)) * 100  # Converti in percentuale
                
                rates.append({
                    "timestamp": timestamp,
                    "funding_rate": rate
                })
            
            df = pd.DataFrame(rates)
            
            # Ordina per timestamp in ordine decrescente
            df = df.sort_values("timestamp", ascending=False)
            
            # Aggiorna la cache
            self.funding_data_cache[symbol] = df
            self.last_update_time[symbol] = datetime.now()
            
            # Aggiorna le statistiche
            self._update_funding_stats(symbol, df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dei tassi di funding per {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _update_funding_stats(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Aggiorna le statistiche di funding per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            df: DataFrame con i dati di funding
        """
        if df.empty:
            return
        
        try:
            rates = df["funding_rate"].values
            
            stats = {
                "avg_rate": np.mean(rates),
                "max_rate": np.max(rates),
                "min_rate": np.min(rates),
                "volatility": np.std(rates),
                "samples": len(rates)
            }
            
            # Calcola medie separate per tassi positivi e negativi
            positive_rates = rates[rates > 0]
            negative_rates = rates[rates < 0]
            
            stats["avg_rate_positive"] = np.mean(positive_rates) if len(positive_rates) > 0 else 0.0
            stats["avg_rate_negative"] = np.mean(negative_rates) if len(negative_rates) > 0 else 0.0
            
            # Aggiorna le statistiche
            self.funding_stats[symbol] = stats
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento delle statistiche di funding per {symbol}: {str(e)}")
    
    def get_next_funding(self, symbol: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Ottiene informazioni sul prossimo pagamento di funding
        
        Args:
            symbol: Simbolo della coppia
            force_update: Se forzare un aggiornamento dalla cache
            
        Returns:
            Informazioni sul prossimo funding
        """
        # Verifica la cache
        if not force_update and symbol in self.next_funding_cache:
            cache_entry = self.next_funding_cache[symbol]
            if datetime.now() - cache_entry.get("timestamp", datetime.min) < timedelta(minutes=5):
                return cache_entry
        
        try:
            # Ottieni il ticker che contiene i dati di funding
            ticker = self.exchange.get_ticker(symbol)
            
            # Estrai il tasso di funding previsto e il timestamp
            predicted_rate = float(ticker.get("fundingRate", 0)) * 100  # Converti in percentuale
            next_funding_time = int(ticker.get("nextFundingTime", 0)) / 1000  # Converti in secondi
            
            # Calcola il countdown
            countdown = max(0, next_funding_time - time.time())
            hours, remainder = divmod(countdown, 3600)
            minutes, seconds = divmod(remainder, 60)
            countdown_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Crea l'oggetto risultato
            result = {
                "symbol": symbol,
                "predicted_rate": predicted_rate,
                "next_funding_time": datetime.fromtimestamp(next_funding_time),
                "countdown": countdown_str,
                "countdown_seconds": countdown,
                "timestamp": datetime.now()
            }
            
            # Aggiorna la cache
            self.next_funding_cache[symbol] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero del prossimo funding per {symbol}: {str(e)}")
            
            # Restituisci dati vuoti in caso di errore
            return {
                "symbol": symbol,
                "predicted_rate": 0.0,
                "next_funding_time": None,
                "countdown": "N/A",
                "countdown_seconds": 0,
                "timestamp": datetime.now()
            }
    
    def get_funding_stats(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene le statistiche di funding per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Statistiche di funding
        """
        # Assicurati che le statistiche siano aggiornate
        if symbol not in self.funding_stats or self.funding_stats[symbol]["samples"] == 0:
            self.get_funding_history(symbol)
        
        return dict(self.funding_stats[symbol])
    
    def analyze_funding_opportunity(self, symbol: str, 
                                  min_rate: float = 0.01, 
                                  smoothing_window: int = 3) -> Dict[str, Any]:
        """
        Analizza se un simbolo offre una buona opportunità di arbitraggio di funding
        
        Args:
            symbol: Simbolo della coppia
            min_rate: Tasso minimo di funding per considerare un'opportunità (%)
            smoothing_window: Finestra di smoothing per le medie
            
        Returns:
            Analisi dell'opportunità di funding
        """
        try:
            # Ottieni la cronologia di funding
            df = self.get_funding_history(symbol)
            
            if df.empty:
                return {"symbol": symbol, "opportunity": False, "reason": "Nessun dato di funding disponibile"}
            
            # Ottieni le informazioni sul prossimo funding
            next_funding = self.get_next_funding(symbol)
            
            # Calcola una media mobile dei tassi di funding
            if len(df) >= smoothing_window:
                avg_rate = df["funding_rate"].head(smoothing_window).mean()
            else:
                avg_rate = df["funding_rate"].mean()
            
            # Ottieni il tasso previsto
            predicted_rate = next_funding.get("predicted_rate", 0.0)
            
            # Determina se è un'opportunità
            is_opportunity = False
            direction = ""
            reason = ""
            
            if abs(predicted_rate) >= min_rate:
                is_opportunity = True
                
                if predicted_rate > 0:
                    # Funding rate positivo: i long pagano agli short
                    # Quindi conviene aprire una posizione short
                    direction = "short"
                    reason = f"Tasso di funding positivo ({format_number(predicted_rate)}%/8h) - gli short ricevono pagamenti"
                else:
                    # Funding rate negativo: gli short pagano ai long
                    # Quindi conviene aprire una posizione long
                    direction = "long"
                    reason = f"Tasso di funding negativo ({format_number(predicted_rate)}%/8h) - i long ricevono pagamenti"
            else:
                reason = f"Tasso di funding troppo basso ({format_number(predicted_rate)}%/8h)"
            
            # Stima il rendimento annualizzato
            # Assumendo che il tasso di funding sia pagato ogni 8 ore (3 volte al giorno)
            annual_return = abs(predicted_rate) * 3 * 365 / 100  # Converti in decimale
            
            # Costruisci il risultato
            result = {
                "symbol": symbol,
                "opportunity": is_opportunity,
                "direction": direction,
                "reason": reason,
                "predicted_rate": predicted_rate,
                "avg_rate": avg_rate,
                "annual_return": annual_return,
                "next_funding_time": next_funding.get("next_funding_time"),
                "countdown": next_funding.get("countdown"),
                "updated_at": datetime.now()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi dell'opportunità di funding per {symbol}: {str(e)}")
            
            return {
                "symbol": symbol,
                "opportunity": False,
                "reason": f"Errore nell'analisi: {str(e)}",
                "predicted_rate": 0.0,
                "avg_rate": 0.0,
                "annual_return": 0.0,
                "updated_at": datetime.now()
            }
    
    def get_best_funding_opportunities(self, symbols: List[str], 
                                     min_rate: float = 0.01,
                                     top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Trova le migliori opportunità di arbitraggio di funding tra un elenco di simboli
        
        Args:
            symbols: Lista di simboli da analizzare
            min_rate: Tasso minimo di funding per considerare un'opportunità (%)
            top_n: Numero massimo di opportunità da restituire
            
        Returns:
            Lista delle migliori opportunità di funding
        """
        opportunities = []
        
        for symbol in symbols:
            try:
                # Metti in coda per aggiornamento asincrono
                self.update_queue.put(symbol)
                
                # Ottieni l'analisi
                analysis = self.analyze_funding_opportunity(symbol, min_rate)
                
                if analysis.get("opportunity", False):
                    opportunities.append(analysis)
                    
            except Exception as e:
                self.logger.error(f"Errore nell'analisi di {symbol}: {str(e)}")
        
        # Ordina per rendimento annualizzato decrescente
        opportunities.sort(key=lambda x: x.get("annual_return", 0.0), reverse=True)
        
        # Restituisci le prime N opportunità
        return opportunities[:top_n]
    
    def calculate_funding_impact(self, symbol: str, position_value: float, 
                               direction: str, holding_periods: int = 3) -> Dict[str, Any]:
        """
        Calcola l'impatto del funding su una posizione
        
        Args:
            symbol: Simbolo della coppia
            position_value: Valore della posizione in USD
            direction: Direzione della posizione ('long' o 'short')
            holding_periods: Numero di periodi di funding da considerare
            
        Returns:
            Impatto del funding sulla posizione
        """
        try:
            # Ottieni le informazioni sul prossimo funding
            next_funding = self.get_next_funding(symbol)
            predicted_rate = next_funding.get("predicted_rate", 0.0)
            
            # Calcola l'impatto del funding (+ se guadagno, - se perdita)
            funding_impact = 0.0
            
            if direction.lower() == 'long':
                # I long pagano quando il tasso è positivo, ricevono quando è negativo
                funding_impact = -predicted_rate * position_value / 100
            else:  # short
                # Gli short pagano quando il tasso è negativo, ricevono quando è positivo
                funding_impact = predicted_rate * position_value / 100
            
            # Calcola l'impatto totale su più periodi
            total_impact = funding_impact * holding_periods
            
            # Calcola il rendimento in percentuale
            impact_percent = (total_impact / position_value) * 100
            
            return {
                "symbol": symbol,
                "position_value": position_value,
                "direction": direction,
                "funding_rate": predicted_rate,
                "funding_impact_per_period": funding_impact,
                "holding_periods": holding_periods,
                "total_impact": total_impact,
                "impact_percent": impact_percent,
                "next_funding": next_funding.get("next_funding_time"),
                "countdown": next_funding.get("countdown")
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo dell'impatto del funding per {symbol}: {str(e)}")
            
            return {
                "symbol": symbol,
                "position_value": position_value,
                "direction": direction,
                "funding_rate": 0.0,
                "funding_impact_per_period": 0.0,
                "holding_periods": holding_periods,
                "total_impact": 0.0,
                "impact_percent": 0.0,
                "error": str(e)
            }
    
    def should_adjust_position_for_funding(self, symbol: str, position_value: float,
                                         direction: str, threshold_percent: float = 0.05) -> Tuple[bool, str]:
        """
        Determina se una posizione dovrebbe essere aggiustata prima del prossimo funding
        
        Args:
            symbol: Simbolo della coppia
            position_value: Valore della posizione in USD
            direction: Direzione della posizione ('long' o 'short')
            threshold_percent: Soglia percentuale per l'impatto del funding
            
        Returns:
            Tupla (aggiustare, motivo)
        """
        try:
            # Calcola l'impatto del funding
            impact = self.calculate_funding_impact(symbol, position_value, direction, 1)
            
            # Controlla se l'impatto è significativo
            impact_percent = impact.get("impact_percent", 0.0)
            
            if abs(impact_percent) >= threshold_percent:
                if impact_percent < 0:
                    # L'impatto è negativo (perdita)
                    return True, f"Alto impatto negativo del funding: {impact_percent:.3f}% ({impact.get('total_impact', 0.0):.2f} USD)"
                else:
                    # L'impatto è positivo (guadagno)
                    return False, f"Impatto positivo del funding: +{impact_percent:.3f}% (+{impact.get('total_impact', 0.0):.2f} USD)"
            else:
                # L'impatto è trascurabile
                return False, f"Impatto del funding trascurabile: {impact_percent:.3f}% ({impact.get('total_impact', 0.0):.2f} USD)"
                
        except Exception as e:
            self.logger.error(f"Errore nella valutazione dell'aggiustamento per {symbol}: {str(e)}")
            
            # In caso di errore, meglio essere conservativi
            return False, f"Errore nella valutazione dell'impatto del funding: {str(e)}"
    
    def get_funding_arbitrage_pairs(self, min_rate_diff: float = 0.03) -> List[Dict[str, Any]]:
        """
        Trova coppie di simboli correlati con differenze significative nei tassi di funding
        che possono essere usate per arbitraggio (long su uno, short sull'altro)
        
        Args:
            min_rate_diff: Differenza minima di funding rate per considerare un'opportunità (%)
            
        Returns:
            Lista di opportunità di arbitraggio
        """
        opportunities = []
        
        # Coppie di simboli correlati (esempi)
        correlated_pairs = [
            ("BTC/USDT", "ETH/USDT"),  # BTC e ETH sono spesso correlati
            ("ETH/USDT", "SOL/USDT"),  # ETH e SOL sono spesso correlati
            ("BNB/USDT", "ETH/USDT"),  # BNB e ETH possono essere correlati
            # Aggiungi altre coppie correlate se necessario
        ]
        
        for symbol1, symbol2 in correlated_pairs:
            try:
                # Ottieni i prossimi tassi di funding
                next_funding1 = self.get_next_funding(symbol1)
                next_funding2 = self.get_next_funding(symbol2)
                
                rate1 = next_funding1.get("predicted_rate", 0.0)
                rate2 = next_funding2.get("predicted_rate", 0.0)
                
                # Calcola la differenza di tasso
                rate_diff = rate1 - rate2
                
                # Verifica se la differenza è significativa
                if abs(rate_diff) >= min_rate_diff:
                    # Determina la strategia di arbitraggio
                    if rate_diff > 0:
                        # symbol1 ha un tasso più alto: short su symbol1, long su symbol2
                        long_symbol = symbol2
                        short_symbol = symbol1
                        long_rate = rate2
                        short_rate = rate1
                    else:
                        # symbol2 ha un tasso più alto: short su symbol2, long su symbol1
                        long_symbol = symbol1
                        short_symbol = symbol2
                        long_rate = rate1
                        short_rate = rate2
                    
                    # Calcola il rendimento netto dell'arbitraggio
                    # Il long riceve quando il tasso è negativo, lo short riceve quando il tasso è positivo
                    net_rate = (short_rate if short_rate > 0 else 0) + (-long_rate if long_rate < 0 else 0)
                    
                    # Calcola il rendimento annualizzato (3 pagamenti al giorno)
                    annual_return = net_rate * 3 * 365 / 100  # Converti in decimale
                    
                    # Aggiungi l'opportunità
                    opportunities.append({
                        "long_symbol": long_symbol,
                        "short_symbol": short_symbol,
                        "long_rate": long_rate,
                        "short_rate": short_rate,
                        "rate_diff": abs(rate_diff),
                        "net_rate": net_rate,
                        "annual_return": annual_return,
                        "next_funding_time": min(
                            next_funding1.get("next_funding_time", datetime.max),
                            next_funding2.get("next_funding_time", datetime.max)
                        ),
                        "updated_at": datetime.now()
                    })
                    
            except Exception as e:
                self.logger.error(f"Errore nell'analisi dell'arbitraggio per {symbol1}-{symbol2}: {str(e)}")
        
        # Ordina per rendimento annualizzato decrescente
        opportunities.sort(key=lambda x: x.get("annual_return", 0.0), reverse=True)
        
        return opportunities
    
    def get_funding_historical_analysis(self, symbol: str, periods: int = 30) -> Dict[str, Any]:
        """
        Analizza i trend storici dei tassi di funding
        
        Args:
            symbol: Simbolo della coppia
            periods: Numero di periodi di funding da analizzare
            
        Returns:
            Analisi storica dei tassi di funding
        """
        try:
            # Ottieni la cronologia di funding
            df = self.get_funding_history(symbol, limit=periods)
            
            if df.empty:
                return {"symbol": symbol, "status": "no_data", "message": "Nessun dato di funding disponibile"}
            
            # Calcola statistiche descrittive
            mean_rate = df["funding_rate"].mean()
            median_rate = df["funding_rate"].median()
            std_rate = df["funding_rate"].std()
            max_rate = df["funding_rate"].max()
            min_rate = df["funding_rate"].min()
            
            # Calcola la percentuale di tassi positivi/negativi
            positive_pct = (df["funding_rate"] > 0).mean() * 100
            negative_pct = (df["funding_rate"] < 0).mean() * 100
            
            # Calcola la tendenza (trend)
            if len(df) >= 5:
                # Usa gli ultimi 5 periodi per determinare il trend
                recent_trend = df["funding_rate"].head(5).mean()
                overall_mean = df["funding_rate"].mean()
                
                if recent_trend > overall_mean * 1.2:
                    trend = "increasing"  # Trend in aumento
                elif recent_trend < overall_mean * 0.8:
                    trend = "decreasing"  # Trend in diminuzione
                else:
                    trend = "stable"  # Trend stabile
            else:
                trend = "unknown"  # Dati insufficienti
            
            # Estrai i tassi di funding come lista
            rates = df["funding_rate"].tolist()
            timestamps = [ts.isoformat() for ts in df["timestamp"].tolist()]
            
            return {
                "symbol": symbol,
                "status": "success",
                "mean_rate": mean_rate,
                "median_rate": median_rate,
                "std_rate": std_rate,
                "max_rate": max_rate,
                "min_rate": min_rate,
                "positive_pct": positive_pct,
                "negative_pct": negative_pct,
                "trend": trend,
                "rates": rates,
                "timestamps": timestamps,
                "periods": len(df),
                "updated_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi storica del funding per {symbol}: {str(e)}")
            
            return {
                "symbol": symbol,
                "status": "error",
                "message": f"Errore nell'analisi: {str(e)}",
                "updated_at": datetime.now()
            }