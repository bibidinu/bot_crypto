"""
Modulo per il calendario economico e le news rilevanti
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import threading
import requests
import json

from utils.logger import get_logger
from config.settings import ECONOMIC_CALENDAR_UPDATE_INTERVAL
from config.credentials import ECONOMIC_CALENDAR_API_KEY

logger = get_logger(__name__)

class EconomicCalendar:
    """Classe per il calendario economico e le news rilevanti"""
    
    def __init__(self):
        """Inizializza il calendario economico"""
        self.logger = get_logger(__name__)
        
        # Cache dei dati
        self.events_cache = []
        self.last_update = None
        
        # Thread per l'aggiornamento asincrono
        self.update_thread = None
        self.running = False
        
        # Elenchi di parole chiave per filtrare eventi rilevanti per le crypto
        self.crypto_keywords = self._init_crypto_keywords()
        
        self.logger.info("EconomicCalendar inizializzato")
    
    def _init_crypto_keywords(self) -> List[str]:
        """
        Inizializza le parole chiave per eventi rilevanti per le crypto
        
        Returns:
            Lista di parole chiave
        """
        return [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "blockchain", "digital currency", "digital asset", "defi",
            "nft", "token", "binance", "coinbase", "exchange", "wallet",
            "mining", "miner", "cbdc", "central bank digital currency",
            "regulation", "sec", "cftc", "fed", "interest rate", "inflation",
            "stablecoin", "usdt", "usdc", "ripple", "xrp", "cardano",
            "solana", "altcoin"
        ]
    
    def start_background_updates(self) -> None:
        """
        Avvia il thread di aggiornamento in background
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            self.logger.warning("Thread del calendario già in esecuzione")
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._background_update_worker)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Thread del calendario avviato")
    
    def stop_background_updates(self) -> None:
        """
        Ferma il thread di aggiornamento in background
        """
        self.running = False
        
        if self.update_thread is not None:
            try:
                self.update_thread.join(timeout=5.0)
            except:
                pass
                
        self.logger.info("Thread del calendario fermato")
    
    def _background_update_worker(self) -> None:
        """Thread worker per aggiornamenti in background"""
        self.logger.info("Worker del calendario avviato")
        
        while self.running:
            try:
                now = datetime.now()
                
                # Determina se è necessario un aggiornamento
                needs_update = (
                    self.last_update is None or
                    now - self.last_update > timedelta(seconds=ECONOMIC_CALENDAR_UPDATE_INTERVAL)
                )
                
                if needs_update:
                    self.logger.debug("Aggiornamento calendario economico")
                    
                    # Aggiorna i dati
                    self.update_calendar()
                    
                    self.logger.debug(f"Calendario aggiornato: {len(self.events_cache)} eventi")
                
                # Sleep 
                time.sleep(3600.0)  # Controlla ogni ora
                
            except Exception as e:
                self.logger.error(f"Errore nel thread del calendario: {str(e)}")
                time.sleep(3600.0)  # Sleep più lungo in caso di errore
    
    def update_calendar(self) -> None:
        """
        Aggiorna i dati del calendario economico
        """
        try:
            # Date per la richiesta
            now = datetime.now()
            start_date = now.strftime("%Y-%m-%d")
            end_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")
            
            # In un'implementazione reale, questa funzione chiamerebbe un'API
            # esterna per ottenere i dati del calendario economico
            # Qui simuliamo la risposta
            
            events = self._get_simulated_events(start_date, end_date)
            
            # Filtra eventi rilevanti per le crypto
            filtered_events = self._filter_crypto_events(events)
            
            # Aggiorna la cache
            self.events_cache = filtered_events
            self.last_update = now
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento del calendario: {str(e)}")
    
    def _get_simulated_events(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Simula il recupero di eventi dal calendario economico
        
        Args:
            start_date: Data di inizio
            end_date: Data di fine
            
        Returns:
            Lista di eventi
        """
        # Simuliamo alcuni eventi economici
        now = datetime.now()
        
        events = [
            {
                "id": 1,
                "title": "Federal Reserve Interest Rate Decision",
                "country": "United States",
                "date": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
                "time": "18:00:00",
                "impact": "high",
                "forecast": "5.00%",
                "previous": "5.25%",
                "description": "The Federal Reserve is expected to cut interest rates by 25 basis points."
            },
            {
                "id": 2,
                "title": "SEC Decision on Bitcoin ETF Application",
                "country": "United States",
                "date": (now + timedelta(days=3)).strftime("%Y-%m-%d"),
                "time": "20:00:00",
                "impact": "high",
                "forecast": "Approval",
                "previous": "N/A",
                "description": "The SEC is expected to make a decision on the latest Bitcoin ETF application."
            },
            {
                "id": 3,
                "title": "European Central Bank Policy Statement",
                "country": "Eurozone",
                "date": (now + timedelta(days=4)).strftime("%Y-%m-%d"),
                "time": "12:45:00",
                "impact": "medium",
                "forecast": "3.75%",
                "previous": "3.75%",
                "description": "The ECB is expected to maintain current interest rates."
            },
            {
                "id": 4,
                "title": "US Inflation Data (CPI)",
                "country": "United States",
                "date": (now + timedelta(days=5)).strftime("%Y-%m-%d"),
                "time": "12:30:00",
                "impact": "high",
                "forecast": "3.1%",
                "previous": "3.2%",
                "description": "Consumer Price Index data for the previous month."
            },
            {
                "id": 5,
                "title": "Japan Digital Currency Announcement",
                "country": "Japan",
                "date": (now + timedelta(days=6)).strftime("%Y-%m-%d"),
                "time": "06:00:00",
                "impact": "medium",
                "forecast": "N/A",
                "previous": "N/A",
                "description": "Bank of Japan expected to announce progress on CBDC development."
            }
        ]
        
        return events
    
    def _filter_crypto_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra gli eventi rilevanti per le crypto
        
        Args:
            events: Lista di eventi
            
        Returns:
            Lista di eventi filtrati
        """
        filtered = []
        
        for event in events:
            # Ottieni il testo completo dell'evento
            event_text = (
                event.get("title", "") + " " +
                event.get("description", "")
            ).lower()
            
            # Verifica se contiene parole chiave crypto
            is_crypto_relevant = any(keyword.lower() in event_text for keyword in self.crypto_keywords)
            
            # Verifica se è ad alto impatto
            is_high_impact = event.get("impact") == "high"
            
            # Includi se è rilevante per crypto o ad alto impatto
            if is_crypto_relevant or is_high_impact:
                filtered.append(event)
        
        return filtered
    
    def get_upcoming_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Ottiene i prossimi eventi
        
        Args:
            days: Numero di giorni futuri
            
        Returns:
            Lista di eventi
        """
        # Se i dati non sono aggiornati, aggiorna
        now = datetime.now()
        if self.last_update is None or now - self.last_update > timedelta(days=1):
            self.update_calendar()
        
        # Filtra eventi nei prossimi 'days' giorni
        end_date = now + timedelta(days=days)
        
        events = []
        for event in self.events_cache:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            
            if now.date() <= event_date.date() <= end_date.date():
                events.append(event)
        
        # Ordina per data
        events.sort(key=lambda e: e["date"] + " " + e.get("time", "00:00:00"))
        
        return events
    
    def get_events_by_impact(self, impact: str = "high") -> List[Dict[str, Any]]:
        """
        Ottiene eventi filtrati per livello di impatto
        
        Args:
            impact: Livello di impatto (high, medium, low)
            
        Returns:
            Lista di eventi
        """
        return [e for e in self.events_cache if e.get("impact") == impact]
    
    def get_crypto_specific_events(self) -> List[Dict[str, Any]]:
        """
        Ottiene eventi specifici per le crypto
        
        Returns:
            Lista di eventi
        """
        crypto_events = []
        
        for event in self.events_cache:
            # Ottieni il testo completo dell'evento
            event_text = (
                event.get("title", "") + " " +
                event.get("description", "")
            ).lower()
            
            # Verifica se contiene parole chiave crypto
            is_crypto_specific = any(keyword.lower() in event_text for keyword in self.crypto_keywords)
            
            if is_crypto_specific:
                crypto_events.append(event)
        
        return crypto_events
    
    def get_next_high_impact_event(self) -> Optional[Dict[str, Any]]:
        """
        Ottiene il prossimo evento ad alto impatto
        
        Returns:
            Prossimo evento o None
        """
        high_impact = self.get_events_by_impact("high")
        
        if not high_impact:
            return None
        
        # Ordina per data
        high_impact.sort(key=lambda e: e["date"] + " " + e.get("time", "00:00:00"))
        
        # Trova il primo evento futuro
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        for event in high_impact:
            event_datetime = event["date"] + " " + event.get("time", "00:00:00")
            
            if event_datetime > now_str:
                return event
        
        return None
    
    def get_news_alerts(self) -> List[Dict[str, Any]]:
        """
        Ottiene le ultime notizie rilevanti
        
        Returns:
            Lista di notizie
        """
        # In un'implementazione reale, questa funzione chiamerebbe un'API
        # di notizie per ottenere le ultime news
        # Qui simuliamo la risposta
        
        now = datetime.now()
        
        news = [
            {
                "id": 1,
                "title": "Major Exchange Announces New Listing",
                "source": "CryptoNews",
                "date": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "category": "exchanges",
                "impact": "medium",
                "description": "A major cryptocurrency exchange has announced the listing of a new token."
            },
            {
                "id": 2,
                "title": "Bitcoin Hashrate Reaches All-Time High",
                "source": "BlockchainInsider",
                "date": (now - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "category": "mining",
                "impact": "low",
                "description": "Bitcoin's network hashrate has reached a new all-time high, indicating strong network security."
            },
            {
                "id": 3,
                "title": "Regulatory Framework Proposed in Major Market",
                "source": "FinancialTimes",
                "date": (now - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                "category": "regulation",
                "impact": "high",
                "description": "A major economy has proposed a new regulatory framework for cryptocurrencies."
            }
        ]
        
        return news
    
    def get_market_analysis(self) -> Dict[str, Any]:
        """
        Ottiene un'analisi di mercato basata su eventi e notizie
        
        Returns:
            Dizionario con l'analisi
        """
        try:
            # Ottieni eventi e notizie
            upcoming_events = self.get_upcoming_events(3)  # Prossimi 3 giorni
            high_impact_events = self.get_events_by_impact("high")
            crypto_events = self.get_crypto_specific_events()
            news = self.get_news_alerts()
            
            # Calcola un punteggio di sentiment
            sentiment_score = 0.0
            
            # Eventi ad alto impatto pesano di più
            for event in high_impact_events:
                if "rate" in event.get("title", "").lower() and "cut" in event.get("description", "").lower():
                    sentiment_score += 0.2  # Rate cuts are positive
                elif "rate" in event.get("title", "").lower() and "hike" in event.get("description", "").lower():
                    sentiment_score -= 0.2  # Rate hikes are negative
                elif "inflation" in event.get("title", "").lower():
                    forecast = event.get("forecast", "0%").replace("%", "")
                    previous = event.get("previous", "0%").replace("%", "")
                    
                    try:
                        if float(forecast) < float(previous):
                            sentiment_score += 0.1  # Lower inflation is positive
                        else:
                            sentiment_score -= 0.1  # Higher inflation is negative
                    except:
                        pass
            
            # Eventi specifici crypto
            for event in crypto_events:
                title = event.get("title", "").lower()
                desc = event.get("description", "").lower()
                
                if "etf" in title or "etf" in desc:
                    if "approval" in desc or "approved" in desc:
                        sentiment_score += 0.3  # ETF approvals are very positive
                elif "regulation" in title or "regulation" in desc:
                    if "positive" in desc or "framework" in desc:
                        sentiment_score += 0.1  # Positive regulation is good
                    elif "ban" in desc or "restrict" in desc:
                        sentiment_score -= 0.3  # Bans are very negative
            
            # Notizie
            for item in news:
                impact = item.get("impact", "low")
                title = item.get("title", "").lower()
                desc = item.get("description", "").lower()
                
                impact_factor = 0.1 if impact == "low" else 0.2 if impact == "medium" else 0.3
                
                # Parole positive
                positives = ["bullish", "surge", "gain", "rally", "adoption", "partnership", "investment"]
                if any(word in title or word in desc for word in positives):
                    sentiment_score += impact_factor
                
                # Parole negative
                negatives = ["bearish", "crash", "fall", "drop", "ban", "hack", "security breach", "vulnerability"]
                if any(word in title or word in desc for word in negatives):
                    sentiment_score -= impact_factor
            
            # Limita il punteggio tra -1.0 e 1.0
            sentiment_score = max(min(sentiment_score, 1.0), -1.0)
            
            # Determina il sentiment complessivo
            if sentiment_score > 0.2:
                sentiment = "bullish"
                recommendation = "Consider long positions on dips"
            elif sentiment_score < -0.2:
                sentiment = "bearish"
                recommendation = "Reduce exposure, consider short positions on rallies"
            else:
                sentiment = "neutral"
                recommendation = "Maintain balanced positions, focus on risk management"
            
            # Componi l'analisi
            analysis = {
                "sentiment_score": sentiment_score,
                "sentiment": sentiment,
                "recommendation": recommendation,
                "high_impact_count": len(high_impact_events),
                "crypto_specific_count": len(crypto_events),
                "news_count": len(news),
                "key_events": [e.get("title") for e in high_impact_events[:3]],
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi di mercato: {str(e)}")
            
            return {
                "sentiment_score": 0.0,
                "sentiment": "neutral",
                "recommendation": "Unable to generate analysis due to error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
