"""
Modulo per l'analisi del sentiment del mercato
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import requests
import re
import json
from collections import defaultdict

from utils.logger import get_logger
from config.settings import (
    DEFAULT_TRADING_PAIRS, SENTIMENT_UPDATE_INTERVAL,
    SENTIMENT_WEIGHT
)
from config.credentials import (
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
)

logger = get_logger(__name__)

class SentimentAnalyzer:
    """Classe per l'analisi del sentiment del mercato"""
    
    def __init__(self):
        """Inizializza l'analizzatore di sentiment"""
        self.logger = get_logger(__name__)
        
        # Cache dei sentiment
        self.sentiment_cache = {}
        self.last_update = {}
        
        # Lista dei simboli da monitorare
        self.symbols = DEFAULT_TRADING_PAIRS
        
        # Thread per l'aggiornamento asincrono
        self.update_thread = None
        self.running = False
        
        # Social media client
        self.twitter_client = None
        
        # Dizionario per le keyword associate ai simboli
        self.symbol_keywords = self._init_symbol_keywords()
        
        self.logger.info("SentimentAnalyzer inizializzato")
    
    def _init_symbol_keywords(self) -> Dict[str, List[str]]:
        """
        Inizializza le keyword associate ai simboli
        
        Returns:
            Dizionario {symbol: [keywords]}
        """
        keywords = {}
        
        # Definisci le keyword per ogni simbolo
        # Formato: {"BTC/USDT": ["bitcoin", "btc", "xbt", ...]}
        keywords["BTC/USDT"] = ["bitcoin", "btc", "xbt", "#bitcoin", "#btc"]
        keywords["ETH/USDT"] = ["ethereum", "eth", "#ethereum", "#eth"]
        keywords["SOL/USDT"] = ["solana", "sol", "#solana", "#sol"]
        keywords["BNB/USDT"] = ["binance", "bnb", "#binance", "#bnb", "binance coin"]
        keywords["XRP/USDT"] = ["ripple", "xrp", "#ripple", "#xrp"]
        keywords["ADA/USDT"] = ["cardano", "ada", "#cardano", "#ada"]
        
        return keywords
    
    def start_background_updates(self) -> None:
        """
        Avvia il thread di aggiornamento in background
        """
        if self.update_thread is not None and self.update_thread.is_alive():
            self.logger.warning("Thread di sentiment già in esecuzione")
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._background_update_worker)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        self.logger.info("Thread di sentiment avviato")
    
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
                
        self.logger.info("Thread di sentiment fermato")
    
    def _background_update_worker(self) -> None:
        """Thread worker per aggiornamenti in background"""
        self.logger.info("Worker di sentiment avviato")
        
        while self.running:
            try:
                # Aggiorna il sentiment per tutti i simboli
                for symbol in self.symbols:
                    now = datetime.now()
                    
                    # Determina se è necessario un aggiornamento
                    needs_update = (
                        symbol not in self.last_update or
                        now - self.last_update[symbol] > timedelta(seconds=SENTIMENT_UPDATE_INTERVAL)
                    )
                    
                    if needs_update:
                        self.logger.debug(f"Aggiornamento sentiment per {symbol}")
                        
                        # Aggiorna il sentiment
                        sentiment = self.analyze_sentiment(symbol)
                        
                        # Aggiorna la cache
                        self.sentiment_cache[symbol] = sentiment
                        self.last_update[symbol] = now
                        
                        self.logger.debug(f"Sentiment aggiornato per {symbol}: {sentiment}")
                        
                        # Sleep tra i simboli per non sovraccaricare le API
                        time.sleep(2.0)
                
                # Sleep più lungo tra i cicli completi
                time.sleep(60.0)
                
            except Exception as e:
                self.logger.error(f"Errore nel thread di sentiment: {str(e)}")
                time.sleep(300.0)  # Sleep più lungo in caso di errore
    
    def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analizza il sentiment per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Dizionario con i dati di sentiment
        """
        try:
            # Se il sentiment è in cache e aggiornato, usalo
            now = datetime.now()
            if (symbol in self.sentiment_cache and
                symbol in self.last_update and
                now - self.last_update[symbol] < timedelta(seconds=SENTIMENT_UPDATE_INTERVAL)):
                return self.sentiment_cache[symbol]
            
            # Altrimenti, calcola il nuovo sentiment
            
            # 1. Ottieni i dati dai social media
            tweets = self._get_tweets(symbol)
            
            # 2. Ottieni i dati dalle notizie
            news = self._get_news(symbol)
            
            # 3. Ottieni i dati dal mercato (volumi, volatilità)
            market_data = self._get_market_data(symbol)
            
            # 4. Analizza il sentiment dai dati
            sentiment_data = self._analyze_data(tweets, news, market_data)
            
            # 5. Calcola un punteggio complessivo di sentiment
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            
            # Crea il risultato
            result = {
                "symbol": symbol,
                "score": sentiment_score,
                "sentiment": "bullish" if sentiment_score > 0.1 else ("bearish" if sentiment_score < -0.1 else "neutral"),
                "data": sentiment_data,
                "timestamp": now.isoformat()
            }
            
            # Aggiorna la cache
            self.sentiment_cache[symbol] = result
            self.last_update[symbol] = now
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi del sentiment per {symbol}: {str(e)}")
            
            # In caso di errore, restituisci un sentiment neutro
            return {
                "symbol": symbol,
                "score": 0.0,
                "sentiment": "neutral",
                "data": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _get_tweets(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Ottiene i tweet relativi al simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Lista di tweet
        """
        # In un'implementazione reale, questa funzione userebbe l'API di Twitter
        # per ottenere i tweet relativi al simbolo
        # Qui simuliamo il risultato
        
        # Ottieni le keyword per il simbolo
        keywords = self.symbol_keywords.get(symbol, [])
        
        if not keywords:
            return []
        
        # Simula alcuni tweet
        tweets = []
        
        # Qui simuliamo una risposta da Twitter
        simulated_tweets = [
            {
                "text": f"I'm really bullish on {keywords[0]} right now! The fundamentals look great.",
                "sentiment": 0.8,
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "text": f"{keywords[0]} could see a massive price increase soon! #bullmarket",
                "sentiment": 0.9,
                "created_at": (datetime.now() - timedelta(hours=5)).isoformat()
            },
            {
                "text": f"Not sure about {keywords[0]}, the market seems uncertain right now.",
                "sentiment": 0.0,
                "created_at": (datetime.now() - timedelta(hours=8)).isoformat()
            },
            {
                "text": f"{keywords[0]} is showing bearish signals on the chart. Be careful.",
                "sentiment": -0.6,
                "created_at": (datetime.now() - timedelta(hours=12)).isoformat()
            }
        ]
        
        # Aggiungi solo se ci sono match di keyword
        for tweet in simulated_tweets:
            tweets.append(tweet)
        
        return tweets
    
    def _get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Ottiene le notizie relative al simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Lista di notizie
        """
        # In un'implementazione reale, questa funzione userebbe un'API di news
        # per ottenere le notizie relative al simbolo
        # Qui simuliamo il risultato
        
        # Ottieni le keyword per il simbolo
        keywords = self.symbol_keywords.get(symbol, [])
        
        if not keywords:
            return []
        
        # Simula alcune notizie
        news = []
        
        # Qui simuliamo una risposta da un'API di news
        simulated_news = [
            {
                "title": f"Major partnership announced for {keywords[0]} project",
                "summary": f"The {keywords[0]} project has announced a major partnership with a Fortune 500 company.",
                "sentiment": 0.7,
                "published_at": (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                "title": f"Market analysis shows growing adoption of {keywords[0]}",
                "summary": f"Recent market analysis indicates growing adoption of {keywords[0]} technology.",
                "sentiment": 0.5,
                "published_at": (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                "title": f"Regulatory concerns for {keywords[0]} in some countries",
                "summary": f"Some countries have expressed regulatory concerns about {keywords[0]}.",
                "sentiment": -0.3,
                "published_at": (datetime.now() - timedelta(days=3)).isoformat()
            }
        ]
        
        # Aggiungi solo se ci sono match di keyword
        for article in simulated_news:
            news.append(article)
        
        return news
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Ottiene i dati di mercato per il simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Dati di mercato
        """
        # In un'implementazione reale, questa funzione otterrebbe dati
        # come volume, volatilità, ecc. dall'exchange
        # Qui simuliamo il risultato
        
        return {
            "volume_change_24h": 15.3,  # Percentuale
            "price_change_24h": 2.7,    # Percentuale
            "volatility": 4.2,          # Percentuale
            "long_short_ratio": 1.2     # Rapporto tra posizioni long e short
        }
    
    def _analyze_data(self, tweets: List[Dict[str, Any]], 
                     news: List[Dict[str, Any]],
                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza i dati raccolti
        
        Args:
            tweets: Lista di tweet
            news: Lista di notizie
            market_data: Dati di mercato
            
        Returns:
            Dati di sentiment analizzati
        """
        # 1. Calcola il sentiment medio dei tweet
        tweet_sentiment = 0.0
        if tweets:
            tweet_sentiment = sum(t.get("sentiment", 0.0) for t in tweets) / len(tweets)
        
        # 2. Calcola il sentiment medio delle notizie
        news_sentiment = 0.0
        if news:
            news_sentiment = sum(n.get("sentiment", 0.0) for n in news) / len(news)
        
        # 3. Calcola un punteggio dai dati di mercato
        market_sentiment = 0.0
        if market_data:
            volume_factor = market_data.get("volume_change_24h", 0.0) / 100.0
            price_factor = market_data.get("price_change_24h", 0.0) / 100.0
            long_short_factor = (market_data.get("long_short_ratio", 1.0) - 1.0) * 0.5
            
            market_sentiment = (volume_factor + price_factor + long_short_factor) / 3.0
        
        # 4. Conta il numero di fonti
        tweet_count = len(tweets)
        news_count = len(news)
        
        # 5. Estrai alcune frasi rappresentative
        sample_tweets = [t.get("text", "") for t in tweets[:3]]
        sample_news = [n.get("title", "") for n in news[:3]]
        
        return {
            "twitter": {
                "sentiment": tweet_sentiment,
                "count": tweet_count,
                "samples": sample_tweets
            },
            "news": {
                "sentiment": news_sentiment,
                "count": news_count,
                "samples": sample_news
            },
            "market": {
                "sentiment": market_sentiment,
                "data": market_data
            }
        }
    
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """
        Calcola un punteggio complessivo di sentiment
        
        Args:
            sentiment_data: Dati di sentiment analizzati
            
        Returns:
            Punteggio di sentiment (-1.0 a 1.0)
        """
        # Pesi per i diversi tipi di sentiment
        twitter_weight = 0.3
        news_weight = 0.4
        market_weight = 0.3
        
        # Sentiment per tipo
        twitter_sentiment = sentiment_data.get("twitter", {}).get("sentiment", 0.0)
        news_sentiment = sentiment_data.get("news", {}).get("sentiment", 0.0)
        market_sentiment = sentiment_data.get("market", {}).get("sentiment", 0.0)
        
        # Considera anche il numero di fonti
        twitter_count = sentiment_data.get("twitter", {}).get("count", 0)
        news_count = sentiment_data.get("news", {}).get("count", 0)
        
        twitter_factor = min(twitter_count / 10.0, 1.0)
        news_factor = min(news_count / 5.0, 1.0)
        
        # Calcola il punteggio ponderato
        weighted_score = (
            twitter_sentiment * twitter_weight * twitter_factor +
            news_sentiment * news_weight * news_factor +
            market_sentiment * market_weight
        )
        
        # Normalizza il punteggio tra -1.0 e 1.0
        normalized_score = max(min(weighted_score, 1.0), -1.0)
        
        return normalized_score
    
    def get_sentiment(self, symbol: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Ottiene il sentiment per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            force_update: Se forzare un aggiornamento
            
        Returns:
            Dizionario con i dati di sentiment
        """
        if force_update or symbol not in self.sentiment_cache:
            return self.analyze_sentiment(symbol)
        
        return self.sentiment_cache[symbol]
    
    def get_combined_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Ottiene il sentiment combinato per più simboli
        
        Args:
            symbols: Lista di simboli
            
        Returns:
            Dizionario con il sentiment combinato
        """
        combined_score = 0.0
        sentiments = {}
        
        for symbol in symbols:
            sentiment = self.get_sentiment(symbol)
            sentiments[symbol] = sentiment
            combined_score += sentiment.get("score", 0.0)
        
        # Media dei punteggi
        if symbols:
            combined_score /= len(symbols)
        
        sentiment_type = "bullish" if combined_score > 0.1 else ("bearish" if combined_score < -0.1 else "neutral")
        
        return {
            "combined_score": combined_score,
            "sentiment": sentiment_type,
            "sentiments": sentiments,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Ottiene il sentiment complessivo del mercato
        
        Returns:
            Dizionario con il sentiment del mercato
        """
        return self.get_combined_sentiment(self.symbols)
