#!/usr/bin/env python3
"""
Script per testare la connessione con l'API di Bybit
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Aggiungi la directory corrente al path per l'importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.bybit_api import BybitAPI
from config.settings import BotMode
from utils.logger import get_logger, setup_log_capture

# Configura il logging
logger = get_logger(__name__)
setup_log_capture()

def test_bybit_api():
    """Testa le funzionalità principali dell'API di Bybit"""
    print("Inizializzazione API Bybit (testnet)...")
    api = BybitAPI(mode=BotMode.DEMO)
    
    # Test 1: Ottieni i dati di mercato
    print("\nTest 1: Ottieni dati di mercato (BTCUSDT, 15m)")
    try:
        klines = api.get_klines("BTC/USDT", "15m", limit=10)
        if not klines.empty:
            print(f"✅ Ottenute {len(klines)} candele")
            print(klines.head(2))
        else:
            print("❌ Nessun dato ricevuto")
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
    
    # Test 2: Ottieni ticker
    print("\nTest 2: Ottieni ticker (BTCUSDT)")
    try:
        ticker = api.get_ticker("BTC/USDT")
        print(f"✅ Ticker ottenuto")
        print(f"Prezzo: {ticker.get('lastPrice', 'N/A')}")
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
    
    # Test 3: Ottieni order book
    print("\nTest 3: Ottieni order book (BTCUSDT)")
    try:
        orderbook = api.get_order_book("BTC/USDT", limit=5)
        print(f"✅ Order book ottenuto")
        print(f"Bid più alto: {orderbook['b'][0][0] if 'b' in orderbook and orderbook['b'] else 'N/A'}")
        print(f"Ask più basso: {orderbook['a'][0][0] if 'a' in orderbook and orderbook['a'] else 'N/A'}")
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
    
    # Test 4: Converti intervalli
    print("\nTest 4: Test conversione intervalli")
    intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    for interval in intervals:
        converted = api._convert_interval(interval)
        print(f"  {interval} -> {converted}")
    
    # Test del wallet balance (richiede API key corretta)
    print("\nTest 5: Ottieni bilancio wallet (richiede API key valida)")
    try:
        balance = api.get_wallet_balance()
        print(f"✅ Bilancio ottenuto")
        print(balance)
    except Exception as e:
        print(f"❌ Errore (potrebbe essere normale se non hai configurato API key valide): {str(e)}")
    
    print("\nTest completati!")

if __name__ == "__main__":
    test_bybit_api()
