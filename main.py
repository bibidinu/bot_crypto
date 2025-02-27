#!/usr/bin/env python3
"""
Entry point principale del trading bot
"""
import os
import sys
import time
import argparse
import signal
import logging
from datetime import datetime, timedelta
import threading
import json
import numpy as np
import pandas as pd

from core.bot import TradingBot
from core.agent_manager import AgentManager
from config.settings import BOT_NAME, BOT_VERSION, BOT_MODE, BotMode, DEFAULT_TRADING_PAIRS
from utils.logger import get_logger, setup_log_capture
from data.market_data import MarketData
from api.bybit_api import BybitAPI
from strategy.ml_strategy import MLStrategy
from strategy.technical_strategy import TechnicalStrategy
from models.ml_models import prepare_training_data, generate_directional_target

# Configurazione logging
logger = get_logger(__name__)

def parse_arguments():
    """
    Analizza gli argomenti da linea di comando
    
    Returns:
        Argomenti analizzati
    """
    parser = argparse.ArgumentParser(description=f"{BOT_NAME} - Trading Bot v{BOT_VERSION}")
    
    parser.add_argument('-m', '--mode', type=str, choices=['demo', 'live'], default=BOT_MODE.value,
                        help='Modalità di esecuzione (demo/live)')
    
    parser.add_argument('-s', '--start', action='store_true',
                        help='Avvia il bot automaticamente')
    
    parser.add_argument('-c', '--console', action='store_true',
                        help='Avvia in modalità console (senza GUI)')
    
    parser.add_argument('-p', '--pairs', type=str, nargs='+',
                        help='Coppie di trading da monitorare')
    
    parser.add_argument('-r', '--risk', type=float,
                        help='Percentuale di rischio per trade')
    
    parser.add_argument('-i', '--init-models', action='store_true',
                        help='Inizializza i modelli ML prima di avviare il bot')
    
    parser.add_argument('-d', '--days', type=int, default=30,
                        help='Numero di giorni di dati per training (con --init-models)')
    
    parser.add_argument('-t', '--timeframes', type=str, nargs='+', default=['15m', '1h'],
                        help='Timeframes da utilizzare per il training (con --init-models)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Output verboso')
    
    return parser.parse_args()

def setup_signal_handlers(bot):
    """
    Configura i gestori dei segnali
    
    Args:
        bot: Istanza del bot
    """
    def signal_handler(sig, frame):
        logger.info(f"Segnale {sig} ricevuto, arresto in corso...")
        bot.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def init_ml_models(symbols, timeframes=['15m', '1h'], days=30):
    """
    Inizializza i modelli ML necessari
    
    Args:
        symbols: Lista di simboli
        timeframes: Lista di timeframes
        days: Numero di giorni di dati da utilizzare
    """
    logger.info(f"Inizializzazione modelli ML per {len(symbols)} simboli...")
    
    # Crea directory per i modelli se non esiste
    os.makedirs("models/saved", exist_ok=True)
    
    # Crea oggetti per recuperare i dati
    exchange = BybitAPI()
    market_data = MarketData(exchange)
    technical_strategy = TechnicalStrategy(exchange)
    ml_strategy = MLStrategy(exchange, "InitialMLStrategy", technical_strategy)
    
    for symbol in symbols:
        logger.info(f"Elaborazione per {symbol}")
        
        for timeframe in timeframes:
            try:
                logger.info(f"Recupero dati per {symbol} - {timeframe}")
                
                # Calcola il limite adeguato in base al timeframe
                limit = 0
                if timeframe == '15m':
                    limit = days * 24 * 4  # 4 candele da 15m per ora
                elif timeframe == '1h':
                    limit = days * 24      # 24 candele da 1h per giorno
                elif timeframe == '4h':
                    limit = days * 6       # 6 candele da 4h per giorno
                else:
                    limit = days * 24      # Valore predefinito
                
                # Recupera i dati storici
                df = market_data.get_market_data(symbol, timeframe, limit=limit, force_update=True)
                
                # Se non ci sono abbastanza dati, passa al prossimo timeframe
                if df.empty or len(df) < 100:
                    logger.warning(f"Dati insufficienti per {symbol} - {timeframe}: {len(df) if not df.empty else 0} candele")
                    continue
                    
                # Aggiungi indicatori
                df_with_indicators = technical_strategy.add_indicators(df)
                
                # Rimuovi valori infiniti o NaN che potrebbero causare errori
                df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan)
                df_with_indicators = df_with_indicators.dropna()
                
                # Verifica se ci sono ancora abbastanza dati dopo la pulizia
                if len(df_with_indicators) < 50:
                    logger.warning(f"Dati insufficienti dopo pulizia per {symbol} - {timeframe}: {len(df_with_indicators)} candele")
                    continue
                
                # Prepara i dati di addestramento con gestione errori
                logger.info(f"Preparazione dati di addestramento per {symbol} - {timeframe}")
                
                try:
                    train_data = prepare_training_data(
                        df_with_indicators, 
                        generate_directional_target,
                        future_periods=10,
                        binary=True
                    )
                    
                    # Verifica che il set di addestramento abbia abbastanza dati
                    if train_data.empty or len(train_data) < 30:
                        logger.warning(f"Dati di training insufficienti per {symbol} - {timeframe}: {len(train_data) if not train_data.empty else 0} campioni")
                        continue
                    
                    # Addestra il modello
                    logger.info(f"Addestramento modello per {symbol} - {timeframe}")
                    ml_strategy.train_model(symbol, train_data)
                    
                    logger.info(f"Modello per {symbol} - {timeframe} inizializzato")
                except Exception as e:
                    logger.error(f"Errore nella preparazione dati per {symbol} - {timeframe}: {str(e)}")
                    continue
                
            except Exception as e:
                logger.error(f"Errore nell'inizializzazione del modello per {symbol} - {timeframe}: {str(e)}")
    
    logger.info("Inizializzazione modelli ML completata")

def console_mode(bot, agent_manager):
    """
    Esegue il bot in modalità console
    
    Args:
        bot: Istanza del bot
        agent_manager: Istanza del gestore degli agenti
    """
    print(f"\n{'-'*50}")
    print(f" {BOT_NAME} v{BOT_VERSION} - Console Mode")
    print(f"{'-'*50}\n")
    
    print("Comandi disponibili:")
    print("  start      - Avvia il bot")
    print("  stop       - Ferma il bot")
    print("  status     - Mostra lo stato del bot")
    print("  positions  - Mostra le posizioni aperte")
    print("  balance    - Mostra il saldo del wallet")
    print("  scan       - Scansiona opportunità")
    print("  analyze    - Analizza un simbolo (es: analyze BTC/USDT)")
    print("  agents     - Mostra statistiche degli agenti")
    print("  funding    - Mostra i tassi di funding")
    print("  report     - Genera report di performance")
    print("  chart      - Genera un grafico (es: chart BTC/USDT)")
    print("  train      - Avvia addestramento modelli ML")
    print("  settings   - Mostra impostazioni attuali")
    print("  help       - Mostra questa guida")
    print("  exit       - Esci")
    print()
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            args = ' '.join(parts[1:])
            
            if cmd == "exit":
                bot.stop()
                agent_manager.stop()
                agent_manager.save_agents()
                print("Arrivederci!")
                break
                
            elif cmd == "help":
                print("Comandi disponibili:")
                print("  start      - Avvia il bot")
                print("  stop       - Ferma il bot")
                print("  status     - Mostra lo stato del bot")
                print("  positions  - Mostra le posizioni aperte")
                print("  balance    - Mostra il saldo del wallet")
                print("  scan       - Scansiona opportunità")
                print("  analyze    - Analizza un simbolo (es: analyze BTC/USDT)")
                print("  agents     - Mostra statistiche degli agenti")
                print("  funding    - Mostra i tassi di funding")
                print("  report     - Genera report di performance")
                print("  chart      - Genera un grafico (es: chart BTC/USDT)")
                print("  train      - Avvia addestramento modelli ML")
                print("  settings   - Mostra impostazioni attuali")
                print("  help       - Mostra questa guida")
                print("  exit       - Esci")
                
            elif cmd == "start":
                bot.start()
                print(f"Bot avviato in modalità {bot.mode.value}")
                
            elif cmd == "stop":
                bot.stop()
                print("Bot fermato")
                
            elif cmd == "status":
                status = bot.status.value
                uptime = bot.get_uptime()
                mode = bot.mode.value
                print(f"Stato: {status}")
                print(f"Modalità: {mode}")
                print(f"Uptime: {uptime}")
                print(f"Coppie: {len(bot.trading_pairs)}")
                
                try:
                    balance = bot.get_wallet_balance()
                    print(f"Saldo: {balance:.2f} USDT")
                except Exception as e:
                    print(f"Errore nel recupero del saldo: {str(e)}")
                
                # Mostra statistiche generali
                try:
                    overall_stats = bot.performance.get_overall_stats()
                    print("\nStatistiche generali:")
                    print(f"Trade totali: {overall_stats['total_trades']}")
                    print(f"Trade vincenti: {overall_stats['winning_trades']}")
                    print(f"Trade perdenti: {overall_stats['losing_trades']}")
                    print(f"Win rate: {overall_stats['win_rate']*100:.2f}%")
                    print(f"Profit factor: {overall_stats['profit_factor']:.2f}")
                    print(f"Profitto totale: {overall_stats['total_profit']:.2f} USDT")
                    
                    # Verifica se pronto per live
                    if bot.mode == BotMode.DEMO and overall_stats['win_rate'] >= 0.85:
                        print("\n🚀 Il bot ha raggiunto performance sufficienti per modalità LIVE!")
                except Exception as e:
                    print(f"Errore nel recupero delle statistiche: {str(e)}")
                
            elif cmd == "positions":
                try:
                    positions = bot.get_open_positions()
                    if positions:
                        print(f"Posizioni aperte ({len(positions)}):")
                        for i, pos in enumerate(positions):
                            print(f"  {i+1}. {pos['symbol']} ({pos['entry_type']}) - "
                                f"Size: {pos['size']:.5f}, "
                                f"Entry: {pos['entry_price']:.5f}, "
                                f"PnL: {pos.get('unrealized_pnl', 0):.2f} USDT")
                    else:
                        print("Nessuna posizione aperta")
                except Exception as e:
                    print(f"Errore nel recupero delle posizioni: {str(e)}")
                    
            elif cmd == "balance":
                try:
                    balance = bot.get_wallet_balance()
                    print(f"Saldo wallet: {balance:.2f} USDT")
                except Exception as e:
                    print(f"Errore nel recupero del saldo: {str(e)}")
                
            elif cmd == "scan":
                print("Scansione opportunità in corso...")
                try:
                    opportunities = bot.scan_for_opportunities()
                    if opportunities:
                        print(f"Opportunità trovate ({len(opportunities)}):")
                        for i, opp in enumerate(opportunities):
                            print(f"  {i+1}. {opp['symbol']} ({opp['signal']}) - "
                                f"Strength: {opp['signal_strength']:.2f}, "
                                f"Price: {opp['current_price']:.5f}")
                            if 'reason' in opp:
                                print(f"     Motivo: {opp['reason']}")
                    else:
                        print("Nessuna opportunità trovata")
                except Exception as e:
                    print(f"Errore nella scansione delle opportunità: {str(e)}")
                    
            elif cmd == "analyze":
                if not args:
                    print("Specificare un simbolo (es: analyze BTC/USDT)")
                    continue
                
                print(f"Analisi di {args} in corso...")
                try:
                    analysis = bot.analyze_symbol(args)
                    
                    if analysis:
                        print(f"Simbolo: {analysis['symbol']}")
                        print(f"Prezzo: {analysis['current_price']:.5f}")
                        print(f"Segnale: {analysis['signal']} (forza: {analysis['signal_strength']:.2f})")
                        
                        if 'sentiment' in analysis:
                            print(f"Sentiment: {analysis['sentiment']} (score: {analysis.get('sentiment_score', 0):.2f})")
                        
                        if 'entry_type' in analysis and analysis['entry_type']:
                            print(f"Tipo entrata: {analysis['entry_type']}")
                        
                        if 'stop_loss' in analysis and analysis['stop_loss']:
                            print(f"Stop loss: {analysis['stop_loss']:.5f}")
                        
                        if 'take_profits' in analysis and analysis['take_profits']:
                            for i, tp in enumerate(analysis['take_profits']):
                                print(f"TP{i+1}: {tp:.5f}")
                        
                        if 'reason' in analysis and analysis['reason']:
                            print(f"Motivo: {analysis['reason']}")
                            
                        # Stampa indicatori tecnici se disponibili
                        for indicator in ['rsi', 'macd', 'ema_20', 'ema_50', 'ema_200', 'atr']:
                            if indicator in analysis:
                                print(f"{indicator.upper()}: {analysis[indicator]:.4f}")
                                
                        # Stampa supporti e resistenze se disponibili
                        if 'next_support' in analysis:
                            print(f"Supporto: {analysis['next_support']:.5f}")
                        if 'next_resistance' in analysis:
                            print(f"Resistenza: {analysis['next_resistance']:.5f}")
                    else:
                        print(f"Impossibile analizzare {args}")
                except Exception as e:
                    print(f"Errore nell'analisi di {args}: {str(e)}")
                    
            elif cmd == "agents":
                try:
                    stats = agent_manager.get_agents_stats()
                    print("Statistiche degli agenti:")
                    print(f"Totale agenti: {stats['total_agents']}")
                    print("\nAgenti per simbolo:")
                    for symbol, count in stats.get('agents_by_symbol', {}).items():
                        print(f"  {symbol}: {count}")
                        
                    print("\nAgenti per tipo:")
                    for type_name, count in stats.get('agents_by_type', {}).items():
                        print(f"  {type_name}: {count}")
                        
                    print("\nMigliori agenti:")
                    for symbol, agent_info in stats.get('best_agents', {}).items():
                        print(f"  {symbol}: {agent_info['agent_id']} - Win rate: {agent_info['win_rate']:.2f}, "
                              f"Trades: {agent_info['total_trades']}, "
                              f"Profit: {agent_info['total_profit']:.2f} USDT")
                except Exception as e:
                    print(f"Errore nel recupero delle statistiche degli agenti: {str(e)}")
                    
            elif cmd == "funding":
                try:
                    from data.funding_analyzer import FundingAnalyzer
                    funding_analyzer = FundingAnalyzer(bot.exchange)
                    
                    if not args:
                        # Mostra opportunità di funding per tutte le coppie
                        print("Ricerca opportunità di funding...")
                        opps = funding_analyzer.get_best_funding_opportunities(bot.trading_pairs)
                        
                        if opps:
                            print(f"Migliori opportunità di funding ({len(opps)}):")
                            for i, opp in enumerate(opps):
                                print(f"  {i+1}. {opp['symbol']} - "
                                      f"Direction: {opp['direction']}, "
                                      f"Rate: {opp['predicted_rate']:.4f}%, "
                                      f"Annual Return: {opp['annual_return']*100:.2f}%")
                                print(f"     Next funding: {opp['countdown']}")
                                print(f"     Reason: {opp['reason']}")
                        else:
                            print("Nessuna opportunità di funding trovata")
                    else:
                        # Mostra analisi dettagliata per il simbolo specificato
                        symbol = args
                        print(f"Analisi funding per {symbol}...")
                        
                        # Info sul prossimo funding
                        next_funding = funding_analyzer.get_next_funding(symbol)
                        print(f"Prossimo funding: {next_funding['countdown']} - Rate: {next_funding['predicted_rate']:.4f}%")
                        
                        # Analisi storica
                        history = funding_analyzer.get_funding_historical_analysis(symbol)
                        if history['status'] == 'success':
                            print(f"Media funding rate: {history['mean_rate']:.4f}%")
                            print(f"Min/Max funding rate: {history['min_rate']:.4f}% / {history['max_rate']:.4f}%")
                            print(f"% Tassi positivi: {history['positive_pct']:.1f}%")
                            print(f"% Tassi negativi: {history['negative_pct']:.1f}%")
                            print(f"Trend: {history['trend']}")
                        else:
                            print(f"Errore nell'analisi storica: {history.get('message', 'Unknown error')}")
                            
                except ImportError:
                    print("Modulo di analisi funding non disponibile")
                except Exception as e:
                    print(f"Errore nell'analisi funding: {str(e)}")
                    
            elif cmd == "report":
                print("Generazione report in corso...")
                try:
                    report = bot.generate_report()
                    
                    if report and 'report' in report:
                        report_data = report['report']
                        overall_stats = report_data.get('overall_stats', {})
                        
                        print("\nReport di performance:")
                        print(f"Trades totali: {overall_stats.get('total_trades', 0)}")
                        print(f"Win rate: {overall_stats.get('win_rate', 0) * 100:.2f}%")
                        print(f"Profit factor: {overall_stats.get('profit_factor', 0):.2f}")
                        print(f"Profitto totale: {overall_stats.get('total_profit', 0):.2f} USDT")
                        print(f"Max drawdown: {overall_stats.get('max_drawdown', 0):.2f} USDT ({overall_stats.get('drawdown_percent', 0):.2f}%)")
                        
                        # Mostra migliori simboli
                        best_symbols = report_data.get('best_symbols', [])
                        if best_symbols:
                            print("\nMigliori simboli:")
                            for i, symbol_data in enumerate(best_symbols):
                                print(f"  {i+1}. {symbol_data['symbol']} - "
                                    f"Win rate: {symbol_data['win_rate']*100:.2f}%, "
                                    f"Profit: {symbol_data['total_profit']:.2f} USDT")
                        
                        # Statistiche giornaliere recenti
                        daily_stats = report_data.get('daily_stats', {})
                        if daily_stats:
                            print("\nStatistiche giornaliere recenti:")
                            for day, stats in list(daily_stats.items())[-5:]:  # Ultime 5 giornate
                                win_rate = 0
                                if stats['total_trades'] > 0:
                                    win_rate = stats['winning_trades']/stats['total_trades']*100
                                print(f"  {day}: {stats['total_profit']:.2f} USDT, "
                                    f"Trades: {stats['total_trades']}, "
                                    f"Win rate: {win_rate:.1f}%")
                        
                        # Path dei grafici generati
                        if 'chart_paths' in report and report['chart_paths']:
                            print("\nGrafici generati:")
                            for path in report['chart_paths']:
                                print(f"  {path}")
                    else:
                        print("Nessun report generato")
                except Exception as e:
                    print(f"Errore nella generazione del report: {str(e)}")
                    
            elif cmd == "chart":
                if not args:
                    print("Specificare un simbolo (es: chart BTC/USDT)")
                    continue
                
                parts = args.split()
                symbol = parts[0]
                timeframe = parts[1] if len(parts) > 1 else "1h"
                
                print(f"Generazione grafico per {symbol} ({timeframe})...")
                try:
                    chart_path = bot.generate_chart(symbol, timeframe)
                    
                    if chart_path:
                        print(f"Grafico generato: {chart_path}")
                        # Invia su Telegram se configurato
                        if bot.telegram:
                            bot.telegram.send_chart(chart_path, f"Grafico {symbol} - {timeframe}")
                            print("Grafico inviato via Telegram")
                    else:
                        print(f"Impossibile generare il grafico per {symbol}")
                except Exception as e:
                    print(f"Errore nella generazione del grafico: {str(e)}")
                    
            elif cmd == "train":
                if not args:
                    print("Avvio addestramento per tutti i simboli e modelli...")
                    # Addestra tutti i modelli per tutti i simboli
                    for symbol in bot.trading_pairs:
                        try:
                            print(f"Addestramento modelli per {symbol}...")
                            # Recupera dati aggiornati
                            data = bot.market_data.get_data_with_indicators(symbol, force_update=True)
                            
                            # Rimuovi valori infiniti o NaN
                            data = data.replace([np.inf, -np.inf], np.nan)
                            data = data.dropna()
                            
                            if not data.empty and len(data) > 100:
                                # Prepara i dati
                                train_data = prepare_training_data(
                                    data, 
                                    generate_directional_target,
                                    future_periods=10,
                                    binary=True
                                )
                                # Addestra il modello
                                bot.ml_strategy.train_model(symbol, train_data)
                                print(f"Modello per {symbol} addestrato con successo")
                            else:
                                print(f"Dati insufficienti per {symbol}: {len(data) if not data.empty else 0} candele")
                        except Exception as e:
                            print(f"Errore nell'addestramento per {symbol}: {str(e)}")
                else:
                    # Addestra solo il simbolo specificato
                    symbol = args
                    print(f"Addestramento modelli per {symbol}...")
                    try:
                        # Recupera dati aggiornati
                        data = bot.market_data.get_data_with_indicators(symbol, force_update=True)
                        
                        # Rimuovi valori infiniti o NaN
                        data = data.replace([np.inf, -np.inf], np.nan)
                        data = data.dropna()
                        
                        if not data.empty and len(data) > 100:
                            # Prepara i dati
                            train_data = prepare_training_data(
                                data, 
                                generate_directional_target,
                                future_periods=10,
                                binary=True
                            )
                            # Addestra il modello
                            bot.ml_strategy.train_model(symbol, train_data)
                            print(f"Modello per {symbol} addestrato con successo")
                        else:
                            print(f"Dati insufficienti per {symbol}: {len(data) if not data.empty else 0} candele")
                    except Exception as e:
                        print(f"Errore nell'addestramento per {symbol}: {str(e)}")
                        
            elif cmd == "settings":
                # Mostra impostazioni correnti
                from config.settings import (
                    RISK_PER_TRADE_PERCENT, MAX_POSITION_SIZE_PERCENT, MAX_CONCURRENT_TRADES,
                    STOP_LOSS_PERCENT, TAKE_PROFIT_LEVELS, DEFAULT_LEVERAGE, DEFAULT_MARGIN_MODE
                )
                
                print("Impostazioni correnti:")
                print(f"Rischio per trade: {RISK_PER_TRADE_PERCENT}%")
                print(f"Dimensione massima posizione: {MAX_POSITION_SIZE_PERCENT}%")
                print(f"Trade simultanei massimi: {MAX_CONCURRENT_TRADES}")
                print(f"Stop loss predefinito: {STOP_LOSS_PERCENT}%")
                print("Livelli take profit:")
                for i, tp in enumerate(TAKE_PROFIT_LEVELS):
                    print(f"  TP{i+1}: {tp['percent']}% (chiude {tp['size_percent']}% della posizione)")
                print(f"Leverage predefinito: {DEFAULT_LEVERAGE}x")
                print(f"Margin mode: {DEFAULT_MARGIN_MODE}")
                
                print("\nCoppie di trading:")
                for symbol in bot.trading_pairs:
                    print(f"  {symbol}")
                    
            else:
                print(f"Comando non riconosciuto: {cmd}")
                print("Digita 'help' per la lista dei comandi")
                
        except KeyboardInterrupt:
            bot.stop()
            agent_manager.stop()
            agent_manager.save_agents()
            print("\nArrivederci!")
            break
        except Exception as e:
            print(f"Errore: {str(e)}")

def main():
    """Funzione principale"""
    try:
        # Analizza gli argomenti
        args = parse_arguments()
        
        # Configura il logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Avvio {BOT_NAME} v{BOT_VERSION} in modalità {args.mode}")
        
        # Se è richiesto, inizializza i modelli ML prima di avviare il bot
        if args.init_models:
            symbols = args.pairs if args.pairs else DEFAULT_TRADING_PAIRS
            init_ml_models(symbols, timeframes=args.timeframes, days=args.days)
        
        # Crea l'istanza del bot
        bot = TradingBot()
        
        # Crea il gestore degli agenti
        agent_manager = AgentManager()
        
        # Configura la modalità
        if args.mode == "live":
            bot.set_mode(BotMode.LIVE)
        else:
            bot.set_mode(BotMode.DEMO)
        
        # Configura le coppie di trading
        if args.pairs:
            bot.trading_pairs = []
            for pair in args.pairs:
                bot.add_trading_pair(pair)
        
        # Configura il rischio
        if args.risk:
            bot.set_risk_per_trade(args.risk)
        
        # Configura i gestori dei segnali
        setup_signal_handlers(bot)
        
        # Carica gli agenti
        agent_manager.load_agents()
        
        # Avvia il gestore
        # Avvia il gestore degli agenti
        agent_manager.start()
        
        # Avvia il bot se richiesto
        if args.start:
            bot.start()
        
        # Modalità console
        if args.console:
            console_mode(bot, agent_manager)
        else:
            # Tieni il processo in esecuzione
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Interruzione da tastiera ricevuta, arresto in corso...")
        if 'bot' in locals():
            bot.stop()
        if 'agent_manager' in locals():
            agent_manager.stop()
            agent_manager.save_agents()
    except Exception as e:
        logger.error(f"Errore nell'avvio del bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()