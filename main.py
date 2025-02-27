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
from datetime import datetime

from core.bot import TradingBot
from core.agent_manager import AgentManager
from config.settings import BOT_NAME, BOT_VERSION, BOT_MODE, BotMode
from utils.logger import get_logger, setup_log_capture

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

def console_mode(bot):
    """
    Esegue il bot in modalità console
    
    Args:
        bot: Istanza del bot
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
                print("Arrivederci!")
                break
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
                print(f"Saldo: {bot.get_wallet_balance():.2f} USDT")
            elif cmd == "positions":
                positions = bot.get_open_positions()
                if positions:
                    print(f"Posizioni aperte ({len(positions)}):")
                    for i, pos in enumerate(positions):
                        print(f"  {i+1}. {pos['symbol']} ({pos['entry_type']}) - "
                              f"Size: {pos['size']:.5f}, "
                              f"Entry: {pos['entry_price']:.5f}, "
                              f"PnL: {pos.get('unrealized_pnl', 0):.2f}")
                else:
                    print("Nessuna posizione aperta")
            elif cmd == "balance":
                balance = bot.get_wallet_balance()
                print(f"Saldo wallet: {balance:.2f} USDT")
            elif cmd == "scan":
                print("Scansione opportunità in corso...")
                opportunities = bot.scan_for_opportunities()
                if opportunities:
                    print(f"Opportunità trovate ({len(opportunities)}):")
                    for i, opp in enumerate(opportunities):
                        print(f"  {i+1}. {opp['symbol']} ({opp['signal']}) - "
                              f"Strength: {opp['signal_strength']:.2f}, "
                              f"Price: {opp['current_price']:.5f}")
                else:
                    print("Nessuna opportunità trovata")
            elif cmd == "analyze":
                if not args:
                    print("Specificare un simbolo (es: analyze BTC/USDT)")
                    continue
                
                print(f"Analisi di {args} in corso...")
                analysis = bot.analyze_symbol(args)
                
                if analysis:
                    print(f"Simbolo: {analysis['symbol']}")
                    print(f"Prezzo: {analysis['current_price']:.5f}")
                    print(f"Segnale: {analysis['signal']} ({analysis['signal_strength']:.2f})")
                    
                    if 'entry_type' in analysis and analysis['entry_type']:
                        print(f"Tipo entrata: {analysis['entry_type']}")
                    
                    if 'stop_loss' in analysis and analysis['stop_loss']:
                        print(f"Stop loss: {analysis['stop_loss']:.5f}")
                    
                    if 'take_profits' in analysis and analysis['take_profits']:
                        for i, tp in enumerate(analysis['take_profits']):
                            print(f"TP{i+1}: {tp:.5f}")
                    
                    if 'reason' in analysis and analysis['reason']:
                        print(f"Motivo: {analysis['reason']}")
                else:
                    print(f"Impossibile analizzare {args}")
            else:
                print(f"Comando non riconosciuto: {cmd}")
                
        except KeyboardInterrupt:
            bot.stop()
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
        
        # Avvia il gestore degli agenti
        agent_manager.start()
        
        # Avvia il bot se richiesto
        if args.start:
            bot.start()
        
        # Modalità console
        if args.console:
            console_mode(bot)
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
    except Exception as e:
        logger.error(f"Errore nell'avvio del bot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()