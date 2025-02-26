# AI Trading Bot

Un bot di trading automatico con intelligenza artificiale per il trading di criptovalute.

## Caratteristiche

- Trading automatico con supporto per Bybit
- Modalità demo e live trading
- AI agents che apprendono dalle strategie migliori
- Analisi tecnica con molteplici indicatori
- Analisi del sentiment dai social e dalle news
- Gestione del rischio avanzata
- Multiple take profit e trailing stop
- Notifiche via Telegram e Discord
- Report dettagliati e grafici
- Interfaccia da console

## Struttura del Progetto

Il bot è organizzato in moduli per una facile manutenzione e estensibilità:

```
tradingbot/
├── config/             # Configurazioni
├── core/               # Componenti principali
├── api/                # Wrapper API exchange
├── strategy/           # Strategie di trading
├── risk_management/    # Gestione del rischio
├── data/               # Raccolta e analisi dati
├── communication/      # Integrazione social
├── stats/              # Performance e reportistica
├── commands/           # Gestione dei comandi
├── utils/              # Funzioni di utilità
├── models/             # Modelli ML e RL
├── database/           # Persistenza dei dati
└── main.py             # Entry point
```

## Requisiti

- Python 3.8+
- Pacchetti Python in requirements.txt
- Accesso API Bybit (per demo/live trading)
- (Opzionale) Bot Telegram
- (Opzionale) Webhook Discord

## Installazione

1. Clona il repository:

```bash
git clone https://github.com/username/ai-trading-bot.git
cd ai-trading-bot
```

2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

3. Configura le credenziali:

Copia `config/credentials.py.example` in `config/credentials.py` e inserisci le tue credenziali API.

4. Modifica le impostazioni nel file `config/settings.py` secondo le tue preferenze.

## Utilizzo

### Modalità Demo (Default)

Avvia il bot in modalità demo per testare le strategie senza rischiare fondi reali:

```bash
python main.py --mode demo --start
```

### Modalità Live

Quando sei pronto per il trading reale, usa la modalità live:

```bash
python main.py --mode live --start
```

### Modalità Console

Per un'interfaccia interattiva, usa la modalità console:

```bash
python main.py --console
```

### Parametri da Linea di Comando

- `--mode` o `-m`: Modalità di esecuzione (`demo` o `live`)
- `--start` o `-s`: Avvia il bot automaticamente
- `--console` o `-c`: Avvia in modalità console
- `--pairs` o `-p`: Specifica le coppie di trading
- `--risk` o `-r`: Imposta la percentuale di rischio per trade
- `--verbose` o `-v`: Output verboso

Esempio:

```bash
python main.py --mode demo --pairs BTC/USDT ETH/USDT --risk 1.5 --console
```

## AI Agents

Il bot utilizza una combinazione di strategie di machine learning e reinforcement learning per ottimizzare continuamente le sue decisioni di trading:

1. **Agents ML**: Utilizzano RandomForest, GradientBoosting e altri algoritmi per prevedere i movimenti di prezzo
2. **Agents RL**: Utilizzano Deep Q-Learning per adattarsi dinamicamente alle condizioni di mercato
3. **Strategie Tecniche**: Analizzano pattern di prezzo, indicatori e volumi

Gli agenti migliorano costantemente con l'esperienza di trading, ottimizzando le strategie e adattandosi alle condizioni di mercato.

## Integrazione con Exchange

Attualmente il bot supporta:

- Bybit (API v5)

È possibile estendere il supporto implementando nuovi wrapper nell'interfaccia `ExchangeInterface`.

## Notifiche

Il bot può inviare notifiche su:

- Telegram: segnali di trading, esecuzione ordini, performance
- Discord: report dettagliati, grafici, panoramica di mercato

## Gestione del Rischio

Il bot include una gestione del rischio completa:

- Stop loss automatici basati su ATR
- Take profit multipli (fino a 3 livelli)
- Trailing stop per massimizzare i profitti
- Controllo dell'esposizione massima
- Dimensionamento delle posizioni basato sul rischio
- Analisi della correlazione per evitare eccessiva esposizione

## Licenza

[MIT License](LICENSE)
