"""
File credenziali per l'accesso alle API
ATTENZIONE: Non condividere mai questo file!
Questo file dovrebbe essere aggiunto a .gitignore
"""
from typing import Dict

# Credenziali Bybit
BYBIT_CREDENTIALS: Dict[str, Dict[str, str]] = {
    "demo": {
        "api_key": "Tot",
        "api_secret": "RT",
        "testnet": True  # Usa il testnet per la demo
    },
    "live": {
        "api_key": "il_tuo_api_key_live",
        "api_secret": "il_tuo_api_secret_live",
        "testnet": False  # Usa il mainnet per il live trading
    }
}

# Credenziali Discord per il webhook
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/"

# Credenziali Telegram Bot
TELEGRAM_BOT_TOKEN = "il_tuo_token_bot_telegram"
TELEGRAM_CHAT_ID = "id_chat_o_canale_telegram"

# Altre API per sentiment analysis e news
TWITTER_API_KEY = "twitter_api_key"
TWITTER_API_SECRET = "twitter_api_secret"
TWITTER_ACCESS_TOKEN = "twitter_access_token"
TWITTER_ACCESS_SECRET = "twitter_access_secret"

# API per dati economici
ECONOMIC_CALENDAR_API_KEY = "economic_calendar_api_key"
