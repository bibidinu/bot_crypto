"""
Modulo per la gestione della leva e del rischio associato nei futures
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from api.bybit_api import BybitAPI
from utils.logger import get_logger
from config.settings import (
    DEFAULT_LEVERAGE, MAX_LEVERAGE, DEFAULT_MARGIN_MODE,
    RISK_PER_TRADE_PERCENT, MAX_POSITION_SIZE_PERCENT
)

logger = get_logger(__name__)

class LeverageManager:
    """Classe per la gestione della leva e del rischio nei futures"""
    
    def __init__(self, exchange: BybitAPI):
        """
        Inizializza il gestore della leva
        
        Args:
            exchange: Istanza dell'exchange API
        """
        self.exchange = exchange
        self.logger = get_logger(__name__)
        
        # Cache delle informazioni sugli strumenti
        self.instrument_info_cache = {}
        
        # Cache delle impostazioni di leva per simbolo
        self.leverage_settings = {}
        
        # Cache dei limiti di rischio
        self.risk_limits_cache = {}
        
        # Tracciamento della volatilità
        self.volatility_cache = {}
        
        self.logger.info("LeverageManager inizializzato")
    
    def get_instrument_info(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Ottiene e memorizza nella cache le informazioni sullo strumento
        
        Args:
            symbol: Simbolo della coppia
            force_refresh: Se forzare un aggiornamento dalla cache
            
        Returns:
            Informazioni sullo strumento
        """
        if symbol not in self.instrument_info_cache or force_refresh:
            try:
                info = self.exchange.get_instrument_info(symbol)
                self.instrument_info_cache[symbol] = info
                return info
            except Exception as e:
                self.logger.error(f"Errore nel recupero delle informazioni per {symbol}: {str(e)}")
                return {}
        
        return self.instrument_info_cache[symbol]
    
    def get_max_leverage(self, symbol: str) -> float:
        """
        Ottiene il leverage massimo consentito per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Leverage massimo consentito
        """
        info = self.get_instrument_info(symbol)
        
        try:
            max_leverage = float(info.get("leverageFilter", {}).get("maxLeverage", DEFAULT_LEVERAGE))
            return min(max_leverage, MAX_LEVERAGE)
        except (KeyError, ValueError, TypeError):
            return DEFAULT_LEVERAGE
    
    def get_risk_limits(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Ottiene i limiti di rischio per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            
        Returns:
            Lista dei limiti di rischio
        """
        if symbol not in self.risk_limits_cache:
            info = self.get_instrument_info(symbol)
            
            try:
                risk_limits = info.get("riskLimitInfo", [])
                if risk_limits:
                    self.risk_limits_cache[symbol] = risk_limits
                    return risk_limits
            except (KeyError, ValueError, TypeError):
                pass
            
            # Default risk limit
            self.risk_limits_cache[symbol] = [
                {"id": 1, "limit": 1000000, "maintainMargin": 0.01, "initialMargin": 0.02},
                {"id": 2, "limit": 5000000, "maintainMargin": 0.015, "initialMargin": 0.03},
                {"id": 3, "limit": 10000000, "maintainMargin": 0.02, "initialMargin": 0.04}
            ]
        
        return self.risk_limits_cache[symbol]
    
    def get_risk_limit_id_for_position_size(self, symbol: str, position_value: float) -> int:
        """
        Determina l'ID del limite di rischio appropriato per una dimensione di posizione
        
        Args:
            symbol: Simbolo della coppia
            position_value: Valore della posizione in USD
            
        Returns:
            ID del limite di rischio
        """
        risk_limits = self.get_risk_limits(symbol)
        
        # Ordina i limiti in ordine crescente
        sorted_limits = sorted(risk_limits, key=lambda x: x.get("limit", 0))
        
        # Trova il limite appropriato per la dimensione della posizione
        for risk_limit in sorted_limits:
            limit = float(risk_limit.get("limit", 0))
            
            if position_value <= limit:
                return int(risk_limit.get("id", 1))
        
        # Se la posizione è più grande di tutti i limiti, usa l'ultimo
        return int(sorted_limits[-1].get("id", 1)) if sorted_limits else 1
    
    def calculate_dynamic_leverage(self, symbol: str, current_price: float, 
                                  capital: float, risk_percent: float = RISK_PER_TRADE_PERCENT,
                                  volatility_period: int = 14) -> float:
        """
        Calcola un valore di leva dinamico basato sulla volatilità del mercato
        
        Args:
            symbol: Simbolo della coppia
            current_price: Prezzo corrente
            capital: Capitale disponibile
            risk_percent: Percentuale di rischio per trade
            volatility_period: Periodo per il calcolo della volatilità
            
        Returns:
            Leverage raccomandato
        """
        # Ottieni la volatilità del mercato
        volatility = self.get_market_volatility(symbol, volatility_period)
        
        # Massimo leverage consentito
        max_leverage = self.get_max_leverage(symbol)
        
        # Se la volatilità non è disponibile, usa un approccio conservativo
        if volatility <= 0:
            return min(DEFAULT_LEVERAGE, max_leverage)
        
        # Formula: leverage = max_leverage * (1 / volatility) * adjustment_factor
        # Dove adjustment_factor è un valore tra 0.5 e 1.0 basato sul rischio
        adjustment_factor = 0.75 * (risk_percent / RISK_PER_TRADE_PERCENT)
        
        # Calcola il leverage dinamico
        dynamic_leverage = (1 / volatility) * adjustment_factor * max_leverage
        
        # Limite min/max
        dynamic_leverage = max(1.0, min(dynamic_leverage, max_leverage))
        
        # Arrotonda a un decimale
        dynamic_leverage = round(dynamic_leverage, 1)
        
        self.logger.info(f"Leverage dinamico calcolato per {symbol}: {dynamic_leverage}x " +
                        f"(volatilità: {volatility:.3f}, max: {max_leverage}x)")
        
        return dynamic_leverage
    
    def get_market_volatility(self, symbol: str, period: int = 14) -> float:
        """
        Calcola la volatilità attuale del mercato
        
        Args:
            symbol: Simbolo della coppia
            period: Periodo per il calcolo della volatilità
            
        Returns:
            Valore di volatilità (deviazione standard dei rendimenti giornalieri)
        """
        try:
            # Usa la cache se disponibile e aggiornata (max 1 ora)
            cache_key = f"{symbol}_{period}"
            now = datetime.now()
            
            if (cache_key in self.volatility_cache and 
                now - self.volatility_cache[cache_key]["time"] < timedelta(hours=1)):
                return self.volatility_cache[cache_key]["value"]
            
            # Ottieni i dati storici
            data = self.exchange.get_klines(symbol, "1d", limit=period + 10)
            
            if data.empty or len(data) < period:
                return 0.1  # Default conservativo se non ci sono abbastanza dati
            
            # Calcola i rendimenti giornalieri
            returns = data['close'].pct_change().dropna()
            
            # Calcola la volatilità come deviazione standard dei rendimenti
            volatility = returns.std()
            
            # Se la volatilità è troppo bassa, usa un minimo
            volatility = max(volatility, 0.01)
            
            # Aggiorna la cache
            self.volatility_cache[cache_key] = {
                "value": volatility,
                "time": now
            }
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo della volatilità per {symbol}: {str(e)}")
            return 0.1  # Default conservativo in caso di errore
    
    def setup_position_settings(self, symbol: str, capital: float, 
                               risk_percent: float = RISK_PER_TRADE_PERCENT,
                               margin_mode: str = DEFAULT_MARGIN_MODE,
                               leverage: Optional[float] = None,
                               position_mode: str = "MergedSingle") -> Dict[str, Any]:
        """
        Configura le impostazioni della posizione per un simbolo
        
        Args:
            symbol: Simbolo della coppia
            capital: Capitale disponibile
            risk_percent: Percentuale di rischio per trade
            margin_mode: Modalità di margine ("ISOLATED" o "CROSS")
            leverage: Valore della leva (se None, calcola dinamicamente)
            position_mode: Modalità di posizione ("MergedSingle" o "BothSide")
            
        Returns:
            Impostazioni configurate
        """
        try:
            # Ottieni il prezzo corrente
            ticker = self.exchange.get_ticker(symbol)
            current_price = float(ticker.get("lastPrice", 0))
            
            if current_price <= 0:
                self.logger.error(f"Prezzo non valido per {symbol}: {current_price}")
                return {}
            
            # Calcola o usa il leverage specificato
            if leverage is None:
                leverage = self.calculate_dynamic_leverage(symbol, current_price, capital, risk_percent)
            
            # Imposta il leverage
            self.exchange.set_leverage(symbol, leverage)
            
            # Imposta la modalità di margine
            self.exchange.set_margin_mode(symbol, margin_mode)
            
            # Imposta la modalità di posizione
            self.exchange.set_position_mode(symbol, position_mode)
            
            # Imposta la modalità TPSL
            self.exchange.set_tpsl_mode(symbol, "Partial")  # Usiamo la modalità parziale per multi-TP
            
            # Salva le impostazioni nella cache
            self.leverage_settings[symbol] = {
                "leverage": leverage,
                "margin_mode": margin_mode,
                "position_mode": position_mode,
                "updated_at": datetime.now()
            }
            
            self.logger.info(f"Impostazioni di posizione configurate per {symbol}: " +
                           f"leverage={leverage}x, mode={margin_mode}, position={position_mode}")
            
            return self.leverage_settings[symbol]
            
        except Exception as e:
            self.logger.error(f"Errore nella configurazione delle impostazioni per {symbol}: {str(e)}")
            return {}
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, capital: float,
                               leverage: float = DEFAULT_LEVERAGE, 
                               risk_percent: float = RISK_PER_TRADE_PERCENT) -> float:
        """
        Calcola la dimensione della posizione in base al rischio e alla leva
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            stop_loss: Prezzo di stop loss
            capital: Capitale disponibile
            leverage: Valore della leva
            risk_percent: Percentuale di rischio per trade
            
        Returns:
            Dimensione della posizione
        """
        # Rischio in valuta
        risk_amount = capital * (risk_percent / 100)
        
        # Percentuale di perdita da entry a stop loss
        price_diff_percent = abs((entry_price - stop_loss) / entry_price)
        
        if price_diff_percent <= 0 or entry_price <= 0:
            self.logger.error(f"Prezzi non validi per {symbol}: entry={entry_price}, stop={stop_loss}")
            return 0.0
        
        # Con leva, il capitale effettivo è moltiplicato
        effective_capital = capital * leverage
        
        # Calcola la dimensione della posizione
        # Importante: consideriamo che con leva se il prezzo si muove del price_diff_percent
        # perdiamo price_diff_percent * leverage del capitale
        position_size = risk_amount / (price_diff_percent * entry_price * leverage)
        
        # Verifica che la posizione non sia troppo grande rispetto al capitale
        max_position_value = capital * (MAX_POSITION_SIZE_PERCENT / 100)
        position_value = position_size * entry_price
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            self.logger.info(f"Posizione ridotta per {symbol} per rispettare il limite massimo di posizione")
        
        # Verifica che la posizione rispetti i limiti dello strumento
        info = self.get_instrument_info(symbol)
        
        try:
            # Ottieni i limiti di dimensione
            lot_size_filter = info.get("lotSizeFilter", {})
            min_qty = float(lot_size_filter.get("minOrderQty", 0))
            max_qty = float(lot_size_filter.get("maxOrderQty", float('inf')))
            qty_step = float(lot_size_filter.get("qtyStep", 0.001))
            
            # Arrotonda la dimensione al tick step più vicino
            position_size = round(position_size / qty_step) * qty_step
            
            # Verifica i limiti min/max
            position_size = max(min_qty, min(position_size, max_qty))
            
        except (KeyError, ValueError, TypeError):
            self.logger.warning(f"Impossibile applicare i filtri di dimensione per {symbol}")
        
        self.logger.info(f"Dimensione posizione calcolata per {symbol}: {position_size} " +
                        f"(valore: {position_size * entry_price:.2f}, rischio: {risk_amount:.2f})")
        
        return position_size
    
    def calculate_liquidation_price(self, symbol: str, entry_price: float, 
                                  position_size: float, leverage: float,
                                  direction: str, capital: float = None,
                                  margin_mode: str = "ISOLATED") -> float:
        """
        Calcola il prezzo di liquidazione per una posizione con leva
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            position_size: Dimensione della posizione
            leverage: Valore della leva
            direction: Direzione della posizione ('long' o 'short')
            capital: Capitale allocato (necessario per CROSS margin)
            margin_mode: Modalità di margine ("ISOLATED" o "CROSS")
            
        Returns:
            Prezzo di liquidazione stimato
        """
        try:
            # Ottieni le informazioni sullo strumento
            info = self.get_instrument_info(symbol)
            
            # Per semplificare, assumiamo che il maintenance margin sia del 0.5% per ISOLATED
            # e 0.35% per CROSS, salvo diversa specifica dello strumento
            maintenance_margin_rate = 0.005 if margin_mode == "ISOLATED" else 0.0035
            
            # Se disponibile, ottieni il tasso esatto
            try:
                risk_limits = self.get_risk_limits(symbol)
                position_value = position_size * entry_price
                
                for limit in risk_limits:
                    if position_value <= float(limit.get("limit", 0)):
                        maintenance_margin_rate = float(limit.get("maintainMargin", maintenance_margin_rate))
                        break
            except:
                pass
            
            # Calcola il margine richiesto per la posizione
            initial_margin = position_size * entry_price / leverage
            
            # Calcola il prezzo di liquidazione
            if direction.lower() == 'long':
                # Formula: liq_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
                liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
            else:  # short
                # Formula: liq_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
                liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
            
            # Per CROSS margin, il prezzo di liquidazione è più lontano perché
            # può usare tutto il capitale disponibile
            if margin_mode == "CROSS" and capital is not None and capital > initial_margin:
                # Adatta la formula per considerare tutto il capitale
                leverage_modifier = (capital - initial_margin) / (position_size * entry_price)
                if direction.lower() == 'long':
                    liquidation_price = entry_price * (1 - (1/leverage) - leverage_modifier + maintenance_margin_rate)
                else:  # short
                    liquidation_price = entry_price * (1 + (1/leverage) + leverage_modifier - maintenance_margin_rate)
            
            # Assicurati che il prezzo di liquidazione sia valido (non negativo)
            liquidation_price = max(0, liquidation_price)
            
            self.logger.info(f"Prezzo di liquidazione per {symbol} {direction}: {liquidation_price:.5f} " +
                           f"(entry: {entry_price:.5f}, leverage: {leverage}x)")
            
            return liquidation_price
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo del prezzo di liquidazione per {symbol}: {str(e)}")
            
            # In caso di errore, restituisci una stima approssimativa
            if direction.lower() == 'long':
                return entry_price * 0.80  # -20% dal prezzo di entrata
            else:
                return entry_price * 1.20  # +20% dal prezzo di entrata
    
    def calculate_safe_stop_loss(self, symbol: str, entry_price: float, 
                               position_size: float, leverage: float,
                               direction: str, capital: float = None,
                               margin_mode: str = "ISOLATED", 
                               min_distance_percent: float = 5.0) -> float:
        """
        Calcola uno stop loss sicuro che sia sufficientemente lontano dal prezzo di liquidazione
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            position_size: Dimensione della posizione
            leverage: Valore della leva
            direction: Direzione della posizione ('long' o 'short')
            capital: Capitale allocato
            margin_mode: Modalità di margine ("ISOLATED" o "CROSS")
            min_distance_percent: Distanza minima dallo stop loss al prezzo di liquidazione (%)
            
        Returns:
            Prezzo di stop loss sicuro
        """
        try:
            # Calcola il prezzo di liquidazione
            liquidation_price = self.calculate_liquidation_price(
                symbol, entry_price, position_size, leverage, direction, capital, margin_mode
            )
            
            # Calcola una distanza sicura dal prezzo di liquidazione
            # Più è alto il leverage, più lo stop dovrebbe essere lontano dalla liquidazione
            safe_distance_percent = max(min_distance_percent, 2 * min_distance_percent * (leverage / 10))
            
            # Calcola lo stop loss sicuro
            if direction.lower() == 'long':
                # Per posizioni long, lo stop loss è sotto il prezzo di entrata
                # Deve essere sopra il prezzo di liquidazione di almeno safe_distance_percent
                liquidation_distance_percent = (entry_price - liquidation_price) / entry_price * 100
                stop_loss_distance_percent = min(liquidation_distance_percent * 0.7, max(0.5 * liquidation_distance_percent, safe_distance_percent))
                stop_loss = entry_price * (1 - stop_loss_distance_percent / 100)
                
                # Assicurati che lo stop loss sia sopra il prezzo di liquidazione
                min_stop_loss = liquidation_price * (1 + safe_distance_percent / 100)
                stop_loss = max(stop_loss, min_stop_loss)
                
            else:  # short
                # Per posizioni short, lo stop loss è sopra il prezzo di entrata
                # Deve essere sotto il prezzo di liquidazione di almeno safe_distance_percent
                liquidation_distance_percent = (liquidation_price - entry_price) / entry_price * 100
                stop_loss_distance_percent = min(liquidation_distance_percent * 0.7, max(0.5 * liquidation_distance_percent, safe_distance_percent))
                stop_loss = entry_price * (1 + stop_loss_distance_percent / 100)
                
                # Assicurati che lo stop loss sia sotto il prezzo di liquidazione
                max_stop_loss = liquidation_price * (1 - safe_distance_percent / 100)
                stop_loss = min(stop_loss, max_stop_loss)
            
            self.logger.info(f"Stop loss sicuro per {symbol} {direction}: {stop_loss:.5f} " +
                           f"(entry: {entry_price:.5f}, liquidation: {liquidation_price:.5f})")
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo dello stop loss sicuro per {symbol}: {str(e)}")
            
            # In caso di errore, usa uno stop loss predefinito
            if direction.lower() == 'long':
                return entry_price * 0.95  # -5% dal prezzo di entrata
            else:
                return entry_price * 1.05  # +5% dal prezzo di entrata
    
    def calculate_take_profit_levels(self, symbol: str, entry_price: float, 
                                   leverage: float, direction: str, 
                                   risk_reward_ratios: List[float] = [1.5, 3.0, 5.0],
                                   stop_loss: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Calcola livelli di take profit ottimali per una posizione con leva
        
        Args:
            symbol: Simbolo della coppia
            entry_price: Prezzo di entrata
            leverage: Valore della leva
            direction: Direzione della posizione ('long' o 'short')
            risk_reward_ratios: Rapporti rischio/rendimento per i livelli
            stop_loss: Prezzo di stop loss (opzionale)
            
        Returns:
            Lista di livelli take profit con prezzi e dimensioni
        """
        try:
            # Se lo stop loss non è specificato, usa un valore predefinito
            if stop_loss is None:
                if direction.lower() == 'long':
                    stop_loss = entry_price * 0.95  # -5% dal prezzo di entrata
                else:
                    stop_loss = entry_price * 1.05  # +5% dal prezzo di entrata
            
            # Calcola la distanza di rischio
            risk_distance = abs(entry_price - stop_loss)
            
            # Per futures con leva, adatta le distanze in base alla leva
            # Maggiore è la leva, più piccole sono le distanze percentuali
            leverage_adjusted_ratios = [ratio / np.sqrt(leverage) for ratio in risk_reward_ratios]
            
            # Calcola i livelli di take profit
            take_profit_levels = []
            
            for i, ratio in enumerate(leverage_adjusted_ratios):
                # Calcola il prezzo di take profit
                if direction.lower() == 'long':
                    take_profit_price = entry_price + (risk_distance * ratio)
                else:
                    take_profit_price = entry_price - (risk_distance * ratio)
                
                # Calcola la percentuale di posizione da chiudere
                # Usiamo un approccio a piramide inversa (più grande il primo TP)
                size_percents = [0.5, 0.3, 0.2]  # 50%, 30%, 20%
                size_percent = size_percents[i] if i < len(size_percents) else 0.2
                
                # Aggiungi il livello
                take_profit_levels.append({
                    "price": take_profit_price,
                    "size_percent": size_percent,
                    "r_multiple": ratio
                })
            
            self.logger.info(f"Take profit calcolati per {symbol} {direction}: " +
                           f"{[round(tp['price'], 5) for tp in take_profit_levels]}")
            
            return take_profit_levels
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo dei take profit per {symbol}: {str(e)}")
            
            # In caso di errore, usa take profit predefiniti
            if direction.lower() == 'long':
                return [
                    {"price": entry_price * 1.02, "size_percent": 0.5, "r_multiple": 1.5},
                    {"price": entry_price * 1.04, "size_percent": 0.3, "r_multiple": 3.0},
                    {"price": entry_price * 1.06, "size_percent": 0.2, "r_multiple": 5.0}
                ]
            else:
                return [
                    {"price": entry_price * 0.98, "size_percent": 0.5, "r_multiple": 1.5},
                    {"price": entry_price * 0.96, "size_percent": 0.3, "r_multiple": 3.0},
                    {"price": entry_price * 0.94, "size_percent": 0.2, "r_multiple": 5.0}
                ]
    
    def check_leverage_safety(self, symbol: str, leverage: float, 
                           volatility_period: int = 14) -> Tuple[bool, str, float]:
        """
        Verifica se il leverage impostato è sicuro considerando la volatilità attuale
        
        Args:
            symbol: Simbolo della coppia
            leverage: Valore della leva da verificare
            volatility_period: Periodo per il calcolo della volatilità
            
        Returns:
            Tupla (è_sicuro, motivo, leverage_suggerito)
        """
        try:
            # Ottieni la volatilità del mercato
            volatility = self.get_market_volatility(symbol, volatility_period)
            
            # Leverage massimo consentito
            max_leverage = self.get_max_leverage(symbol)
            
            # Calcola un leverage sicuro in base alla volatilità
            safe_leverage = 1.0 / (volatility * 4)  # 4x è un fattore di sicurezza
            safe_leverage = min(max(safe_leverage, 1.0), max_leverage)
            safe_leverage = round(safe_leverage, 1)
            
            # Verifica se il leverage richiesto è sicuro
            is_safe = leverage <= safe_leverage
            
            if is_safe:
                reason = f"Leverage {leverage}x è sicuro per la volatilità attuale {volatility:.3f}"
            else:
                reason = f"Leverage {leverage}x è troppo alto per la volatilità attuale {volatility:.3f}. Suggerito: {safe_leverage}x"
            
            return is_safe, reason, safe_leverage
            
        except Exception as e:
            self.logger.error(f"Errore nella verifica del leverage per {symbol}: {str(e)}")
            
            # In caso di errore, suggerisce un leverage conservativo
            return leverage <= 3, f"Errore nella verifica, usando approccio conservativo", min(3, max_leverage)