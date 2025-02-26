"""
Modulo per la gestione dei comandi del bot
"""
from typing import Dict, List, Optional, Any, Callable, Union
import re
import inspect
import textwrap
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)

class CommandProcessor:
    """Classe per la gestione dei comandi del bot"""
    
    def __init__(self):
        """Inizializza il processore dei comandi"""
        self.logger = get_logger(__name__)
        
        # Dizionario dei comandi registrati (comando -> handler)
        self.commands: Dict[str, Callable] = {}
        
        # Descrizioni dei comandi (comando -> descrizione)
        self.command_descriptions: Dict[str, str] = {}
        
        # Gruppi di comandi (gruppo -> lista comandi)
        self.command_groups: Dict[str, List[str]] = {}
        
        # Aliases dei comandi (alias -> comando)
        self.command_aliases: Dict[str, str] = {}
        
        # Registra il comando di help
        self.register_command("help", self.cmd_help, "Mostra l'elenco dei comandi disponibili", group="generali")
        
        self.logger.info("CommandProcessor inizializzato")
    
    def register_command(self, command: str, handler: Callable, 
                        description: str = "", group: str = "altro", 
                        aliases: List[str] = None) -> None:
        """
        Registra un comando
        
        Args:
            command: Nome del comando
            handler: Funzione da chiamare quando il comando viene eseguito
            description: Descrizione del comando
            group: Gruppo del comando
            aliases: Alias per il comando
        """
        command = command.lower()
        
        # Registra il comando
        self.commands[command] = handler
        self.command_descriptions[command] = description
        
        # Aggiungi al gruppo
        if group not in self.command_groups:
            self.command_groups[group] = []
            
        self.command_groups[group].append(command)
        
        # Registra gli alias
        if aliases:
            for alias in aliases:
                self.command_aliases[alias.lower()] = command
        
        self.logger.info(f"Comando '{command}' registrato")
    
    def process_command(self, command_text: str) -> str:
        """
        Elabora un comando
        
        Args:
            command_text: Testo del comando
            
        Returns:
            Risultato dell'elaborazione
        """
        if not command_text:
            return "Comando non specificato. Usa /help per vedere i comandi disponibili."
        
        # Rimuovi lo slash iniziale se presente
        if command_text.startswith("/"):
            command_text = command_text[1:]
        
        # Dividi il comando dagli argomenti
        parts = command_text.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Verifica se è un alias
        if command in self.command_aliases:
            command = self.command_aliases[command]
        
        # Verifica se il comando esiste
        if command not in self.commands:
            return f"Comando '{command}' non riconosciuto. Usa /help per vedere i comandi disponibili."
        
        try:
            # Esegui il comando
            handler = self.commands[command]
            result = handler(args)
            
            # Se il risultato è None, restituisci un messaggio generico
            if result is None:
                result = f"Comando '{command}' eseguito con successo."
                
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione del comando '{command}': {str(e)}")
            return f"Errore nell'esecuzione del comando '{command}': {str(e)}"
    
    def cmd_help(self, args: str = "") -> str:
        """
        Comando di help
        
        Args:
            args: Argomenti del comando (opzionale)
            
        Returns:
            Messaggio di help
        """
        # Se è specificato un comando, mostra l'help per quel comando
        if args:
            command = args.lower()
            
            # Verifica se è un alias
            if command in self.command_aliases:
                command = self.command_aliases[command]
            
            # Verifica se il comando esiste
            if command not in self.commands:
                return f"Comando '{command}' non riconosciuto."
            
            # Ottieni la descrizione
            description = self.command_descriptions.get(command, "Nessuna descrizione disponibile.")
            
            # Ottieni i parametri dalla firma della funzione
            handler = self.commands[command]
            signature = inspect.signature(handler)
            params = []
            
            for name, param in signature.parameters.items():
                if name != "self" and name != "cls":
                    if param.default == inspect.Parameter.empty:
                        params.append(f"<{name}>")
                    else:
                        params.append(f"[{name}]")
            
            # Formatta l'help
            help_text = f"Comando: /{command}\n"
            
            if params:
                help_text += f"Utilizzo: /{command} {' '.join(params)}\n"
            
            help_text += f"Descrizione: {description}\n"
            
            # Aggiungi gli alias
            aliases = [a for a, c in self.command_aliases.items() if c == command]
            if aliases:
                help_text += f"Alias: {', '.join(['/' + a for a in aliases])}\n"
            
            return help_text
        
        # Altrimenti, mostra l'elenco di tutti i comandi
        help_text = "Comandi disponibili:\n\n"
        
        # Raggruppa i comandi
        for group, commands in sorted(self.command_groups.items()):
            # Salta gruppi vuoti
            if not commands:
                continue
                
            help_text += f"== {group.upper()} ==\n"
            
            for command in sorted(commands):
                # Ottieni la descrizione breve (prima riga)
                description = self.command_descriptions.get(command, "")
                if description:
                    description = description.split("\n")[0]
                    
                    # Tronca la descrizione se troppo lunga
                    if len(description) > 50:
                        description = description[:47] + "..."
                
                help_text += f"/{command} - {description}\n"
            
            help_text += "\n"
        
        help_text += "Usa /help <comando> per maggiori informazioni su un comando specifico."
        
        return help_text
    
    def get_available_commands(self) -> Dict[str, Dict[str, Any]]:
        """
        Ottiene l'elenco dei comandi disponibili
        
        Returns:
            Dizionario dei comandi disponibili
        """
        commands = {}
        
        for command in self.commands:
            commands[command] = {
                "description": self.command_descriptions.get(command, ""),
                "group": next((g for g, cmds in self.command_groups.items() if command in cmds), "altro"),
                "aliases": [a for a, c in self.command_aliases.items() if c == command]
            }
        
        return commands
    
    def parse_arguments(self, args_str: str, expected_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza gli argomenti di un comando
        
        Args:
            args_str: Stringa degli argomenti
            expected_args: Dizionario degli argomenti attesi (nome -> valore predefinito)
            
        Returns:
            Dizionario degli argomenti analizzati
        """
        # Inizializza con i valori predefiniti
        parsed_args = expected_args.copy()
        
        if not args_str:
            return parsed_args
        
        # Dividi gli argomenti
        args_parts = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")++', args_str)
        
        # Analizza gli argomenti posizionali e nominativi
        position = 0
        for arg in args_parts:
            # Verifica se è un argomento nominativo
            match = re.match(r'([a-zA-Z0-9_-]+)=(.+)', arg)
            
            if match:
                # Argomento nominativo
                name, value = match.groups()
                
                # Rimuovi le virgolette se presenti
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                # Converti il valore al tipo corretto
                if name in parsed_args:
                    if isinstance(parsed_args[name], bool):
                        parsed_args[name] = value.lower() in ["true", "1", "yes", "y", "t"]
                    elif isinstance(parsed_args[name], int):
                        try:
                            parsed_args[name] = int(value)
                        except ValueError:
                            pass
                    elif isinstance(parsed_args[name], float):
                        try:
                            parsed_args[name] = float(value)
                        except ValueError:
                            pass
                    else:
                        parsed_args[name] = value
            else:
                # Argomento posizionale
                # Rimuovi le virgolette se presenti
                if arg.startswith('"') and arg.endswith('"'):
                    arg = arg[1:-1]
                
                # Assegna all'argomento posizionale corrispondente
                arg_names = list(parsed_args.keys())
                if position < len(arg_names):
                    name = arg_names[position]
                    
                    # Converti il valore al tipo corretto
                    if isinstance(parsed_args[name], bool):
                        parsed_args[name] = arg.lower() in ["true", "1", "yes", "y", "t"]
                    elif isinstance(parsed_args[name], int):
                        try:
                            parsed_args[name] = int(arg)
                        except ValueError:
                            pass
                    elif isinstance(parsed_args[name], float):
                        try:
                            parsed_args[name] = float(arg)
                        except ValueError:
                            pass
                    else:
                        parsed_args[name] = arg
                        
                    position += 1
        
        return parsed_args
