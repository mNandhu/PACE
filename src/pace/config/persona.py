"""
Persona Configuration for PACE

This module defines the Persona class, which is responsible for loading
and managing character personalities from JSON configuration files.
"""

import json
import os
import logging
from rich.console import Console

from src.pace.config.constants import persona_settings

console = Console()
logger = logging.getLogger(__name__)


class Persona:
    """
    Manages a character's persona, loaded from a JSON file.

    This class handles loading persona directives, replacing placeholders,
    and providing easy access to character details.
    """

    def __init__(self, persona_name: str, user_name: str = "user"):
        """
        Initializes the Persona object by loading data from a JSON file.

        Args:
            persona_name (str): The name of the persona to load (e.g., 'sam').
            user_name (str): The name of the user interacting with the persona.
        """
        self.persona_name = persona_name
        self.user_name = user_name
        self.character_name = ""
        self.core_persona_directives = []
        self._load_persona()

    def _load_persona(self):
        """
        Loads the persona configuration from the corresponding JSON file.
        """
        persona_file = os.path.join(
            persona_settings["personas_dir"], f"{self.persona_name}.json"
        )
        if not os.path.exists(persona_file):
            error_msg = f"Persona file not found at '{persona_file}'"
            console.print(f"[bold red]Error: {error_msg}[/bold red]")
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                persona_data = json.load(f)
            logger.info(f"Successfully loaded persona file: {persona_file}")
        except json.JSONDecodeError as e:
            error_msg = f"Error decoding JSON from {persona_file}: {e}"
            console.print(f"[bold red]Error: {error_msg}[/bold red]")
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed to read persona file {persona_file}: {e}"
            console.print(f"[bold red]Error: {error_msg}[/bold red]")
            logger.error(error_msg)
            raise

        self.character_name = persona_data.get(
            "character_name", self.persona_name.capitalize()
        )

        # Replace placeholders in persona directives
        raw_directives = persona_data.get("core_persona_directives", [])
        self.core_persona_directives = [
            self._replace_placeholders(directive) for directive in raw_directives
        ]

    def _replace_placeholders(self, text: str) -> str:
        """
        Replaces {{char}} and {{user}} placeholders in a string.

        Args:
            text (str): The string containing placeholders.

        Returns:
            str: The string with placeholders replaced.
        """
        return text.replace("{{char}}", self.character_name).replace(
            "{{user}}", self.user_name
        )

    def get_system_prompt(self) -> str:
        """
        Generates the complete system prompt from the core directives.

        Returns:
            str: A single string containing all persona directives.
        """
        return "\n".join(self.core_persona_directives)

    @staticmethod
    def get_available_personas() -> list[str]:
        """
        Scans the characters directory for available persona files.

        Returns:
            list[str]: A list of persona names (without the .json extension).
        """
        persona_dir = persona_settings["personas_dir"]
        if not os.path.isdir(persona_dir):
            logger.warning(f"Personas directory not found at: {persona_dir}")
            return []

        try:
            personas = [
                f.replace(".json", "")
                for f in os.listdir(persona_dir)
                if f.endswith(".json")
            ]
            logger.info(f"Found {len(personas)} available personas in {persona_dir}.")
            return personas
        except Exception as e:
            logger.error(f"Error scanning for personas in {persona_dir}: {e}")
            return []
