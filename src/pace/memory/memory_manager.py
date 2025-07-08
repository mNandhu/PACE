"""
Memory Manager Module for Project PACE

This module handles the initialization and management of Mem0 instances
and conversation history management for the PACE (Personality Accentuating Conversational Engine) project.
"""

import json
import logging
import os
import tiktoken
import copy
from datetime import datetime
from typing import Dict, Any, Optional, List
from mem0 import Memory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.pace.config.constants import mem0_config, conversation_settings
from src.pace.utils.message_conversion import (
    convert_dict_to_messages,
)

# Configure logging for this module
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages Mem0 instance initialization and provides methods for conversational memory operations.

    This class encapsulates the Mem0 memory system, handling initialization,
    error management, and providing a clean interface for memory operations.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        user_name: str = "default_user",
        persona_name: str = "default_persona",
    ):
        """
        Initialize the MemoryManager with Mem0 configuration.

        Args:
            config: Configuration dictionary for Mem0. If None, uses default from config.py
            user_name: The name of the user.
            persona_name: The name of the persona.

        Raises:
            Exception: If Mem0 initialization fails
        """
        self.user_id = user_name  # Mem0 uses user_id to identify the user
        self.persona_name = persona_name
        self.mem0_instance: Optional[Memory] = None

        # Use provided config or fall back to imported mem0_config, making a deep copy
        base_config = (
            copy.deepcopy(config) if config is not None else copy.deepcopy(mem0_config)
        )

        # Dynamically set the collection name using the template from config
        if "vector_store" in base_config and "config" in base_config["vector_store"]:
            collection_template = base_config["vector_store"]["config"]["collection_name"]
            collection_name = collection_template.format(
                user_name=self.user_id,
                persona_name=self.persona_name
            )
            base_config["vector_store"]["config"]["collection_name"] = collection_name

        self.config = base_config

        # Generate persona-specific file paths for conversation logs
        self._setup_persona_file_paths()

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
            logger.warning("Could not initialize tokenizer, using character estimation")

        # Initialize Mem0 instance
        self._initialize_mem0()

    def _setup_persona_file_paths(self):
        """
        Setup persona-specific file paths for conversation logs and backups.
        """
        # Generate persona-specific conversation log file path
        self.conversation_log_file = conversation_settings["persona_log_format"].format(
            user_name=self.user_id, persona_name=self.persona_name
        )

        # Generate persona-specific backup format
        self.backup_format = conversation_settings["backup_format"]

        logger.info(f"Conversation log file set to: {self.conversation_log_file}")

    def _initialize_mem0(self) -> None:
        """
        Initialize the Mem0 instance with error handling and logging.

        This method attempts to create a Mem0 Memory instance using the
        provided configuration. It includes comprehensive error handling
        to catch and log any initialization issues.
        """
        try:
            logger.info(
                f"Initializing Mem0 with config for user: {self.user_id}, persona: {self.persona_name}"
            )
            self.mem0_instance = Memory.from_config(self.config)
            logger.info("Mem0 instance successfully initialized")

        except Exception as e:
            error_msg = f"Failed to initialize Mem0 instance: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def add_conversation_turn(
        self, messages: List[Dict[str, str]], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a conversation turn to Mem0 memory.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Format: [{'role': 'system', 'content': '...'},
                             {'role': 'user', 'content': '...'},
                             {'role': 'assistant', 'content': '...'}]
            session_id: Optional session identifier for organizing conversations

        Returns:
            Dictionary containing the result from Mem0's add operation

        Raises:
            Exception: If Mem0 instance is not initialized or add operation fails
        """
        if self.mem0_instance is None:
            error_msg = "Mem0 instance not initialized. Cannot add conversation turn."
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            logger.info(
                f"Adding conversation turn for user {self.user_id}, persona: {self.persona_name}, session: {session_id}"
            )
            logger.debug(f"Messages to add: {len(messages)} messages")
            # Call Mem0's add method with user_id
            # Note: session_id may be used in metadata or future Mem0 versions
            result = self.mem0_instance.add(messages, user_id=self.user_id)

            logger.info("Conversation turn successfully added to memory")
            logger.debug(f"Add operation result: {result}")

            return result

        except Exception as e:
            error_msg = f"Failed to add conversation turn to memory: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def search_memories(
        self, query_text: str, session_id: Optional[str] = None, limit: int = 3
    ) -> Dict[str, Any]:
        """
        Search for relevant memories using a query string.

        Args:
            query_text: Text to search for in stored memories
            session_id: Optional session identifier to limit search scope
            limit: Maximum number of memories to return (default: 3)

        Returns:
            Dictionary containing search results with 'memories' and 'relations' keys

        Raises:
            Exception: If Mem0 instance is not initialized or search operation fails
        """
        if self.mem0_instance is None:
            error_msg = "Mem0 instance not initialized. Cannot search memories."
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            logger.info(
                f"Searching memories for user {self.user_id}, persona: {self.persona_name}, session: {session_id}"
            )
            logger.debug(f"Query: '{query_text}', limit: {limit}")
            # Call Mem0's search method
            # Note: session_id may be used in metadata or future Mem0 versions
            search_results = self.mem0_instance.search(
                query_text, user_id=self.user_id, limit=limit
            )

            logger.info(f"Memory search completed. Found {len(search_results)} results")
            logger.debug(f"Search results: {search_results}")

            return search_results

        except Exception as e:
            error_msg = f"Failed to search memories: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def reset_memories(self) -> None:
        """
        Reset/clear all stored memories for this user.

        This method clears all memories associated with the current user_id
        from the Mem0 storage system.

        Raises:
            Exception: If Mem0 instance is not initialized or reset operation fails
        """
        if self.mem0_instance is None:
            error_msg = "Mem0 instance not initialized. Cannot reset memories."
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            logger.info(
                f"Resetting all memories for user {self.user_id}, persona: {self.persona_name}"
            )

            # Use Mem0's reset method to clear all memories
            self.mem0_instance.reset()

            logger.info("All memories successfully reset")

        except Exception as e:
            error_msg = f"Failed to reset memories: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_conversation_log_path(self) -> str:
        """
        Get the path to the persona-specific conversation log file.

        Returns:
            Path to the conversation log file for this user-persona combination
        """
        return self.conversation_log_file

    def get_user_id(self) -> str:
        """
        Get the current user ID.

        Returns:
            Current user identifier string
        """
        return self.user_id

    def is_initialized(self) -> bool:
        """
        Check if the Mem0 instance is properly initialized.

        Returns:
            True if Mem0 instance is initialized, False otherwise
        """
        return self.mem0_instance is not None

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: rough estimate of 4 characters per token
        return len(text) // 4

    def _count_message_tokens(self, message: BaseMessage) -> int:
        """
        Count tokens in a message, including role overhead.

        Args:
            message: The message to count tokens for

        Returns:
            Number of tokens including role formatting
        """
        # Add some overhead for role formatting (varies by model but typically 3-4 tokens per message)
        role_overhead = 4
        content = message.content
        # Handle both string and list content types
        if isinstance(content, str):
            content_tokens = self._count_tokens(content)
        else:
            # For complex content, convert to string representation
            content_tokens = self._count_tokens(str(content))
        return content_tokens + role_overhead

    def _count_messages_tokens(self, messages: List[BaseMessage]) -> int:
        """
        Count total tokens in a list of messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Total number of tokens
        """
        return sum(self._count_message_tokens(msg) for msg in messages)

    def backup_main_conversation_log(self, filepath: Optional[str] = None) -> str:
        """
        Create a timestamped backup of the existing conversation log.

        Args:
            filepath: Optional path to the main conversation log. If None, uses persona-specific path.

        Returns:
            Path to the backup file created
        """
        if filepath is None:
            filepath = self.conversation_log_file

        if not os.path.exists(filepath):
            logger.info(f"No existing conversation log found at {filepath}")
            return ""

        # Create timestamped backup filename using persona-specific format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = self.backup_format.format(
            user_name=self.user_id, persona_name=self.persona_name, timestamp=timestamp
        )

        try:
            with open(filepath, "r", encoding="utf-8") as source:
                os.makedirs(os.path.dirname(backup_filename), exist_ok=True)
                with open(backup_filename, "w", encoding="utf-8") as backup:
                    backup.write(source.read())

            logger.info(f"Conversation log backed up to: {backup_filename}")
            return backup_filename

        except Exception as e:
            error_msg = f"Failed to backup conversation log: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def load_main_conversation_log(
        self, filepath: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load and return the entire conversation history from the JSON file.

        Args:
            filepath: Optional path to the conversation log file. If None, uses persona-specific path.

        Returns:
            List of conversation turns with metadata
        """
        if filepath is None:
            filepath = self.conversation_log_file

        if not os.path.exists(filepath):
            logger.info(
                f"No conversation log found at {filepath}, returning empty history"
            )
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                conversations = json.load(f)

            logger.info(
                f"Loaded {len(conversations)} conversation turns from {filepath}"
            )
            return conversations

        except Exception as e:
            error_msg = f"Failed to load conversation log: {str(e)}"
            logger.error(error_msg)
            return []  # Return empty list instead of raising exception

    def append_to_main_conversation_log(
        self,
        current_user_input: str,
        final_response: str,
        filepath: Optional[str] = None,
    ) -> None:
        """
        Append a new conversation turn to the JSON log with timestamps.

        Args:
            current_user_input: The user's input message
            final_response: The persona's response message
            filepath: Optional path to the conversation log file. If None, uses persona-specific path.
        """
        if filepath is None:
            filepath = self.conversation_log_file

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Load existing conversations
        conversations = self.load_main_conversation_log(filepath)

        # Create new conversation entry
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": current_user_input,
            "final_response": final_response,
            "user_id": self.user_id,
            "persona_name": self.persona_name,
        }

        conversations.append(new_entry)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)

            logger.info(f"Appended conversation turn to {filepath}")

        except Exception as e:
            error_msg = f"Failed to append to conversation log: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def get_pruned_history_for_prompt(
        self, full_history: List[Dict[str, Any]], max_tokens: int
    ) -> List[BaseMessage]:
        """
        Get recent conversation turns as LangChain message objects that fit within token limit.

        Maximizes the number of recent turns included while staying within the limit.

        Args:
            full_history: Complete conversation history list
            max_tokens: Maximum tokens allowed for the message list

        Returns:
            List of BaseMessage objects representing the conversation history
        """
        if not full_history:
            return []

        # Start from the most recent conversations and work backwards
        selected_messages: List[BaseMessage] = []
        current_tokens = 0

        for conversation in reversed(full_history):
            user_input = conversation.get("user_input", "")
            final_response = conversation.get("final_response", "")

            # Create message objects
            user_message = HumanMessage(content=user_input)
            ai_message = AIMessage(content=final_response)

            # Calculate tokens for this turn
            turn_tokens = self._count_message_tokens(
                user_message
            ) + self._count_message_tokens(ai_message)

            # Check if we can include this turn
            if current_tokens + turn_tokens <= max_tokens:
                selected_messages.insert(
                    0, ai_message
                )  # Insert at beginning to maintain order
                selected_messages.insert(0, user_message)
                current_tokens += turn_tokens
            else:
                break

        if selected_messages:
            logger.info(
                f"Pruned conversation history: {len(selected_messages) // 2} turns, {current_tokens} tokens"
            )
            return selected_messages
        else:
            logger.info("No conversation history could fit within token limit")
            return []

    def get_full_conversation_as_messages(
        self, filepath: Optional[str] = None
    ) -> List[BaseMessage]:
        """
        Load the full conversation history as LangChain message objects.

        Args:
            filepath: Optional path to the conversation log file. If None, uses persona-specific path.

        Returns:
            List of BaseMessage objects representing the full conversation
        """
        conversations = self.load_main_conversation_log(filepath)
        return convert_dict_to_messages(conversations)


# Configure logging for the module
def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration for the memory manager module.

    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# Initialize logging when module is imported
setup_logging()
