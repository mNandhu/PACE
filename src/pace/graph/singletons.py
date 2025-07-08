"""
Singleton instances for shared resources in the PACE graph.

This module provides global instances of MemoryManager and LLMWrapper
to avoid creating new instances in each node. These singletons are
initialized per-session with persona-specific data.
"""

import logging
from typing import Optional
from src.pace.memory.memory_manager import MemoryManager
from src.pace.llm.llm_wrapper import LLMWrapper
from src.pace.config.constants import (
    llm_configs,
    token_limits,
)

logger = logging.getLogger(__name__)

# Global singleton instances
_memory_manager: Optional[MemoryManager] = None
_foundational_llm: Optional[LLMWrapper] = None


def initialize_singletons(
    user_name: str, persona_name: str, mem0_config: Optional[dict] = None
) -> None:
    """
    Initialize singleton instances with persona-specific data.

    This should be called once per session from the main CLI after
    persona selection is complete.

    Args:
        user_name: The name of the user
        persona_name: The name of the selected persona
        mem0_config: Optional Mem0 configuration override
    """
    global _memory_manager, _foundational_llm

    logger.info(
        f"Initializing singletons for user '{user_name}' with persona '{persona_name}'"
    )

    # Reset existing instances
    _memory_manager = None
    _foundational_llm = None

    # Initialize MemoryManager with persona data
    if mem0_config is not None:
        _memory_manager = MemoryManager(
            config=mem0_config, user_name=user_name, persona_name=persona_name
        )
    else:
        # Import here to avoid circular import
        from src.pace.config.constants import mem0_config as default_config

        _memory_manager = MemoryManager(
            config=default_config, user_name=user_name, persona_name=persona_name
        )

    # Initialize LLM
    foundational_config = llm_configs["foundational_llm"]
    max_tokens = token_limits["foundational_llm_max_prompt_tokens"]
    _foundational_llm = LLMWrapper(foundational_config, max_tokens)

    logger.info("Singleton instances initialized successfully")


def get_memory_manager() -> MemoryManager:
    """
    Get the global MemoryManager instance.

    The instance must be initialized with initialize_singletons() first.

    Returns:
        Global MemoryManager instance

    Raises:
        RuntimeError: If singletons haven't been initialized
    """
    global _memory_manager
    if _memory_manager is None:
        raise RuntimeError(
            "MemoryManager singleton not initialized. Call initialize_singletons() first."
        )
    return _memory_manager


def get_foundational_llm() -> LLMWrapper:
    """
    Get the global foundational LLMWrapper instance.

    The instance must be initialized with initialize_singletons() first.

    Returns:
        Global LLMWrapper instance for foundational LLM

    Raises:
        RuntimeError: If singletons haven't been initialized
    """
    global _foundational_llm
    if _foundational_llm is None:
        raise RuntimeError(
            "Foundational LLM singleton not initialized. Call initialize_singletons() first."
        )
    return _foundational_llm


def reset_singletons():
    """
    Reset all singleton instances (useful for testing or configuration changes).
    """
    global _memory_manager, _foundational_llm
    logger.info("Resetting singleton instances")
    _memory_manager = None
    _foundational_llm = None


def are_singletons_initialized() -> bool:
    """
    Check if singleton instances are initialized.

    Returns:
        True if both singletons are initialized, False otherwise
    """
    return _memory_manager is not None and _foundational_llm is not None
