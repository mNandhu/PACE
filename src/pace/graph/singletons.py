"""
Singleton instances for shared resources in the PACE graph.

This module provides global instances of MemoryManager and LLMWrapper
to avoid creating new instances in each node.
"""

import logging
from typing import Optional
from src.pace.memory.memory_manager import MemoryManager
from src.syce.llm.llm_wrapper import LLMWrapper
from src.pace.config.constants import (
    llm_configs,
    token_limits,
    app_settings,
)

logger = logging.getLogger(__name__)

# Global singleton instances
_memory_manager: Optional[MemoryManager] = None
# _foundational_llm: Optional[LLMWrapper] = None


def get_memory_manager() -> MemoryManager:
    """
    Get or create the global MemoryManager instance.

    Returns:
        Global MemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        logger.info("Initializing global MemoryManager instance")
        _memory_manager = MemoryManager(user_id=app_settings["user_id"])
    return _memory_manager


def get_foundational_llm() -> LLMWrapper:
    """
    Get or create the global foundational LLMWrapper instance.

    Returns:
        Global LLMWrapper instance for foundational Sumire LLM
    """
    global _foundational_llm
    if _foundational_llm is None:
        logger.info("Initializing global foundational LLM instance")
        foundational_config = llm_configs["foundational_sumire_llm"]
        max_tokens = token_limits["foundational_llm_max_prompt_tokens"]
        _foundational_llm = LLMWrapper(foundational_config, max_tokens)
    return _foundational_llm


def reset_singletons():
    """
    Reset all singleton instances (useful for testing or configuration changes).
    """
    global _memory_manager, _foundational_llm
    logger.info("Resetting singleton instances")
    _memory_manager = None
    _foundational_llm = None
