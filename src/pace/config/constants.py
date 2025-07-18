"""
Configuration Management for PACE

This module manages all configuration settings for the PACE system,
including LLM configurations, Mem0 settings, token limits, and persona directives.
"""

# TODO: Make Configuration Management easier to update during runtime
# TODO: Switch to TypedDicts for better type safety and validation
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Mem0 Configuration (for memory management)
# Template Support: Use {user_name} and {persona_name} placeholders in collection names
# These will be automatically replaced at runtime with actual user and persona values
mem0_config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "gemini/gemini-2.5-flash",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:f16",  # "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
            "ollama_base_url": "http://localhost:11435",
            "embedding_dims": 4096,  # 1024,
        },
    },
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "pace_{user_name}_{persona_name}",  # Template: {user_name} and {persona_name} are replaced at runtime
            "embedding_model_dims": 4096,  # 1024,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.environ.get("NEO4J_USER", "neo4j"),
            "password": os.environ.get("NEO4J_PASSWORD", "xxx"),
        },
    },
}

# PACE LLM Configurations (separate from Mem0's LLM usage)
llm_configs = {
    "foundational_llm": {
        "model": "gemini/gemini-2.5-flash",  # Main PACE LLM
        "temperature": 0.3,
        "max_tokens": 4000,
        "timeout": 60,
    }
}

# Token Management Settings
token_limits = {
    "foundational_llm_max_prompt_tokens": 32768,  # Max tokens for full prompt
    "conversation_history_max_tokens": 32768,  # Max tokens for conversation history
}


# Conversation History Settings
conversation_settings: dict[str, str] = {
    "main_log_filename": "Assistant/chats/conversation_log.json",  # Default/fallback TODO: Remove this backward compatibility
    "persona_log_format": "Assistant/chats/{user_name}_{persona_name}_conversation_log.json",  # Template: {user_name} and {persona_name} are replaced at runtime
    "backup_format": "Assistant/chats/backup/{user_name}_{persona_name}_conversation_backup_{timestamp}.json",  # Template: includes {timestamp} for backup files
    "max_history_turns_for_prompt": "20",  # Maximum conversation turns to include
}

# Persona Settings
persona_settings = {
    "personas_dir": "Assistant/characters",
}

# Application Settings
app_settings = {
    "user_id": "kiruthik_main_user",
    "default_session_prefix": "pace_session",
    "logging_level": "INFO",
    "log_file": "logs/pace_app.log",
    "verbose_log_file": "logs/pace_app_verbose.log",
    "debug_mode": False,
}

# Graph Logic Settings
graph_logic_settings = {
    "max_memories_retrieved": 10,  # Max memories to retrieve in a single search
    "use_reranking": False,  # Enable reranking for memory search results
}

additional_endpoint_settings = {
    "reranking_endpoint": "http://localhost:7001/",  # Qwen3 reranking endpoint
}


# LLM Retry Settings
llm_retry_settings = {
    "max_retries": 3,  # Maximum retry attempts for LLM calls
    "base_delay": 2.0,  # Base delay in seconds between retries
    "max_delay": 30.0,  # Maximum delay in seconds for exponential backoff
}
