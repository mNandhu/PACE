"""
LangGraph Nodes for PACE Application

This module contains the node functions that form the core workflow
of the PACE conversation processing pipeline.
"""

import logging
import uuid
import time

from typing import Dict, Any
from src.pace.config.constants import token_limits, graph_logic_settings
from .tools import tools
from .utils import (
    process_memory_results,
    build_conversation_messages,
)
from .singletons import get_memory_manager, get_foundational_llm

logger = logging.getLogger(__name__)


def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point node for the PACE conversation pipeline.

    Logs the start of a new interaction and initializes processing metadata.

    Args:
        state: Current graph state

    Returns:
        Updated state with initialized metadata
    """
    node_start_time = time.time()
    logger.info("Starting PACE conversation processing pipeline")

    # Initialize processing metadata
    state["processing_metadata"] = {
        "pipeline_id": str(uuid.uuid4()),
        "start_time": node_start_time,
        "nodes_executed": ["start_node"],
        "node_timings": {},  # Will store timing for each node
    }

    # Ensure required fields exist
    if "distilled_context_summary" not in state:
        state["distilled_context_summary"] = ""
    if "requires_memory_lookup" not in state:
        state["requires_memory_lookup"] = False
    if "memory_search_query" not in state:
        state["memory_search_query"] = None
    if "final_response" not in state:
        state["final_response"] = ""
    if "messages" not in state:
        state["messages"] = []

    # Record timing for this node
    node_execution_time = time.time() - node_start_time
    state["processing_metadata"]["node_timings"]["start_node"] = node_execution_time

    logger.debug(
        f"Pipeline initialized with ID: {state['processing_metadata']['pipeline_id']}"
    )
    return state


def identify_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses user input to for memory search and identifies relevant context using mem0's search and reranking capabilities.
    """
    node_start_time = time.time()
    logger.info("Executing context identification node")
    state["processing_metadata"]["nodes_executed"].append("identify_context_node")
    current_user_input = state.get("current_user_input", "")
    session_id = state.get("session_id")
    persona = state.get("persona")

    if not current_user_input:
        logger.warning("No user input provided to identify_context_node")
        state["distilled_context_summary"] = ""
        # Record timing even for early return
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["identify_context_node"] = node_execution_time
        return state

    try:
        # Use global memory manager instance
        memory_manager = get_memory_manager()

        search_query = current_user_input  # Default to user input

        # Log with persona context
        if persona:
            logger.info(
                f"Memory lookup for {persona.character_name} with query: {search_query}"
            )
        else:
            logger.info(f"Memory lookup with query: {search_query}")

        search_results = memory_manager.search_memories(
            query_text=search_query,
            session_id=session_id,
            limit=graph_logic_settings.get(
                "max_memories_retrieved", 5
            ),  # Configurable limit
        )

        # Process search results using enhanced utility
        selected_memories = process_memory_results(
            search_results,
            query=search_query,
            rerank=graph_logic_settings.get("use_reranking", False),
            include_relations=True,  # TODO: Set to True when graph relations are ready
        )

        if selected_memories:
            context_summary = "\n".join(selected_memories)
            if persona:
                logger.info(
                    f"Retrieved {len(selected_memories)} relevant memories for {persona.character_name}"
                )
            else:
                logger.info(f"Retrieved {len(selected_memories)} relevant memories")
        else:
            context_summary = ""
            logger.info("No usable memories found in search results")

        state["distilled_context_summary"] = context_summary
        
        # Record timing for this node
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["identify_context_node"] = node_execution_time
        
        return state

    except Exception as e:
        logger.error(f"Error in identify_context_node: {str(e)}")
        state["distilled_context_summary"] = ""
        # Record timing even for error case
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["identify_context_node"] = node_execution_time
        return state


def foundational_llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate Persona's response using the foundational LLM.

    Constructs a comprehensive prompt with persona directives, retrieved context,
    conversation history, and current user input using BaseMessage objects,
    then calls the Foundational LLM.

    Args:
        state: Current graph state with user input and context

    Returns:
        Updated state with Persona's response
    """
    node_start_time = time.time()
    logger.info("Executing foundational LLM node")

    # Track node execution
    state["processing_metadata"]["nodes_executed"].append("foundational_llm_node")

    # Get input data
    current_user_input = state.get("current_user_input", "")
    distilled_context_summary = state.get("distilled_context_summary", "")
    persona = state.get("persona")

    if not current_user_input:
        logger.warning("No user input provided to foundational_llm_node")

        # Use persona-aware fallback message
        if persona:
            fallback_msg = (
                f"I'm sorry, I didn't receive any input from you, {persona.user_name}."
            )
        else:
            fallback_msg = "I'm sorry, I didn't receive any input from you."

        state["final_response"] = fallback_msg
        # Record timing for early return
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["foundational_llm_node"] = node_execution_time
        return state

    if not persona:
        logger.error("No persona provided in state")
        state["final_response"] = (
            "I'm sorry, I'm having some configuration issues right now."
        )
        # Record timing for error case
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["foundational_llm_node"] = node_execution_time
        return state

    try:
        # Use global memory manager instance
        memory_manager = get_memory_manager()
        full_history = memory_manager.load_main_conversation_log()

        # Get pruned conversation history
        max_history_tokens = token_limits["conversation_history_max_tokens"]
        pruned_history = memory_manager.get_pruned_history_for_prompt(
            full_history, max_history_tokens
        )

        # Build comprehensive message list using enhanced utility with persona data
        response_instruction = f"Respond as {persona.character_name}, keeping in mind all the above context and maintaining consistency with past interactions."

        messages = build_conversation_messages(
            persona_directives=persona.core_persona_directives,
            context_summary=distilled_context_summary,
            history=pruned_history,
            current_input=current_user_input,
            response_instruction=response_instruction,
        )

        # Use global foundational LLM instance
        foundational_llm = get_foundational_llm()

        logger.debug(
            f"Sending {len(messages)} messages to Foundational LLM for {persona.character_name}"
        )

        # Get Persona's response using BaseMessage list
        response = foundational_llm.get_llm_response(messages, tools=tools)
        messages.append(response)  # Append response to messages

        # Update messages in state for tool context
        state["messages"] = messages

        # Update state with response - extract content for backward compatibility
        if hasattr(response, "content"):
            state["final_response"] = response.content # TODO: Remove this when BaseMessage is fully adopted
        else:
            state["final_response"] = str(response)

        logger.debug(
            f"Foundational LLM response generated for {persona.character_name} (length: {len(state['final_response'])} chars)"
        )
        
        # Record timing for this node
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["foundational_llm_node"] = node_execution_time
        
        return state

    except Exception as e:
        logger.error(f"Error in foundational_llm_node: {str(e)}")

        # Provide a persona-aware graceful fallback response
        if persona:
            fallback_msg = f"I'm sorry, {persona.user_name}, I'm having some technical difficulties right now. Please try again in a moment."
        else:
            fallback_msg = "I'm sorry, I'm having some technical difficulties right now. Please try again in a moment."

        state["final_response"] = fallback_msg
        # Record timing for error case
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["foundational_llm_node"] = node_execution_time
        return state


def update_memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update both conversation history and long-term semantic memory.

    Handles immediate conversation logging to JSON file and asynchronous
    storage to Mem0 for long-term semantic memory.

    Args:
        state: Current graph state with conversation data

    Returns:
        Updated state (memory operations are side effects)
    """
    node_start_time = time.time()
    logger.info("Executing memory update node")

    # Track node execution
    state["processing_metadata"]["nodes_executed"].append("update_memory_node")

    # Get conversation data
    current_user_input = state.get("current_user_input", "")
    final_response = state.get("final_response", "")
    session_id = state.get("session_id")
    persona = state.get("persona")

    if not current_user_input or not final_response:
        logger.warning("Incomplete conversation data for memory update")
        # Record timing for early return
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["update_memory_node"] = node_execution_time
        return state

    try:
        # Use global memory manager instance
        memory_manager = get_memory_manager()

        # 1. Immediate conversation history logging
        if persona:
            logger.info(f"Updating main conversation log for {persona.character_name}")
        else:
            logger.info("Updating main conversation log")

        memory_manager.append_to_main_conversation_log(
            current_user_input=current_user_input, final_response=final_response
        )

        # 2. Long-term semantic memory storage (Mem0)
        if persona:
            logger.info(
                f"Adding conversation to long-term semantic memory for {persona.character_name}"
            )
        else:
            logger.info("Adding conversation to long-term semantic memory")

        # Construct messages for Mem0
        messages = [
            {"role": "user", "content": current_user_input},
            {"role": "assistant", "content": final_response},
        ]

        # Add to Mem0
        mem0_result = memory_manager.add_conversation_turn(
            messages=messages, session_id=session_id
        )

        logger.info("Memory update completed successfully")
        logger.debug(f"Mem0 result: {mem0_result}")

        # Update metadata
        state["processing_metadata"]["memory_updated"] = True

        # Record timing for this node
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["update_memory_node"] = node_execution_time

        return state

    except Exception as e:
        logger.error(f"Error in update_memory_node: {str(e)}")
        # Don't fail the pipeline if memory update fails
        state["processing_metadata"]["memory_update_error"] = str(e)
        # Record timing for error case
        node_execution_time = time.time() - node_start_time
        state["processing_metadata"]["node_timings"]["update_memory_node"] = node_execution_time
        return state
