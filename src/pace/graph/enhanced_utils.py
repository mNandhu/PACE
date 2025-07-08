"""
Enhanced utilities for SYCE Graph processing.

This module contains utility functions for flexible memory management,
context processing, and prompt construction.
"""

import logging
import requests
import os
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from ..config.worker_prompts import RERANKING_TASK_PROMPT
from ..config.constants import additional_endpoint_settings


logger = logging.getLogger(__name__)


def rerank_using_qwen(
    query: str, search_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rerank search results using Qwen model via HTTP API.

    Args:
        query: The original search query used to retrieve results
        search_results: List of search result dictionaries given by mem0 search

    Returns:
        List of reranked search results (sorted by score)
    """
    request_url = additional_endpoint_settings["reranking_endpoint"]

    # FIXME: DEBUG
    print(f"Before Reranking: {search_results}")

    if not search_results:
        return search_results

    try:
        # First check if the server is alive
        health_url = os.path.join(request_url, "health")
        health_response = requests.get(health_url, timeout=5)

        if health_response.status_code != 200:
            logger.warning(
                "Reranking server health check failed, returning original results"
            )
            return search_results

        logger.debug(
            f"Reranking server @ {request_url} is healthy, proceeding with reranking"
        )

    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Reranking server not available: {e}, returning original results"
        )
        return search_results

    try:
        # Extract document texts from search results
        documents = []
        for result in search_results:
            if isinstance(result, dict):
                doc_text = result.get(
                    "memory", result.get("text", result.get("content", ""))
                )
            else:
                doc_text = str(result)

            if doc_text:
                documents.append(doc_text)

        if not documents:
            logger.warning("No valid documents found in search results")
            return search_results

        # Prepare the scoring request
        scoring_url = os.path.join(request_url, "score_single")
        payload = {
            "query": query,
            "documents": documents,
            "task": RERANKING_TASK_PROMPT,
        }

        logger.info(f"Sending reranking request with {len(documents)} documents")
        response = requests.post(scoring_url, json=payload, timeout=30)
        response.raise_for_status()

        scoring_data = response.json()
        scores = scoring_data.get("scores", [])

        if len(scores) != len(search_results):
            logger.warning(
                f"Score count mismatch: {len(scores)} scores for {len(search_results)} results"
            )
            return search_results

        # Pair results with scores and sort by score (descending)
        scored_results = list(zip(search_results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Extract the reranked results
        reranked_results = [result for result, score in scored_results]

        logger.info(f"Successfully reranked {len(reranked_results)} results")
        print(f"After Reranking: {reranked_results}")  # FIXME: DEBUG
        return reranked_results

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during reranking request: {e}")
        return search_results
    except Exception as e:
        logger.error(f"Unexpected error during reranking: {e}")
        return search_results


def process_memory_results(
    search_results: Union[Dict[str, Any], List[Any]],
    include_relations: bool = False,
    rerank: bool = False,
    query: Optional[str] = None,
) -> List[str]:
    """
    Process memory search results into formatted context strings.

    Args:
        search_results: Results from memory search
        include_relations: Whether to include graph relations if available
        rerank: Whether to rerank results using Qwen model
        query: Original search query (required if rerank=True)

    Returns:
        List of formatted memory context strings
    """
    memories_to_process = []
    relations_to_process = []

    logger.info(f"{search_results=}")

    # Extract memories and relations
    if isinstance(search_results, dict):
        if "results" in search_results:
            memories_to_process = search_results["results"]
        if include_relations and "relations" in search_results:
            relations_to_process = search_results["relations"]
    elif isinstance(search_results, list):
        memories_to_process = search_results

    # Apply reranking if requested and we have memories to process
    if rerank and memories_to_process and query:
        logger.info("Applying reranking to memory results")
        memories_to_process = rerank_using_qwen(query, memories_to_process)
    elif rerank and not query:
        logger.warning("Reranking requested but no query provided, skipping reranking")

    selected_memories = []

    # Process memories
    for i, result in enumerate(memories_to_process):
        memory_text = ""
        if isinstance(result, dict):
            memory_text = result.get(
                "memory", result.get("text", result.get("content", ""))
            )
        elif isinstance(result, str):
            memory_text = result

        if memory_text:
            selected_memories.append(f"Relevant context {i + 1}: {memory_text}")

    logger.info(
        f"Relations: {relations_to_process} and Include_Relations: {include_relations}"
    )

    # Process relations if requested
    if include_relations and relations_to_process:
        for i, relation in enumerate(relations_to_process):
            if isinstance(relation, dict):
                relation_text = relation.get("relation", relation.get("content", ""))
                if relation_text:
                    selected_memories.append(
                        f"Related context {i + 1}: {relation_text}"
                    )

    return selected_memories


def build_conversation_messages(
    persona_directives: List[str],
    context_summary: str = "",
    history: Optional[Union[str, List[BaseMessage]]] = None,
    current_input: str = "",
    response_instruction: str = "",
) -> List[BaseMessage]:
    """
    Build a list of LangChain BaseMessage objects for LLM input.

    Args:
        persona_directives: List of persona directive strings
        context_summary: Retrieved context summary
        history: Conversation history (string or BaseMessage list)
        current_input: Current user input
        response_instruction: Final instruction for the model

    Returns:
        List of BaseMessage objects ready for LLM
    """
    messages = []

    # 1. System message with persona directives
    if persona_directives:
        persona_content = "Core Persona Directives:\n" + "\n".join(
            f"- {directive}" for directive in persona_directives
        )
        messages.append(SystemMessage(content=persona_content))

    # 2. Context summary as system message
    if context_summary:
        context_content = (
            f"Relevant Context from Past Conversations:\n{context_summary}"
        )
        messages.append(SystemMessage(content=context_content))

    # 3. Conversation history
    if history:
        if isinstance(history, str):
            messages.append(
                SystemMessage(content=f"Recent Conversation History:\n{history}")
            )
        elif isinstance(history, list):
            # Assume it's already BaseMessage objects
            messages.extend(history)

    # 4. Current user input
    if current_input:
        messages.append(HumanMessage(content=current_input))

    # 5. Response instruction
    if response_instruction:
        messages.append(SystemMessage(content=response_instruction))

    return messages
