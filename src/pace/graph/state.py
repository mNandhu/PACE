"""
State Management for PACE LangGraph Application

This module defines the state schema for the LangGraph workflow,
managing data flow between nodes in the PACE conversation pipeline.
"""

from typing import Dict, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage


class PaceState(TypedDict):
    """
    State schema for the PACE LangGraph application.

    This TypedDict defines all the data that flows between nodes
    in the conversation processing pipeline.
    """

    # Input data
    current_user_input: str
    session_id: Optional[str]

    # Context identification results
    distilled_context_summary: str
    requires_memory_lookup: bool
    memory_search_query: Optional[str]

    # LLM response
    final_response: str

    # Graph-side message history for tool context
    messages: list[AnyMessage]

    # Metadata and tracking
    processing_metadata: Dict[str, Any]
