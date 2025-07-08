"""
LangGraph Application for PACE

This module defines and compiles the LangGraph application that orchestrates
the PACE conversation processing workflow.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.pace.graph.state import PaceState
from src.pace.graph.nodes import (
    start_node,
    identify_context_node,
    foundational_llm_node,
    update_memory_node,
)

from .tools import tools

logger = logging.getLogger(__name__)


def create_pace_graph() -> StateGraph:
    """
    Create and configure the PACE LangGraph application.

    Defines the workflow: START -> identify_context -> foundational_llm -> update_memory -> END

    Returns:
        Configured StateGraph ready for compilation
    """
    logger.info("Creating PACE LangGraph application")

    # Create the graph with our state schema
    graph = StateGraph(PaceState)  # Add nodes to the graph
    graph.add_node("start", start_node)
    graph.add_node("identify_context", identify_context_node)
    graph.add_node("foundational_llm", foundational_llm_node)
    graph.add_node("update_memory", update_memory_node)

    ###
    # Tools
    ###
    graph.add_node("tools", ToolNode(tools))

    # Define the edges (workflow)
    graph.add_edge(START, "start")
    graph.add_edge("start", "identify_context")
    graph.add_edge("identify_context", "foundational_llm")

    # Add conditional edges for tool calling
    graph.add_conditional_edges(
        "foundational_llm",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "update_memory",
        },
    )
    graph.add_edge("tools", "foundational_llm")
    graph.add_edge("update_memory", END)

    logger.info("PACE graph structure defined")
    return graph


def compile_pace_application():
    """
    Compile the PACE LangGraph application for execution.

    Returns:
        Compiled LangGraph application ready for invocation
    """
    logger.info("Compiling PACE LangGraph application")

    try:
        # Create and compile the graph
        graph = create_pace_graph()
        compiled_app = graph.compile()

        logger.info("PACE LangGraph application compiled successfully")
        return compiled_app

    except Exception as e:
        logger.error(f"Failed to compile PACE application: {str(e)}")
        raise Exception(f"PACE compilation failed: {str(e)}") from e


# Create a ready-to-use instance
try:
    pace_app = compile_pace_application()
    logger.info("PACE application is ready for use")
except Exception as e:
    logger.error(f"Failed to initialize PACE application: {e}")
    pace_app = None

if __name__ == "__main__":
    graph = compile_pace_application()
    print(graph.get_graph().draw_mermaid())
