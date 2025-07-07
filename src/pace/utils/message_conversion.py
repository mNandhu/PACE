from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

import logging

logger = logging.getLogger(__name__)


def convert_messages_to_dict_format(
    messages: List[BaseMessage],
) -> List[Dict[str, str]]:
    """
    Convert LangChain BaseMessage objects to dictionary format for Mem0.

    Args:
        messages: List of BaseMessage objects

    Returns:
        List of dictionaries with 'role' and 'content' keys for Mem0
    """
    converted_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            logger.warning(
                f"Unknown message type: {type(message)}. Defaulting to 'user'"
            )
            role = "user"  # fallback

        converted_messages.append(
            {
                "role": role,
                "content": str(message.content) if message.content else "",
            }
        )

    return converted_messages


def convert_dict_to_messages(conversations: List[Dict[str, Any]]) -> List[BaseMessage]:
    """
    Convert conversation history dictionaries to LangChain message objects.

    Args:
        conversations: List of conversation dictionaries with user_input/sumire_response

    Returns:
        List of BaseMessage objects
    """
    messages: List[BaseMessage] = []

    for conversation in conversations:
        user_input = conversation.get("user_input", "")
        sumire_response = conversation.get("sumire_response", "")

        if user_input:
            messages.append(HumanMessage(content=user_input))
        if sumire_response:
            messages.append(AIMessage(content=sumire_response))

    return messages
