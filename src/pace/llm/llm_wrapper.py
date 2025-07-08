"""
LLM Wrapper for Project PACE

This module provides an abstracted interface for LLM interactions using LangChain's LiteLLM,
with failsafe context pruning and rate limiting capabilities.
"""

import logging
import time
import tiktoken
from typing import Dict, Any, Optional, List
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from .rate_limiting import with_rate_limit_handling
from src.pace.config.constants import llm_configs, token_limits

logger = logging.getLogger(__name__)


class LLMWrapper:
    """
    Abstracted LLM wrapper using LangChain's ChatLiteLLM for unified LLM access.

    Provides failsafe context pruning, rate limiting, and error handling
    for all LLM interactions in the PACE system with proper message-based conversation management.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        max_prompt_tokens: Optional[int] = None,
    ):
        """
        Initialize the LLM wrapper with configuration and token limits.

        Args:
            model_config: LiteLLM-compatible model configuration
            max_prompt_tokens: Maximum tokens allowed for prompts (failsafe limit)
        """
        # Use provided config or fall back to foundational LLM defaults
        self.model_config = (
            model_config
            if model_config is not None
            else llm_configs["foundational_llm"]
        )
        # Use provided prompt token limit or default
        self.max_prompt_tokens = (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else token_limits["foundational_llm_max_prompt_tokens"]
        )

        # Extract configuration parameters
        self.model = self.model_config.get(
            "model", llm_configs["foundational_llm"]["model"]
        )
        self.temperature = self.model_config.get(
            "temperature", llm_configs["foundational_llm"]["temperature"]
        )
        self.max_tokens = self.model_config.get(
            "max_tokens", llm_configs["foundational_llm"]["max_tokens"]
        )
        # Timeout may not exist in foundational config, default to 60
        self.timeout = self.model_config.get(
            "timeout", llm_configs["foundational_llm"].get("timeout", 60)
        )

        # Initialize LangChain ChatLiteLLM instance
        self.chat_model = ChatLiteLLM(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.timeout,
        )

        # Initialize tokenizer for token counting (fallback to cl100k_base if model-specific not available)
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model.split("/")[-1])
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"LLMWrapper initialized with model: {self.model}")
        logger.debug(
            f"Config: temp={self.temperature}, max_tokens={self.max_tokens}, max_prompt_tokens={self.max_prompt_tokens}"
        )

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using character estimate: {e}")
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

    def _prune_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Prune messages if they exceed the maximum token limit.

        Preserves system messages and removes older messages while keeping recent ones.

        Args:
            messages: List of messages to potentially prune

        Returns:
            Pruned list of messages that fits within token limits
        """
        total_tokens = self._count_messages_tokens(messages)

        if total_tokens <= self.max_prompt_tokens:
            logger.debug(
                f"Messages within limits: {total_tokens}/{self.max_prompt_tokens} tokens"
            )
            return messages

        logger.warning(
            f"Messages exceed limit: {total_tokens}/{self.max_prompt_tokens} tokens, pruning..."
        )

        # Make a mutable copy to work with
        pruned_messages_list = list(messages)
        # Use the total_tokens calculated at the beginning of the method
        current_processing_tokens = total_tokens

        while current_processing_tokens > self.max_prompt_tokens:
            if not pruned_messages_list:
                # This  state should ideally not be reached if initial total_tokens > 0
                # and max_prompt_tokens is non-negative. Breaking to be safe.
                break

            idx_to_remove = -1
            # Try to find the oldest (first in the list) non-system message to remove
            for i, msg in enumerate(pruned_messages_list):
                if not isinstance(msg, SystemMessage):
                    idx_to_remove = i
                    break

            if idx_to_remove != -1:
                # Found an oldest non-system message; remove it
                pruned_messages_list.pop(idx_to_remove)
            else:
                logger.warning(
                    "No non-system messages found to remove, pruning system messages..."
                )
                # All remaining messages are system messages, or the list became empty.
                # If messages remain, they must be system messages. Remove the oldest one.
                if pruned_messages_list:
                    pruned_messages_list.pop(
                        0
                    )  # Remove the oldest system message (at index 0)
                else:
                    # List is empty, so nothing more to prune.
                    # The loop condition (current_processing_tokens > self.max_prompt_tokens)
                    # should handle termination correctly once tokens are within limits or list is empty.
                    break

            # Recalculate tokens for the modified list
            current_processing_tokens = self._count_messages_tokens(
                pruned_messages_list
            )

            # The pruned list is now final
        final_messages = pruned_messages_list

        final_tokens = self._count_messages_tokens(final_messages)
        logger.info(f"Messages pruned from {total_tokens} to {final_tokens} tokens")
        return final_messages

    @with_rate_limit_handling(max_retries=3, base_delay=2.0, max_delay=30.0)
    def get_llm_response(
        self, messages: List[BaseMessage], tools: Optional[List[Any]] = None
    ) -> AIMessage:
        """
        Get a response from the LLM with failsafe message pruning.

        Args:
            messages: List of BaseMessage objects representing the conversation

        Returns:
            AIMessage: The LLM's response as a AIMessage

        Raises:
            Exception: If the LLM call fails after retries
        """
        try:
            # Apply failsafe message pruning
            pruned_messages = self._prune_messages(messages)

            logger.info(f"Requesting LLM response (model: {self.model})")
            logger.debug(
                f"Message count: {len(pruned_messages)}, total tokens: {self._count_messages_tokens(pruned_messages)}"
            )

            # If tools are provided, bind them to the chat model
            if tools is not None:
                logger.debug(f"Binding tools: {tools} to LLM call")
                _model = self.chat_model.bind_tools(tools)
            else:
                _model = self.chat_model

            # Call LangChain ChatLiteLLM
            start_time = time.time()
            response = _model.invoke(pruned_messages)

            call_duration = time.time() - start_time

            # TODO: Fix error handling. Hint, empty response and wrong type are combined as one error and a warning
            if response and isinstance(response, AIMessage):
                logger.info(
                    f"LLM response received in {call_duration:.2f}s ({len(response.content)} chars)"
                )
                logger.debug(f"Response preview: {response.content[:200]}...")
                return response
            else:
                # Handle unexpected response types
                logger.warning(f"Unexpected response type from LLM: {type(response)}")
                raise Exception("LLM returned empty response content")

        except Exception as e:
            error_msg = f"LLM call failed for model {self.model}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary containing model configuration details
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "timeout": self.timeout,
        }

    def get_streaming_response(
        self, messages: List[BaseMessage]
    ):  # FIXME: This doesn't work for this version of the langchain-litellm package for some reason
        """
        Get a streaming response from the LLM.

        Yields chunks of the LLM response as they arrive.
        """
        # Apply failsafe message pruning
        pruned_messages = self._prune_messages(messages)
        logger.info(f"Requesting streaming LLM response (model: {self.model})")
        logger.debug(
            f"Message count: {len(pruned_messages)}, total tokens: {self._count_messages_tokens(pruned_messages)}"
        )

        # Attempt streaming; fall back to non-streaming on AttributeError (e.g., dict.role missing)
        try:
            stream_iter = self.chat_model.stream(pruned_messages)
        except AttributeError as ae:
            logger.warning(
                f"Streaming not supported for model {self.model}, falling back to non-streaming: {ae}\n{pruned_messages=}"
            )
            full_resp = self.get_llm_response(messages)
            for char in full_resp:
                yield char
            return

        try:
            # Iterate over streaming chunks
            for chunk in stream_iter:
                # Handle LangChain AIMessageChunk objects
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                # Handle dict responses with content
                elif isinstance(chunk, dict) and "content" in chunk:
                    yield chunk["content"]
                # Handle OpenAI-style streaming response
                elif isinstance(chunk, dict) and "choices" in chunk:
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                # Handle direct string responses
                elif isinstance(chunk, str):
                    yield chunk
                else:
                    logger.warning(f"Unknown chunk format: {type(chunk)} - {chunk}")
        except AttributeError as ae:
            # Fall back to non-streaming if iteration hits a dict.role error
            logger.warning(
                f"Streaming iteration failed for model {self.model}, falling back to non-streaming: {ae} \n{pruned_messages=}"
            )
            full_resp = self.get_llm_response(messages)
            for char in full_resp:
                yield char
            return
        except Exception as e:
            error_msg = f"Streaming LLM call failed for model {self.model}: {e}"
            logger.error(error_msg, exc_info=True)
            raise

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the LLM configuration and refresh parameters.
        """
        self.model_config.update(new_config)
        self.model = self.model_config.get("model", self.model)
        self.temperature = self.model_config.get("temperature", self.temperature)
        self.max_tokens = self.model_config.get("max_tokens", self.max_tokens)

        # Preserve or update timeout and prompt tokens if present
        if "timeout" in new_config:
            self.timeout = new_config.get("timeout", self.timeout)
        if "max_prompt_tokens" in new_config:
            self.max_prompt_tokens = new_config.get(
                "max_prompt_tokens", self.max_prompt_tokens
            )

        # Recreate the chat model with new configuration
        self.chat_model = ChatLiteLLM(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.timeout,
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # Simple example: chat loop with message-based conversation and streaming option
    messages: List[BaseMessage] = []
    wrapper = LLMWrapper()

    # Optionally set a custom max prompt token limit
    limit_input = input(f"Max prompt tokens [{wrapper.max_prompt_tokens}]: ").strip()
    if limit_input:
        try:
            wrapper.max_prompt_tokens = int(limit_input)
        except ValueError:
            print("Invalid input, using default token limit.")

    # Optional system message
    system_input = input("System message (optional): ").strip()
    if system_input:
        messages.append(SystemMessage(content=system_input))

    stream_choice = input("Stream response? (y/N): ").strip().lower() == "y"
    print("Starting demo. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Add user message to conversation
        messages.append(HumanMessage(content=user_input))

        # Display token counts before and after pruning
        raw_tokens = wrapper._count_messages_tokens(messages)
        pruned_messages = wrapper._prune_messages(messages)
        pruned_tokens = wrapper._count_messages_tokens(pruned_messages)
        print(
            f"Raw tokens: {raw_tokens}, pruned tokens: {pruned_tokens}/{wrapper.max_prompt_tokens}"
        )

        try:
            if stream_choice:
                print("Assistant: ", end="", flush=True)
                response_str = ""
                for chunk in wrapper.get_streaming_response(messages):
                    response_str += chunk  # type: ignore
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            else:
                response_str = wrapper.get_llm_response(messages)
                # Check if the response is an AIMessage
                if isinstance(response_str, AIMessage):
                    response_str = response_str.content
                print("Assistant:", response_str)

            # Add assistant response to conversation
            messages.append(AIMessage(content=response_str))

        except Exception as e:
            print(f"Error: {e}")
            # Don't add the user message to history if there was an error
            messages.pop()  # Remove the user message that caused the error
