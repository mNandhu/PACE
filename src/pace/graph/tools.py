import asyncio
from src.pace.config.mcp_config import client as mcp_client

import logging
import time

logger = logging.getLogger(__name__)


async def _get_tools_async():
    """Asynchronously get tools from the MCP client."""
    return await mcp_client.get_tools()


def _initialize_tools():
    """Initialize tools by running the async function in a new event loop."""
    return asyncio.run(_get_tools_async())


# Initialize tools once and make them available for import across your application
logger.info("Initializing tools from MCP client")
start_time = time.time()
tools = _initialize_tools()
logger.info(f"Tools initialized in {time.time() - start_time:.2f} seconds")
