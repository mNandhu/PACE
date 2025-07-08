from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["/mnt/g/Projects/SYCE/scripts/run_mcp_server.py"],
            "transport": "stdio",
        },
    }  # type: ignore
)
