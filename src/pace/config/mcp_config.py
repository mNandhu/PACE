from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            "args": ["/mnt/g/Projects/SYCE/scripts/run_mcp_server.py"],
            "transport": "stdio",
        },
        # "todoist": {
        #     "command": "node",
        #     "args": ["G:/Github/todoist-mcp/build/index.js"],
        #     "env": {"TODOIST_API_KEY": os.environ["TODOIST_API_KEY"]},
        #     "transport": "stdio",
        # }
    }  # type: ignore
)
