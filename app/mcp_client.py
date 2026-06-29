import os
import sys

from google.adk.tools.mcp_tool import McpToolset
from mcp import StdioServerParameters


def get_mcp_toolset() -> McpToolset:
    """Constructs and returns the McpToolset connected to the local MCP server.

    Uses the active python interpreter to run app/mcp_server.py.
    """
    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mcp_server.py"
    )

    # Launch local MCP server via stdio using the same python interpreter
    return McpToolset(
        connection_params=StdioServerParameters(
            command=sys.executable,
            args=[server_script],
        )
    )
