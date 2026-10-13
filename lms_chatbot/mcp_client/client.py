"""
Thin async wrapper around the MCP SDK's stdio client: launches
mcp_server.server as a subprocess and exposes list_tools/call_tool over
the MCP protocol, instead of importing the tool functions in-process.

This is what makes the tools "real" MCP tools rather than the old
mcp.py's plain Python function calls: they run in a separate process,
speak the standard protocol, and could be swapped for a remote server
(HTTP transport) later without changing anything above this layer.
"""

import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client


def default_server_params() -> StdioServerParameters:
    """Launch mcp_server.server with the *same* interpreter/venv this
    process is running under, rather than assuming `uv` or `python` on
    PATH resolve to the right environment."""
    # The MCP SDK starts the server with only a minimal default
    # environment; LMS_LOG_FILE (chat mode, see main.py) must be passed
    # on explicitly or the server's logs land on the chat console.
    env = None
    if os.getenv("LMS_LOG_FILE"):
        env = {**get_default_environment(), "LMS_LOG_FILE": os.environ["LMS_LOG_FILE"]}
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "lms_chatbot.mcp_server.server"],
        env=env,
    )


class MCPClient:
    def __init__(self, server_params: Optional[StdioServerParameters] = None):
        self._server_params = server_params or default_server_params()
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()

    async def connect(self) -> None:
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(self._server_params)
        )
        read_stream, write_stream = stdio_transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Not connected - call connect() first (or use `async with`).")
        return self._session

    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> types.CallToolResult:
        return await self.session().call_tool(tool_name, tool_input)

    async def cleanup(self) -> None:
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()


def tool_result_text(result: types.CallToolResult) -> str:
    """Flatten a CallToolResult's content blocks into plain text - our
    tools only ever return a single TextContent block."""
    parts = [block.text for block in result.content if isinstance(block, types.TextContent)]
    return "\n".join(parts)


def to_openai_tool_schema(tool: types.Tool) -> dict:
    """MCP's Tool -> the {"type": "function", "function": {...}} shape
    ChatOllama.bind_tools() (via langchain_core.convert_to_openai_tool)
    expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }
