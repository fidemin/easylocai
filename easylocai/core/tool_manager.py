import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path

from chromadb import ClientAPI
from mcp import StdioServerParameters, stdio_client, ClientSession
from mcp import Tool as McpTool

logger = logging.getLogger(__name__)


class Tool:
    def __init__(self, server_name: str, tool: McpTool):
        self._name = tool.name
        self._input_schema = tool.inputSchema
        self._description = tool.description
        self._server_name = server_name

    def __repr__(self):
        return f"Tool(server={self._server_name}, name={self._name})"

    @property
    def server_name(self):
        return self._server_name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def input_schema(self):
        return self._input_schema


class Server:
    def __init__(
        self,
        name: str,
        params: StdioServerParameters,
    ):
        self.name = name
        self.params = params
        self._current_session: ClientSession | None = None
        self._tools = None
        self._tools_dict = {}

    async def initialize(self, async_stack: AsyncExitStack):
        stdio_receive_stream, stdio_send_stream = await async_stack.enter_async_context(
            stdio_client(self.params)
        )
        self._current_session = await async_stack.enter_async_context(
            ClientSession(stdio_receive_stream, stdio_send_stream)
        )
        await self._current_session.initialize()

    async def list_tools(self) -> list[Tool]:
        self._ensure_initialized()
        if self._tools is not None:
            return self._tools

        tool_result: list[McpTool] = await self._current_session.list_tools()
        self._tools = [Tool(self.name, tool) for tool in tool_result.tools]
        self._tools_dict = {tool.name: tool for tool in self._tools}
        return self._tools

    def get_tool(self, tool_name: str) -> Tool | None:
        self._ensure_initialized()
        return self._tools_dict.get(tool_name)

    async def call_tool(self, tool_name: str, tool_args: dict):
        self._ensure_initialized()
        tool = self.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in server '{self.name}'")
        return await self._current_session.call_tool(tool.name, tool_args)

    def get_session(self):
        self._ensure_initialized()
        return self._current_session

    def _ensure_initialized(self):
        if self._current_session is None:
            raise RuntimeError("Server session is not initialized")


class ServerManager:
    def __init__(self):
        self._servers = {}

    def add_server(self, server: Server):
        self._servers[server.name] = server

    async def initialize_servers(self, async_stack: AsyncExitStack):
        for server in self.list_servers():
            await server.initialize(async_stack)

            tools = await server.list_tools()
            logger.info(
                f"Server Name: {server.name}, Available tools: {[tool.name for tool in tools]}"
            )

    def get_server(self, name: str) -> Server:
        server = self._servers.get(name)
        if server is None:
            raise ValueError(f"Server '{name}' not found")
        return server

    def list_servers(self) -> list[Server]:
        return list(self._servers.values())

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        tool_args: dict,
    ):
        server = self.get_server(server_name)
        return await server.call_tool(tool_name, tool_args)

    @staticmethod
    def from_json_config_file(config_path: Path):
        server_manager = ServerManager()
        with open(config_path) as f:
            servers = json.load(f)["mcpServers"]

            for server_name, config in servers.items():
                if "env" in config:
                    env = {}
                    for k, v in config["env"].items():
                        env[k] = os.path.expandvars(v)
                server_params = StdioServerParameters(
                    **config,
                )

                server = Server(server_name, server_params)
                server_manager.add_server(server)

        return server_manager


class ToolManager:
    def __init__(self, chromadb_client: ClientAPI, *, mpc_servers: dict):
        server_manager = ServerManager()

        for server_name, config in mpc_servers.items():
            if "env" in config:
                env = {}
                for k, v in config["env"].items():
                    env[k] = os.path.expandvars(v)
            server_params = StdioServerParameters(
                **config,
            )

            server = Server(server_name, server_params)
            server_manager.add_server(server)
        self._server_manager = server_manager
        self._tool_collection = chromadb_client.get_or_create_collection("tools")

    async def initialize(self, async_stack: AsyncExitStack):
        await self._server_manager.initialize_servers(async_stack)
        await self._initialize_tools()

    async def _initialize_tools(self):
        # initialize chromadb tool
        ids = []
        metadatas = []
        documents = []
        for server in self._server_manager.list_servers():
            for tool in await server.list_tools():
                id_ = f"{server.name}:{tool.name}"
                metadata = {
                    "server_name": server.name,
                    "tool_name": tool.name,
                }
                ids.append(id_)
                documents.append(tool.description)
                metadatas.append(metadata)

        if len(ids) == 0:
            logger.warning("No tools found to initialize in ToolManager.")
            return

        self._tool_collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

    def search_tools(self, queries: list[str], *, n_results: int) -> list[Tool]:
        search_result = self._tool_collection.query(
            query_texts=queries,
            n_results=n_results,
        )
        tools: list[Tool] = []

        for metadata_list in search_result["metadatas"]:
            for metadata in metadata_list:
                server_name = metadata["server_name"]
                tool_name = metadata["tool_name"]
                tool = self._server_manager.get_server(server_name).get_tool(tool_name)
                tools.append(tool)
        return tools
