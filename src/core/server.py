import json
import os
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client, ClientSession


class Server:
    def __init__(
        self,
        name: str,
        params: StdioServerParameters,
    ):
        self.name = name
        self.params = params
        self.current_session = None

    async def initialize(self, async_stack: AsyncExitStack):
        stdio, write = await async_stack.enter_async_context(stdio_client(self.params))
        self.current_session = await async_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.current_session.initialize()

    async def list_tools(self):
        if self.current_session is None:
            raise RuntimeError("Server session is not initialized")
        tools = await self.current_session.list_tools()
        return tools

    def get_session(self):
        if self.current_session is None:
            raise RuntimeError("Server session is not initialized")
        return self.current_session


class ServerManager:
    def __init__(self):
        self._servers = {}

    def add_server(self, server: Server):
        self._servers[server.name] = server

    def get_server(self, name: str) -> Server | None:
        return self._servers.get(name)

    def list_servers(self):
        return list(self._servers.values())

    @staticmethod
    def from_json_config_file(config_path: str):
        server_manager = ServerManager()
        with open(config_path) as f:
            servers = json.load(f)["mcpServers"]

            server_dict = {}
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
