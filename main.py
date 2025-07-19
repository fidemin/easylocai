import asyncio
import json
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client, ClientSession


async def main():

    with open("mcp_server_config.json") as f:
        config = json.load(f)["mcpServers"]["filesystem"]

    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"],
        env=None
    )

    stack = AsyncExitStack()
    async with stack:
        stdio, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        tools_response = await session.list_tools()
        tool_names = [tool.name for tool in tools_response.tools]

        print(f"tools: {tool_names}")


if __name__ == "__main__":
    asyncio.run(main())
