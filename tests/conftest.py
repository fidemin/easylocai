import pytest
from contextlib import AsyncExitStack

import chromadb
from ollama import AsyncClient

from easylocai.core.tool_manager import ToolManager


@pytest.fixture
def ollama_client():
    return AsyncClient(host="http://localhost:11434")


@pytest.fixture
def chromadb_client():
    return chromadb.Client()


@pytest.fixture
def mcp_servers_config():
    return {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/tmp",
            ],
        }
    }


@pytest.fixture
async def tool_manager(chromadb_client, mcp_servers_config):
    tool_manager = ToolManager(chromadb_client, mpc_servers=mcp_servers_config)
    async with AsyncExitStack() as stack:
        await tool_manager.initialize(stack)
        yield tool_manager