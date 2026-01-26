import logging
import sys
from contextlib import AsyncExitStack

import chromadb
import pytest
from ollama import AsyncClient

from easylocai.core.tool_manager import ToolManager


@pytest.fixture(autouse=True)
def setup_easylocai_logging():
    """Set up DEBUG logging for easylocai module during tests."""
    logger = logging.getLogger("easylocai")
    logger.setLevel(logging.DEBUG)

    # Add console handler to stdout if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    yield

    # Clean up after test
    logger.handlers.clear()


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
