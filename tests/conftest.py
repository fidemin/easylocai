import asyncio
import logging
import sys
from contextlib import AsyncExitStack

import pytest
from ollama import AsyncClient

from easylocai.core.tool_manager import ToolManager
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine


@pytest.fixture(autouse=True)
def setup_easylocai_logging():
    """Set up DEBUG logging for easylocai module during tests."""
    logger = _setup_easylocai_debug_logging()

    yield

    # Clean up after test
    logger.handlers.clear()


def _setup_easylocai_debug_logging():
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
    return logger


@pytest.fixture
def ollama_client():
    return AsyncClient(host="http://localhost:11434")


@pytest.fixture
def search_engine():
    return AdvancedSearchEngine()


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
async def tool_manager(search_engine, mcp_servers_config):
    # anyio cancel scopes cannot be exited from a different task than they were
    # entered in. pytest-asyncio runs fixture teardown in a separate task, so we
    # keep the entire MCP lifecycle (setup + cleanup) inside one background task
    # and signal across the task boundary with events.
    tm = ToolManager(search_engine, mpc_servers=mcp_servers_config)
    ready: asyncio.Event = asyncio.Event()
    done: asyncio.Event = asyncio.Event()
    errors: list = []

    async def _lifecycle():
        try:
            async with AsyncExitStack() as stack:
                await tm.initialize(stack)
                ready.set()
                await done.wait()
        except Exception as exc:
            errors.append(exc)
            ready.set()

    task = asyncio.ensure_future(_lifecycle())
    await ready.wait()

    if errors:
        raise errors[0]

    yield tm

    done.set()
    await task
