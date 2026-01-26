from contextlib import AsyncExitStack

import chromadb
import pytest
from ollama import AsyncClient

from easylocai.agents.single_task_agent_v2 import (
    SingleTaskAgentV2,
    SingleTaskAgentV2Input,
    SingleTaskAgentV2Output,
)
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


class TestSingleTaskAgentV2:
    @pytest.mark.asyncio
    async def test_tool_task_read_file(
        self, ollama_client, chromadb_client, mcp_servers_config
    ):
        """Test SingleTaskAgentV2 with a file read task."""
        async with AsyncExitStack() as stack:
            tool_manager = ToolManager(chromadb_client, mpc_servers=mcp_servers_config)
            await tool_manager.initialize(stack)

            agent = SingleTaskAgentV2(
                client=ollama_client,
                tool_manager=tool_manager,
            )

            input_ = SingleTaskAgentV2Input(
                original_user_query="List files in current directory",
                task="List files in current directory",
                previous_task_results=[],
                user_context=None,
            )

            output: SingleTaskAgentV2Output = await agent.run(input_)

            assert output.result is not None
            assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_reasoning_task(
        self, ollama_client, chromadb_client, mcp_servers_config
    ):
        """Test SingleTaskAgentV2 with a reasoning task."""
        async with AsyncExitStack() as stack:
            tool_manager = ToolManager(chromadb_client, mpc_servers=mcp_servers_config)
            await tool_manager.initialize(stack)

            agent = SingleTaskAgentV2(
                client=ollama_client,
                tool_manager=tool_manager,
            )

            input_ = SingleTaskAgentV2Input(
                original_user_query="What is 2 + 2?",
                task="Calculate 2 + 2 and explain the result",
                previous_task_results=[],
                user_context=None,
            )

            output: SingleTaskAgentV2Output = await agent.run(input_)

            assert output.result is not None
            assert len(output.result) > 0
