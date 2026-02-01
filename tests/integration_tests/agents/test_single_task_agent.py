from contextlib import AsyncExitStack

import pytest
from ollama import AsyncClient

from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentInput,
    SingleTaskAgentOutput,
)
from easylocai.core.tool_manager import ToolManager
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine


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


class TestSingleTaskAgent:
    @pytest.mark.asyncio
    async def test_tool_task_read_file(
        self, ollama_client, search_engine, mcp_servers_config
    ):
        """Test SingleTaskAgent with a file read task."""
        async with AsyncExitStack() as stack:
            tool_manager = ToolManager(search_engine, mpc_servers=mcp_servers_config)
            await tool_manager.initialize(stack)

            agent = SingleTaskAgent(
                client=ollama_client,
                tool_manager=tool_manager,
            )

            input_ = SingleTaskAgentInput(
                original_user_query="List files in current directory",
                task="List files in current directory",
                previous_task_results=[],
                user_context=None,
            )

            output: SingleTaskAgentOutput = await agent.run(input_)

            assert output.result is not None
            assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_reasoning_task(
        self, ollama_client, search_engine, mcp_servers_config
    ):
        """Test SingleTaskAgent with a reasoning task."""
        async with AsyncExitStack() as stack:
            tool_manager = ToolManager(search_engine, mpc_servers=mcp_servers_config)
            await tool_manager.initialize(stack)

            agent = SingleTaskAgent(
                client=ollama_client,
                tool_manager=tool_manager,
            )

            input_ = SingleTaskAgentInput(
                original_user_query="What is 2 + 2?",
                task="Calculate 2 + 2 and explain the result",
                previous_task_results=[],
                user_context=None,
            )

            output: SingleTaskAgentOutput = await agent.run(input_)

            assert output.result is not None
            assert len(output.result) > 0
