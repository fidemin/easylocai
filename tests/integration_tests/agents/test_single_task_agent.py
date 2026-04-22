import pytest

from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentOutput,
)
from easylocai.schemas.context import ConversationHistory, ExecutedTaskResult, SingleTaskAgentContext


class TestSingleTaskAgent:

    @pytest.mark.asyncio
    async def test_basic_task_execution(self, ollama_client, tool_manager):
        agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="What is 2 + 2?",
            original_task="Calculate 2 + 2",
        )
        output: SingleTaskAgentOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert len(output.executed_task) > 0
        assert isinstance(output.result, str)
        assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_task_with_previous_results(self, ollama_client, tool_manager):
        agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="Find Python files and count them",
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Find all Python files",
                    result="Found files: a.py, b.py, c.py",
                )
            ],
            original_task="Count the total number of Python files found",
        )
        output: SingleTaskAgentOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert isinstance(output.result, str)

    @pytest.mark.asyncio
    async def test_task_with_conversation_history(self, ollama_client, tool_manager):
        agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="Count them",
            original_task="Count the number of Python files found previously",
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find all Python files",
                    reformatted_user_query="Find all Python files",
                    response="Found 3 Python files: a.py, b.py, c.py",
                )
            ],
        )
        output: SingleTaskAgentOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert isinstance(output.result, str)
