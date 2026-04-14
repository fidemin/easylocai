import pytest

from easylocai.agents.single_task_agent_contextimprove import (
    SingleTaskAgentContextImprove,
    SingleTaskAgentContextImproveOutput,
)
from easylocai.schemas.context import SingleTaskAgentContext


class TestSingleTaskAgentContextImprove:

    @pytest.mark.asyncio
    async def test_basic_task_execution(self, ollama_client, tool_manager):
        """Happy path: tool 없이 reasoning으로 단순 task 처리."""
        agent = SingleTaskAgentContextImprove(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="What is 2 + 2?",
            original_task="Calculate 2 + 2",
        )
        output: SingleTaskAgentContextImproveOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert len(output.executed_task) > 0
        assert isinstance(output.result, str)
        assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_task_with_previous_results(self, ollama_client, tool_manager):
        """Variant: 이전 task 결과가 있는 경우."""
        from easylocai.schemas.context import ExecutedTaskResult

        agent = SingleTaskAgentContextImprove(
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
        output: SingleTaskAgentContextImproveOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert isinstance(output.result, str)
