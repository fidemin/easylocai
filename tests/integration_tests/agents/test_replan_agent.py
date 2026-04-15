import pytest

from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.schemas.context import ExecutedTaskResult, WorkflowContext


class TestReplanAgent:

    @pytest.mark.asyncio
    async def test_replan_with_remaining_tasks(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files and count lines of code",
            query_context=None,
            reformatted_user_query="Find all Python files and count lines of code",
            task_list=[
                "Find all Python files in the project",
                "Count total lines of code",
            ],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Find all Python files in the project",
                    result="Found 15 Python files",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)

        assert isinstance(output.tasks, list)
        assert output.response is None or isinstance(output.response, str)

    @pytest.mark.asyncio
    async def test_replan_all_tasks_complete(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="List files in /tmp",
            query_context=None,
            reformatted_user_query="List files in /tmp",
            task_list=["List files in /tmp directory"],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="List files in /tmp directory",
                    result="Files: a.txt, b.txt, c.txt",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)

        assert output.response is not None or len(output.tasks) > 0
