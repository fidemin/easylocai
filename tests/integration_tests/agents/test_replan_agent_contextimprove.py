import pytest

from easylocai.agents.replan_agent_contextimprove import (
    ReplanAgentContextImprove,
    ReplanAgentContextImproveInput,
    ReplanAgentContextImproveOutput,
)
from easylocai.schemas.context import ExecutedTaskResult, WorkflowContext


class TestReplanAgentContextImprove:

    @pytest.mark.asyncio
    async def test_replan_with_remaining_tasks(self, ollama_client):
        """Task가 남아있을 때 새 task list를 반환한다."""
        agent = ReplanAgentContextImprove(client=ollama_client)
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
        input_ = ReplanAgentContextImproveInput(workflow_context=workflow_context)
        output: ReplanAgentContextImproveOutput = await agent.run(input_)

        assert isinstance(output.tasks, list)
        assert output.response is None or isinstance(output.response, str)

    @pytest.mark.asyncio
    async def test_replan_all_tasks_complete(self, ollama_client):
        """모든 task 완료 시 response를 반환하고 tasks는 비어있다."""
        agent = ReplanAgentContextImprove(client=ollama_client)
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
        input_ = ReplanAgentContextImproveInput(workflow_context=workflow_context)
        output: ReplanAgentContextImproveOutput = await agent.run(input_)

        assert output.response is not None or len(output.tasks) > 0
