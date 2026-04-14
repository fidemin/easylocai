import pytest

from easylocai.agents.plan_agent_contextimprove import (
    PlanAgentContextImprove,
    PlanAgentContextImproveInput,
    PlanAgentContextImproveOutput,
)
from easylocai.schemas.context import ConversationHistory, WorkflowContext


def assert_output(output: PlanAgentContextImproveOutput) -> None:
    assert isinstance(output.task_list, list)
    assert len(output.task_list) > 0
    for task in output.task_list:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.reformatted_user_query
    assert output.query_context is None or isinstance(output.query_context, str)


class TestPlanAgentContextImprove:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        """Happy path: conversation_histories 없이 기본 query 처리."""
        agent = PlanAgentContextImprove(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files in the current directory and count them",
        )
        input_ = PlanAgentContextImproveInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        """Variant: 이전 대화 기록이 있는 경우."""
        agent = PlanAgentContextImprove(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Summarize it",
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find the README.md file",
                    reformatted_user_query="Find the README.md file",
                    response="Found README.md in the root directory.",
                )
            ],
        )
        input_ = PlanAgentContextImproveInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)
