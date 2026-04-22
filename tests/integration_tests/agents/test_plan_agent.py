import pytest

from easylocai.agents.plan_agent import (
    PlanAgent,
    PlanAgentInput,
    PlanAgentOutput,
)
from easylocai.schemas.context import ConversationHistory, WorkflowContext


def assert_output(output: PlanAgentOutput) -> None:
    assert isinstance(output.task_list, list)
    assert len(output.task_list) > 0
    for task in output.task_list:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.reformatted_user_query
    assert output.query_context is None or isinstance(output.query_context, str)


class TestPlanAgent:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        agent = PlanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files in the current directory and count them",
        )
        input_ = PlanAgentInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)

    def test_planner_input_has_conversation_histories_field(self):
        from easylocai.llm_calls.planner import PlannerInput
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt with 42 lines.",
            )
        ]
        inp = PlannerInput(
            user_query="Summarize alice.txt",
            user_context=None,
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        agent = PlanAgent(client=ollama_client)
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
        input_ = PlanAgentInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)
