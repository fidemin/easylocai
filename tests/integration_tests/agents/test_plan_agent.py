import pytest

from easylocai.agents.plan_agent import PlanAgent, PlanAgentInput, PlanAgentOutput
from easylocai.schemas.common import UserConversation


def assert_plan_output(output: PlanAgentOutput) -> None:
    assert isinstance(output.tasks, list)
    assert len(output.tasks) > 0
    for task in output.tasks:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.context is None or isinstance(output.context, str)


class TestPlanAgent:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        """Happy path: query with no conversation history."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Find all Python files in the current directory and count them",
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        """Variant: query accompanied by one prior conversation turn."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Summarize it",
            user_conversations=[
                UserConversation(
                    user_query="Find the README.md file",
                    assistant_answer="Found README.md in the root directory. It contains project setup instructions.",
                )
            ],
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, ollama_client):
        """Complex case: query with three prior conversation turns."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Now delete them",
            user_conversations=[
                UserConversation(
                    user_query="Find all log files",
                    assistant_answer="Found 3 log files: app.log, error.log, debug.log",
                ),
                UserConversation(
                    user_query="Show me the largest one",
                    assistant_answer="error.log is the largest at 2MB",
                ),
                UserConversation(
                    user_query="What does it contain?",
                    assistant_answer="It contains database connection errors from last week",
                ),
            ],
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)
