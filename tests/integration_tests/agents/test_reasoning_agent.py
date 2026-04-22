import pytest

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)
from easylocai.schemas.context import ConversationHistory


def assert_reasoning_output(output: ReasoningAgentOutput) -> None:
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0
    assert isinstance(output.final, str)
    assert len(output.final) > 0
    assert isinstance(output.confidence, int)
    assert 0 <= output.confidence <= 100


class TestReasoningAgent:

    @pytest.mark.asyncio
    async def test_without_context_or_prior_results(self, ollama_client):
        """Happy path: no query_context, no previous results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="What is 2 + 2?",
            task={"description": "What is 2 + 2?"},
            query_context=None,
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_query_context(self, ollama_client):
        """Variant: query_context provided, no prior results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="Summarize the user's goal",
            task={"description": "Summarize the user's goal"},
            query_context="The user wants to refactor their Python codebase to use async/await.",
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_previous_task_results(self, ollama_client):
        """Variant: previous_task_results provided, no query_context."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="How many Python files were found?",
            task={"description": "How many Python files were found?"},
            query_context=None,
            previous_task_results=[
                {
                    "task": "List all files",
                    "result": "Found: main.py, utils.py, config.py, README.md",
                }
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_previous_subtask_results(self, ollama_client):
        """Variant: previous_subtask_results provided, no query_context or task results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="Combine counts into a total",
            task={"description": "Combine counts into a total"},
            query_context=None,
            previous_task_results=[],
            previous_subtask_results=[
                {"subtask": "Count files in src/", "result": "5 files"},
                {"subtask": "Count files in tests/", "result": "3 files"},
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_conversation_history(self, ollama_client):
        """Variant: conversation_histories provided for multi-turn context."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="Count how many there are",
            task={"description": "Count how many Python files were found"},
            query_context=None,
            previous_task_results=[],
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find all Python files",
                    reformatted_user_query="Find all Python files",
                    response="Found 3 Python files: a.py, b.py, c.py",
                )
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_all_context_combined(self, ollama_client):
        """Complex case: query_context + previous_task_results + previous_subtask_results all populated."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            original_task="Write a summary of all findings",
            task={"description": "Write a summary of all findings"},
            query_context="User is auditing the project structure before a refactor.",
            previous_task_results=[
                {"task": "List top-level directories", "result": "src/, tests/, docs/"},
            ],
            previous_subtask_results=[
                {"subtask": "Count Python files in src/", "result": "12 files"},
                {"subtask": "Count test files", "result": "8 test files"},
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)
