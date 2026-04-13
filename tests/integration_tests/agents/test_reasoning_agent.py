import pytest

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)


def assert_reasoning_output(output: ReasoningAgentOutput) -> None:
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0
    assert isinstance(output.final, str)
    assert len(output.final) > 0
    assert isinstance(output.confidence, int)


class TestReasoningAgent:

    @pytest.mark.asyncio
    async def test_without_context_or_prior_results(self, ollama_client):
        """Happy path: no user_context, no previous results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "What is 2 + 2?"},
            user_context=None,
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_user_context(self, ollama_client):
        """Variant: user_context provided, no prior results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Summarize the user's goal"},
            user_context="The user wants to refactor their Python codebase to use async/await.",
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_previous_task_results(self, ollama_client):
        """Variant: previous_task_results provided, no user_context."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "How many Python files were found?"},
            user_context=None,
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
        """Variant: previous_subtask_results provided, no user_context or task results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Combine counts into a total"},
            user_context=None,
            previous_task_results=[],
            previous_subtask_results=[
                {"subtask": "Count files in src/", "result": "5 files"},
                {"subtask": "Count files in tests/", "result": "3 files"},
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_all_context_combined(self, ollama_client):
        """Complex case: user_context + previous_task_results + previous_subtask_results all populated."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Write a summary of all findings"},
            user_context="User is auditing the project structure before a refactor.",
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
