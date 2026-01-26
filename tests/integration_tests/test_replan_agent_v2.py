import pytest

from ollama import AsyncClient

from easylocai.agents.replan_agent import (
    ReplanAgentV2,
    ReplanAgentV2Input,
    ReplanAgentV2Output,
)


@pytest.fixture
def ollama_client():
    return AsyncClient(host="http://localhost:11434")


class TestReplanAgentV2:

    @pytest.mark.asyncio
    async def test_replan_with_remaining_tasks(self, ollama_client):
        """Test ReplanAgentV2 when tasks remain to be completed."""
        agent = ReplanAgentV2(client=ollama_client)

        input_ = ReplanAgentV2Input(
            user_query="Find notion documents related to redis and summarize them to redis.txt file",
            user_context=None,
            previous_plan=[
                "Search for notion documents related to redis",
                "Extract relevant information from the documents",
                "Summarize the extracted information",
                "Save the summary to redis.txt",
            ],
            task_results=[
                {
                    "task": "Search for notion documents related to redis",
                    "result": "Found 2 documents: redis_intro.md, redis_config.md",
                },
            ],
        )

        output: ReplanAgentV2Output = await agent.run(input_)

        assert output.tasks is not None
        assert isinstance(output.tasks, list)
        # Should have remaining tasks
        assert len(output.tasks) > 0
        # All tasks should be strings
        for task in output.tasks:
            assert isinstance(task, str)
        # Response should be None since tasks remain
        assert output.response is None

    @pytest.mark.asyncio
    async def test_replan_all_tasks_completed(self, ollama_client):
        """Test ReplanAgentV2 when all tasks are completed."""
        agent = ReplanAgentV2(client=ollama_client)

        input_ = ReplanAgentV2Input(
            user_query="Calculate (2+3) * 12 and save to file result.txt",
            user_context=None,
            previous_plan=[
                "Calculate (2+3) * 12",
                "Save the result to result.txt",
            ],
            task_results=[
                {
                    "task": "Calculate (2+3) * 12",
                    "result": "The result is 60",
                },
                {
                    "task": "Save the result to result.txt",
                    "result": "File result.txt created with content: 60",
                },
            ],
        )

        output: ReplanAgentV2Output = await agent.run(input_)

        # When all tasks are completed, tasks should be empty
        assert output.tasks == []
        # Response should contain the final answer
        assert output.response is not None
        assert len(output.response) > 0

    @pytest.mark.asyncio
    async def test_replan_with_user_context(self, ollama_client):
        """Test ReplanAgentV2 with user context from previous conversation."""
        agent = ReplanAgentV2(client=ollama_client)

        input_ = ReplanAgentV2Input(
            user_query="What was the result?",
            user_context="Previous conversation: User calculated (2+3) * 12 and saved to result.txt. The result was 60.",
            previous_plan=[
                "Retrieve the calculation result from context",
            ],
            task_results=[],
        )

        output: ReplanAgentV2Output = await agent.run(input_)

        assert output is not None
        # Should either have tasks or a response
        assert output.tasks is not None or output.response is not None
