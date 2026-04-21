import pytest

from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.schemas.context import ConversationHistory, ExecutedTaskResult, WorkflowContext


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

    def test_replanner_input_has_conversation_histories_field(self):
        from easylocai.llm_calls.replanner import ReplannerInput
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt.",
            )
        ]
        inp = ReplannerInput(
            user_context=None,
            original_user_query="Summarize alice.txt",
            previous_plan=["Summarize alice.txt"],
            task_results=[{"task": "Summarize alice.txt", "result": "It is a short file."}],
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1

    @pytest.mark.asyncio
    async def test_replan_with_conversation_history(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Now count how many there are",
            query_context=None,
            reformatted_user_query="Count the number of Python files",
            task_list=["Count the number of Python files"],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Count the number of Python files",
                    result="There are 15 Python files",
                )
            ],
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find all Python files",
                    reformatted_user_query="Find all Python files",
                    response="Found 15 Python files in the project.",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)
        assert output.response is not None or len(output.tasks) > 0
