import logging
from contextlib import AsyncExitStack
from typing import AsyncGenerator

from chromadb import ClientAPI
from ollama import AsyncClient

from easylocai.agents.plan_agent import PlanAgent, PlanAgentInput, PlanAgentOutput
from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentInput,
    SingleTaskAgentOutput,
)
from easylocai.core.tool_manager import ToolManager
from easylocai.schemas.common import EasyLocaiWorkflowOutput, UserConversation

logger = logging.getLogger(__name__)


def ensure_initialized(func):
    async def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError(
                "EasylocaiWorkflow is not initialized. "
                "Please call 'initialize' method before running the workflow."
            )
        async for item in func(self, *args, **kwargs):
            yield item

    return wrapper


class EasylocaiWorkflow:
    def __init__(
        self,
        *,
        config_dict: dict,
        chromadb_client: ClientAPI,
        ollama_client: AsyncClient,
    ):
        self._config_dict = config_dict

        self._tool_manager = ToolManager(
            chromadb_client, mpc_servers=config_dict["mcpServers"]
        )
        self._plan_agent = PlanAgent(client=ollama_client)
        self._replan_agent = ReplanAgent(client=ollama_client)
        self._single_task_agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=self._tool_manager,
        )
        self._initialized = False

    def initialize(self, stack: AsyncExitStack):
        self._initialized = True
        return self._tool_manager.initialize(stack)

    @ensure_initialized
    async def run(
        self,
        user_query: str,
        *,
        user_conversations: list,
    ) -> AsyncGenerator[EasyLocaiWorkflowOutput, None]:
        plan_agent_input = PlanAgentInput(
            user_query=user_query,
            user_conversations=user_conversations,
        )

        yield EasyLocaiWorkflowOutput(type="status", message="Thinking...")

        plan_agent_output: PlanAgentOutput = await self._plan_agent.run(
            plan_agent_input
        )

        logger.debug(f"Plan Agent Response:\n{plan_agent_output}")

        tasks = plan_agent_output.tasks
        user_context = plan_agent_output.context
        previous_task_results = []

        answer = None
        while True:
            next_task = tasks[0]

            yield EasyLocaiWorkflowOutput(type="status", message=next_task)

            task_agent_input = SingleTaskAgentInput(
                original_user_query=user_query,
                task=next_task,
                previous_task_results=previous_task_results,
                user_context=user_context,
            )

            task_agent_response: SingleTaskAgentOutput = (
                await self._single_task_agent.run(task_agent_input)
            )

            previous_task_results.append(
                {
                    "task": task_agent_response.task,
                    "result": task_agent_response.result,
                }
            )

            yield EasyLocaiWorkflowOutput(type="status", message="Check for completion...")

            replan_agent_input = ReplanAgentInput(
                user_query=user_query,
                previous_plan=tasks,
                task_results=previous_task_results,
                user_context=user_context,
            )

            replan_agent_output: ReplanAgentOutput = await self._replan_agent.run(
                replan_agent_input
            )
            logger.debug(f"ReplanAgent Response:\n{replan_agent_output}")

            response = replan_agent_output.response

            if response is not None:
                answer = response
                break

            tasks = replan_agent_output.tasks

        user_conversations.append(
            UserConversation(user_query=user_query, assistant_answer=answer)
        )
        yield EasyLocaiWorkflowOutput(type="result", message=answer)
