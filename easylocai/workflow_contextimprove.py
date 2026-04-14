import logging
from contextlib import AsyncExitStack
from typing import AsyncGenerator

from ollama import AsyncClient

from easylocai.agents.plan_agent_contextimprove import (
    PlanAgentContextImprove,
    PlanAgentContextImproveInput,
    PlanAgentContextImproveOutput,
)
from easylocai.agents.replan_agent_contextimprove import (
    ReplanAgentContextImprove,
    ReplanAgentContextImproveInput,
    ReplanAgentContextImproveOutput,
)
from easylocai.agents.single_task_agent_contextimprove import (
    SingleTaskAgentContextImprove,
    SingleTaskAgentContextImproveOutput,
)
from easylocai.core.tool_manager import ToolManager
from easylocai.schemas.common import EasyLocaiWorkflowOutput
from easylocai.schemas.context import (
    ConversationHistory,
    ExecutedTaskResult,
    GlobalContext,
    SingleTaskAgentContext,
    WorkflowContext,
)
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine

logger = logging.getLogger(__name__)


def ensure_initialized(func):
    async def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError(
                "EasylocaiWorkflowContextImprove is not initialized. "
                "Please call 'initialize' before running."
            )
        async for item in func(self, *args, **kwargs):
            yield item

    return wrapper


class EasylocaiWorkflowContextImprove:
    def __init__(
        self,
        *,
        config_dict: dict,
        search_engine: AdvancedSearchEngine,
        ollama_client: AsyncClient,
    ):
        self._tool_manager = ToolManager(
            search_engine, mpc_servers=config_dict["mcpServers"]
        )
        self._plan_agent = PlanAgentContextImprove(client=ollama_client)
        self._replan_agent = ReplanAgentContextImprove(client=ollama_client)
        self._single_task_agent = SingleTaskAgentContextImprove(
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
        global_context: GlobalContext,
    ) -> AsyncGenerator[EasyLocaiWorkflowOutput, None]:
        workflow_context = WorkflowContext(
            conversation_histories=global_context.conversation_histories,
            original_user_query=user_query,
        )

        yield EasyLocaiWorkflowOutput(type="status", message="Thinking...")

        plan_output: PlanAgentContextImproveOutput = await self._plan_agent.run(
            PlanAgentContextImproveInput(workflow_context=workflow_context)
        )

        workflow_context.query_context = plan_output.query_context
        workflow_context.reformatted_user_query = plan_output.reformatted_user_query
        workflow_context.task_list = plan_output.task_list

        logger.debug(f"Plan output: {plan_output}")

        answer = None
        while True:
            next_task = workflow_context.task_list[0]
            yield EasyLocaiWorkflowOutput(type="status", message=next_task)

            single_task_context = SingleTaskAgentContext(
                conversation_histories=workflow_context.conversation_histories,
                original_user_query=workflow_context.original_user_query,
                query_context=workflow_context.query_context,
                reformatted_user_query=workflow_context.reformatted_user_query,
                task_list=workflow_context.task_list,
                executed_task_results=workflow_context.executed_task_results,
                original_task=next_task,
            )

            task_output: SingleTaskAgentContextImproveOutput = await self._single_task_agent.run(
                single_task_context
            )

            workflow_context.executed_task_results.append(
                ExecutedTaskResult(
                    executed_task=task_output.executed_task,
                    result=task_output.result,
                )
            )

            yield EasyLocaiWorkflowOutput(type="status", message="Check for completion...")

            replan_output: ReplanAgentContextImproveOutput = await self._replan_agent.run(
                ReplanAgentContextImproveInput(workflow_context=workflow_context)
            )
            logger.debug(f"Replan output: {replan_output}")

            if replan_output.response is not None:
                answer = replan_output.response
                break

            workflow_context.task_list = replan_output.tasks

        # GlobalContext 업데이트
        global_context.conversation_histories.append(
            ConversationHistory(
                original_user_query=user_query,
                reformatted_user_query=workflow_context.reformatted_user_query or user_query,
                query_context=workflow_context.query_context,
                response=answer,
            )
        )

        yield EasyLocaiWorkflowOutput(type="result", message=answer)
