import logging
from typing import Any

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)
from easylocai.core.agent import Agent
from easylocai.core.tool_manager import ToolManager
from easylocai.llm_calls.task_result_filter import (
    TaskResultFilter,
    TaskResultFilterInput,
)
from easylocai.llm_calls.task_router import (
    TaskRouter,
    TaskRouterInput,
    TaskRouterOutput,
)
from easylocai.llm_calls.tool_selector import (
    ToolInput,
    ToolSelectorV2,
    ToolSelectorV2Input,
    ToolSelectorV2Output,
)

logger = logging.getLogger(__name__)


class SingleTaskAgentInput(BaseModel):
    original_user_query: str
    task: str
    previous_task_results: list[dict] = []
    user_context: str | None


class SingleTaskAgentOutput(BaseModel):
    task: str
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentInput, SingleTaskAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_manager: ToolManager,
    ):
        self._ollama_client = client
        self._tool_manager = tool_manager

    async def run(self, input_: SingleTaskAgentInput) -> SingleTaskAgentOutput:
        task = input_.task
        user_context = input_.user_context
        previous_task_results = input_.previous_task_results

        tool_candidates = await self._get_tool_candidates(task)
        iteration_results = []

        while True:
            task_router_output = await self._route_task(
                task=task,
                user_context=user_context,
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
            )

            if task_router_output.finished:
                logger.debug(f"Task finished: {task_router_output.finished_reason}")
                break

            subtask = task_router_output.subtask
            subtask_type = task_router_output.subtask_type

            if subtask_type == "tool":
                result = await self._execute_tool_subtask(
                    subtask=subtask,
                    user_context=user_context,
                    tool_candidates=tool_candidates,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
            elif subtask_type == "reasoning":
                result = await self._execute_reasoning_subtask(
                    subtask=subtask,
                    user_context=user_context,
                    previous_task_results=previous_task_results,
                    previous_subtask_results=iteration_results,
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask_type}")

            iteration_results.append(
                {
                    "subtask": subtask,
                    "result": result,
                }
            )

        filtered_result = await self._filter_task_result(
            original_user_query=input_.original_user_query,
            task=task,
            iteration_results=iteration_results,
            user_context=user_context,
        )

        return SingleTaskAgentOutput(
            task=task,
            result=filtered_result,
        )

    async def _get_tool_candidates(self, task: str) -> list[dict]:
        tools = await self._tool_manager.search_tools([task], n_results=10)

        tool_candidates = []
        for tool in tools:
            tool_candidates.append(
                {
                    "server_name": tool.server_name,
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "tool_input_schema": tool.input_schema,
                }
            )

        return tool_candidates

    async def _route_task(
        self,
        *,
        task: str,
        user_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> TaskRouterOutput:
        task_router_input = TaskRouterInput(
            task=task,
            user_context=user_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )

        task_router = TaskRouter(client=self._ollama_client)
        task_router_output: TaskRouterOutput = await task_router.call(task_router_input)

        logger.debug(f"TaskRouter output: {task_router_output}")
        return task_router_output

    async def _execute_tool_subtask(
        self,
        *,
        subtask: str,
        user_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> dict[str, Any]:
        tool_selector_input = ToolSelectorV2Input(
            subtask=subtask,
            user_context=user_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )

        tool_selector = ToolSelectorV2(client=self._ollama_client)

        try:
            tool_selector_output: ToolSelectorV2Output = await tool_selector.call(
                tool_selector_input
            )
        except ValidationError:
            llm_call_response = tool_selector.current_llm_call_response
            logger.error(
                f"Failed to parse ToolSelectorV2 response: {llm_call_response['message']['content']}"
            )
            return {"error": "Failed to parse tool selector response"}

        if tool_selector_output.selected_tool is None:
            logger.warning(
                f"No tool selected for subtask: {subtask}. Reason: {tool_selector_output.failure_reason}"
            )
            return {"error": tool_selector_output.failure_reason}

        tool_result = await self._call_tool(tool_selector_output.selected_tool)
        return tool_result

    async def _execute_reasoning_subtask(
        self,
        *,
        subtask: str,
        user_context: str | None,
        previous_task_results: list[dict],
        previous_subtask_results: list[dict],
    ) -> dict[str, Any]:
        reasoning_agent_input = ReasoningAgentInput(
            task={"description": subtask},
            user_context=user_context,
            previous_task_results=previous_task_results,
            previous_subtask_results=previous_subtask_results,
        )

        reasoning_agent = ReasoningAgent(client=self._ollama_client)
        reasoning_agent_output: ReasoningAgentOutput = await reasoning_agent.run(
            reasoning_agent_input
        )

        logger.debug(f"ReasoningAgent output: {reasoning_agent_output}")
        return reasoning_agent_output.model_dump()

    async def _call_tool(self, tool_input: ToolInput) -> dict[str, Any]:
        tool_result = await self._tool_manager._server_manager.call_tool(
            tool_input.server_name,
            tool_input.tool_name,
            tool_input.tool_args,
        )
        logger.debug(f"Tool call result: {tool_result}")

        if tool_result.isError:
            return {"error": f"Error occurred when calling tool: {tool_result.content}"}

        if tool_result.structuredContent:
            return tool_result.structuredContent
        else:
            return {"content": tool_result.content}

    async def _filter_task_result(
        self,
        original_user_query: str,
        task: str,
        iteration_results: list[dict],
        user_context: str | None,
    ) -> str:
        task_result_filter_input = TaskResultFilterInput(
            original_user_query=original_user_query,
            task=task,
            iteration_results=iteration_results,
            user_context=user_context,
        )

        task_result_filter = TaskResultFilter(client=self._ollama_client)
        task_result_filter_output = await task_result_filter.call(
            task_result_filter_input
        )

        return task_result_filter_output.root
