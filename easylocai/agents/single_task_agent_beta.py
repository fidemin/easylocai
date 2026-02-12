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
    TaskRouterInput,
    TaskRouterOutputV2,
    TaskRouterV2,
    Subtask,
)
from easylocai.llm_calls.tool_selector import (
    ToolInput,
    ToolSelectorInputV2,
    ToolSelectorV2,
    ToolSelectorOutputV2,
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
            task_router_output: TaskRouterOutputV2 = await self._route_task(
                task=task,
                user_context=user_context,
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
            )

            if task_router_output.finished:
                logger.debug(f"Task finished: {task_router_output.finished_reason}")
                break

            subtasks: list[Subtask] = task_router_output.subtasks
            tool_subtasks = []
            reasoning_subtasks = []

            for subtask in subtasks:
                if subtask.subtask_type == "tool":
                    tool_subtasks.append(subtask)
                elif subtask.subtask_type == "reasoning":
                    reasoning_subtasks.append(subtask)
                else:
                    logger.error(
                        f"Unknown subtask type: {task_router_output.subtask_type}"
                    )
                    continue

            if tool_subtasks:
                subtask_list: list[str] = [subtask.subtask for subtask in tool_subtasks]
                subtask_results = await self._execute_tool_subtask(
                    subtasks=subtask_list,
                    user_context=user_context,
                    tool_candidates=tool_candidates,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
                iteration_results.extend(subtask_results)

            if reasoning_subtasks:
                for subtask in reasoning_subtasks:
                    result = await self._execute_reasoning_subtask(
                        subtask=subtask.subtask,
                        user_context=user_context,
                        previous_task_results=previous_task_results,
                        previous_subtask_results=iteration_results,
                    )
                    iteration_results.append(
                        {
                            "subtask": subtask.subtask,
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
    ) -> TaskRouterOutputV2:
        task_router_input = TaskRouterInput(
            task=task,
            user_context=user_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )

        task_router = TaskRouterV2(client=self._ollama_client)
        task_router_output: TaskRouterOutputV2 = await task_router.call(
            task_router_input
        )

        logger.debug(f"TaskRouter output: {task_router_output}")
        return task_router_output

    async def _execute_tool_subtask(
        self,
        *,
        subtasks: list[str],
        user_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> list[dict[str, Any]]:
        tool_selector_input = ToolSelectorInputV2(
            subtasks=subtasks,
            user_context=user_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )

        tool_selector = ToolSelectorV2(client=self._ollama_client)

        subtask_results = []
        try:
            tool_selector_output: ToolSelectorOutputV2 = await tool_selector.call(
                tool_selector_input
            )
        except ValidationError:
            llm_call_response = tool_selector.current_llm_call_response
            logger.error(
                f"Failed to parse ToolSelectorV2 response: {llm_call_response['message']['content']}"
            )
            for subtask in subtasks:
                subtask_results.append(
                    {"subtask": subtask, "result": "tool select fail: parsing error"}
                )
            return subtask_results

        for tool_selector_result in tool_selector_output.results:
            subtask = tool_selector_result.subtask
            selected_tool = tool_selector_result.selected_tool
            failure_reason = tool_selector_result.failure_reason

            if selected_tool is None:
                logger.warning(
                    f"No tool selected for subtask: {subtask}. Reason: {failure_reason}"
                )
                subtask_results.append(
                    {"subtask": subtask, "result": f"no tool found: {failure_reason}"}
                )
                continue

            tool_result = await self._call_tool(selected_tool)
            subtask_results.append(
                {
                    "subtask": subtask,
                    "result": tool_result,
                }
            )

        return subtask_results

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
