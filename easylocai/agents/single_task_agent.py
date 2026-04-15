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
from easylocai.llm_calls.subtask_result_filter import (
    SubtaskResultFilter,
    SubtaskResultFilterInput,
)
from easylocai.llm_calls.task_result_filter import (
    TaskResultFilter,
    TaskResultFilterInput,
)
from easylocai.llm_calls.task_router import TaskRouter, TaskRouterInput, TaskRouterOutput
from easylocai.llm_calls.tool_selector import (
    ToolInput,
    ToolSelector,
    ToolSelectorInput,
    ToolSelectorOutput,
)
from easylocai.schemas.context import SingleTaskAgentContext, SubtaskResult

logger = logging.getLogger(__name__)


class SingleTaskAgentOutput(BaseModel):
    executed_task: str
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentContext, SingleTaskAgentOutput]):
    N_TOOL_RESULTS = 18

    def __init__(self, *, client: AsyncClient, tool_manager: ToolManager):
        self._ollama_client = client
        self._tool_manager = tool_manager

    async def _run(self, input_: SingleTaskAgentContext) -> SingleTaskAgentOutput:
        ctx = input_
        tool_candidates = await self._get_tool_candidates([ctx.original_task])

        previous_task_results = [
            {"task": r.executed_task, "result": r.result}
            for r in ctx.executed_task_results
        ]

        while True:
            iteration_results = [r.model_dump() for r in ctx.subtask_results]

            task_router_output = await self._route_task(
                task=ctx.original_task,
                user_context=ctx.query_context,
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
                    user_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
            elif subtask_type == "reasoning":
                result = await self._execute_reasoning_subtask(
                    subtask=subtask,
                    user_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    previous_subtask_results=iteration_results,
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask_type}")

            filtered_result = await self._filter_subtask_result(subtask=subtask, result=result)
            ctx.subtask_results.append(SubtaskResult(subtask=subtask, result=filtered_result))

        final_result = await self._filter_task_result(
            task=ctx.original_task,
            subtask_results=[r.model_dump() for r in ctx.subtask_results],
            user_context=ctx.query_context,
        )

        return SingleTaskAgentOutput(
            executed_task=ctx.original_task,
            result=final_result,
        )

    async def _get_tool_candidates(self, queries: list[str]) -> list[dict]:
        tools = await self._tool_manager.search_tools(queries, n_results=self.N_TOOL_RESULTS)
        return [
            {
                "server_name": t.server_name,
                "tool_name": t.name,
                "tool_description": t.description,
                "tool_input_schema": t.input_schema,
            }
            for t in tools
        ]

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
        output: TaskRouterOutput = await task_router.call(task_router_input)
        logger.debug(f"TaskRouter output: {output}")
        return output

    async def _execute_tool_subtask(
        self,
        *,
        subtask: str,
        user_context: str | None,
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> dict[str, Any]:
        subtask_tool_candidates = await self._get_tool_candidates([subtask])
        tool_selector_input = ToolSelectorInput(
            subtask=subtask,
            user_context=user_context,
            tool_candidates=subtask_tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )
        tool_selector = ToolSelector(client=self._ollama_client)
        try:
            tool_selector_output: ToolSelectorOutput = await tool_selector.call(tool_selector_input)
        except ValidationError:
            llm_call_response = tool_selector.current_llm_call_response
            logger.error(
                f"Failed to parse ToolSelector response: {llm_call_response['message']['content']}"
            )
            return {"error": "Failed to parse tool selector response"}

        if tool_selector_output.selected_tool is None:
            logger.warning(f"No tool selected for subtask: {subtask}")
            return {"error": tool_selector_output.failure_reason}

        return await self._call_tool(tool_selector_output.selected_tool)

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
        reasoning_agent_output: ReasoningAgentOutput = await reasoning_agent.run(reasoning_agent_input)
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
        return {"content": tool_result.content}

    async def _filter_subtask_result(self, subtask: str, result: dict[str, Any]) -> str:
        subtask_result_filter_input = SubtaskResultFilterInput(subtask=subtask, result=result)
        subtask_result_filter = SubtaskResultFilter(client=self._ollama_client)
        output = await subtask_result_filter.call(subtask_result_filter_input)
        return output.root

    async def _filter_task_result(
        self,
        task: str,
        subtask_results: list[dict],
        user_context: str | None,
    ) -> str:
        task_result_filter_input = TaskResultFilterInput(
            task=task,
            subtask_results=subtask_results,
            user_context=user_context,
        )
        task_result_filter = TaskResultFilter(client=self._ollama_client)
        output = await task_result_filter.call(task_result_filter_input)
        return output.root
