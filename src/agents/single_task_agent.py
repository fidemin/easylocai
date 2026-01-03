import logging
from typing import Any

from chromadb.types import Collection
from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from src.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentOutput,
    ReasoningAgentInput,
)
from src.core.agent import Agent
from src.core.server import ServerManager
from src.llm_calls.task_result_filter import TaskResultFilter, TaskResultFilterInput
from src.llm_calls.tool_selector import (
    ToolSelector,
    ToolSelectorInput,
    ToolSelectorOutput,
    ToolInput,
)

logger = logging.getLogger(__name__)


class SingleTaskAgentInput(BaseModel):
    original_tasks: list[dict]
    original_user_query: str
    task: dict
    previous_task_results: list[dict] = []
    user_context: str | None


class SingleTaskAgentOutput(BaseModel):
    task: dict
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentInput, SingleTaskAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    async def run(self, input_: SingleTaskAgentInput) -> SingleTaskAgentOutput:
        task = input_.task
        type_ = task["type"]
        task_description = task["description"]
        user_context = input_.user_context

        reasoning_agent = ReasoningAgent(client=self._ollama_client)

        if type_ == "tool":
            previous_task_results = input_.previous_task_results
            original_tasks = input_.original_tasks

            # Example:
            # tool_result = {
            #   "task": task_description,
            #   "iteration_results": [
            #     {
            #       "subtask": "subtask description",
            #       "result": "subtask result",
            #     },
            #   ]
            # }
            tool_result = await self._tool_result(
                task_description=task_description,
                previous_task_results=previous_task_results,
                original_tasks=original_tasks,
                user_context=user_context,
            )

            logger.debug(f"Tool Result:\n{tool_result}")
            iteration_results = tool_result["iteration_results"]

        elif type_ == "llm":
            # Example:
            # result = {
            #   "reasoning": "First, I need to calculate the sum of 2 and 3, which is 5. Next, I will multiply this result by 5. Therefore, 5 multiplied by 5 equals 25.",
            #   "final": "The answer is 25.",
            #   "confidence": 100
            # }
            reasoning_agent_input = ReasoningAgentInput(
                task=task, user_context=user_context
            )
            reasoning_agent_output: ReasoningAgentOutput = await reasoning_agent.run(
                reasoning_agent_input
            )
            iteration_results = [
                {
                    "subtask": task_description,
                    "result": [
                        reasoning_agent_output.model_dump_json(ensure_ascii=False)
                    ],
                }
            ]
            logger.debug(f"LLM Result:\n{reasoning_agent_output}")
        else:
            raise ValueError(f"Unknown task type: {type_}")

        result = await self._filter_task_result(
            original_user_query=input_.original_user_query,
            task=task_description,
            iteration_results=iteration_results,
            user_context=user_context,
        )

        return SingleTaskAgentOutput(
            task=task,
            result=result,
        )

    async def _tool_result(
        self,
        *,
        task_description,
        previous_task_results,
        original_tasks,
        user_context,
    ):
        tool_search_result = self._tool_collection.query(
            query_texts=[task_description],
            n_results=5,
        )

        metadatas = tool_search_result["metadatas"][0]
        tool_candidates = []

        for metadata in metadatas:
            server_name = metadata["server_name"]
            tool_name = metadata["tool_name"]
            tool = self._server_manager.get_server(server_name).get_tool(tool_name)
            tool_candidates.append(
                {
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "tool_input_schema": tool.input_schema,
                }
            )

        iteration_results = []

        if original_tasks is None:
            logger.warning(
                "original_tasks is not provided to SingleTaskAgent, defaulting to [task]"
            )
            original_tasks = []

        while True:
            tool_selector_input = ToolSelectorInput(
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
                original_tasks=original_tasks,
                task=task_description,
                user_context=user_context,
            )
            tool_selector = ToolSelector(client=self._ollama_client)
            try:
                tool_selector_output: ToolSelectorOutput = await tool_selector.call(
                    tool_selector_input
                )
            except ValidationError:
                llm_call_response = tool_selector.current_llm_call_response

                logger.error(
                    f"Failed to parse ToolSelector response as JSON: {llm_call_response['message']['content']}"
                    + f"\nThinking: {llm_call_response['message']['thinking']}"
                )
                iteration_results.append(
                    {
                        "subtask": task_description,
                        "result": [
                            {
                                "error": f"Error to parse llm call response content."
                                f"\nContent:{llm_call_response['message']['content']}"
                                f"\nThinking: {llm_call_response['message']['thinking']}"
                            }
                        ],
                    }
                )
                continue

            finished = tool_selector_output.finished

            if finished:
                return {
                    "iteration_results": iteration_results,
                }

            tool_results = []
            for tool_input in tool_selector_output.selected_tools:
                tool_result = await self._call_tool(tool_input)
                tool_results.append(tool_result)

            iteration_results.append(
                {
                    "subtask": tool_selector_output.subtask,
                    "result": tool_results,
                }
            )
        return {
            "iteration_results": iteration_results,
        }

    async def _call_tool(self, tool_input: ToolInput) -> dict[str, Any]:
        tool_result = await self._server_manager.call_tool(
            tool_input.server_name,
            tool_input.tool_name,
            tool_input.tool_args,
        )
        logger.debug(f"Tool Call result:\n{tool_result}")

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
        iteration_results: list,
        user_context: str,
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
