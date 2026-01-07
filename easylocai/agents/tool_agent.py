import logging
from typing import AsyncIterator, Dict, Any

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader

from easylocai.agents.tool_result_filter_agent import ToolResultFilterAgent
from easylocai.agents.tool_select_agent import ToolSelectAgent
from easylocai.core.agent import Agent
from easylocai.core.tool_manager import ServerManager

logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    _prompt_path = "resources/prompts/task_tool_prompt.txt"

    def __init__(
        self,
        *,
        client,
        model: str,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = model
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    async def run_stream(self, query: str | dict) -> AsyncIterator[Dict[str, Any]]:
        original_user_query = query["original_user_query"]
        plan = query["plan"]

        task_tool_data = await self._get_task_and_tool_data(
            original_user_query,
            plan,
            query["previous_task_results"],
        )
        task = task_tool_data["task"]
        display = task_tool_data["display"]

        yield {"end": False, "display": display, "data": None}

        logger.debug(f"Task Tool Response:\n{task_tool_data}")

        if task_tool_data["use_llm"] is True:
            result = await self._get_llm_result(task)
            logger.debug(f"LLM Result: {result}")
            yield {
                "end": True,
                "display": None,
                "data": {
                    "task": task,
                    "result": result,
                },
            }
            return

        if task_tool_data["use_tool"] is False:
            yield {
                "end": True,
                "display": None,
                "data": {
                    "task": task_tool_data["task"],
                    "result": "No tool to use. No result available.",
                },
            }

        tool_result = await self._server_manager.call_tool(
            task_tool_data["server_name"],
            task_tool_data["tool_name"],
            task_tool_data.get("tool_args"),
        )

        logger.debug(f"original tool result:\n{tool_result}")
        filtered_tool_result = await self._filter_tool_result(
            original_user_query,
            task_tool_data["task"],
            tool_result,
        )
        logger.debug(f"filtered tool result:\n{filtered_tool_result}")

        yield {
            "end": True,
            "display": None,
            "data": {
                "task": task_tool_data["task"],
                "result": filtered_tool_result,
            },
        }
        return

    async def _get_llm_result(self, task):
        response = await self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are task execution AI assistant. Answer the user's query as best as you can.",
                },
                {
                    "role": "user",
                    "content": f"{task}",
                },
            ],
        )
        result = response["message"]["content"]
        return result

    async def _get_task_and_tool_data(
        self,
        original_user_query,
        plan,
        previous_task_results,
    ):
        tool_select_agent = ToolSelectAgent(
            client=self._client,
            model=self._model,
            tool_collection=self._tool_collection,
            server_manager=self._server_manager,
        )

        return await tool_select_agent.run(
            previous_task_results=previous_task_results,
            plan=plan,
            original_user_query=original_user_query,
        )

    async def _filter_tool_result(
        self,
        user_query: str,
        task: str,
        tool_result: str,
    ):
        tool_result_filter_agent = ToolResultFilterAgent(
            client=self._client,
            model=self._model,
        )
        return await tool_result_filter_agent.run(
            user_query=user_query,
            task=task,
            tool_result=tool_result,
        )
