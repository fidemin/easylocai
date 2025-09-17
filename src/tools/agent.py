import json
import logging

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader

from src.core.agent import Agent
from src.core.server import ServerManager
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class TaskToolAgent(Agent):
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

    async def run(self, query: str | dict) -> str | dict:
        original_user_query = query["original_user_query"]
        plan = query["plan"]

        previous_task_results: list[dict] = query["previous_task_results"]
        tool_search_result = self._tool_collection.query(
            query_texts=[plan],
            n_results=5,
        )

        metadatas = tool_search_result["metadatas"][0]
        possible_tools = []

        for metadata in metadatas:
            server_name = metadata["server_name"]
            tool_name = metadata["tool_name"]
            tool = self._server_manager.get_server(server_name).get_tool(tool_name)
            possible_tools.append(
                {
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "tool_input_schema": tool.input_schema,
                }
            )

        prompt = self._prompt_template.render(
            original_user_query=original_user_query,
            plan=plan,
            possible_tools=possible_tools,
            previous_task_results=previous_task_results,
        )

        logger.debug(pretty_prompt_text("Task Tool Prompt", prompt))

        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        response_content = response["message"]["content"]
        task_tool_data = json.loads(response_content)

        logger.debug(f"Task Tool Response:\n{task_tool_data}")

        if task_tool_data["use_llm"] is True:
            task = task_tool_data["task"]
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
            logger.debug(f"LLM Result: {result}")
            return {
                "task": task,
                "result": result,
            }

        if task_tool_data["use_tool"] is False:
            return {
                "task": task_tool_data["task"],
                "result": "No tool to use.",
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

        return {
            "task": task_tool_data["task"],
            "result": filtered_tool_result,
        }

    async def _filter_tool_result(
        self,
        user_query: str,
        task: str,
        tooL_result: str,
    ):
        env = Environment(loader=FileSystemLoader(""))
        template = env.get_template("resources/prompts/tool_result_prompt.txt")
        prompt = template.render(
            {"user_query": user_query, "task": task, "tooL_result": tooL_result}
        )

        logger.debug(pretty_prompt_text("Tool Filter Prompt", prompt))

        tooL_result_str = (
            "TOOL RESULT:\n" + tooL_result
            if isinstance(tooL_result, str)
            else str(tooL_result)
        )

        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": tooL_result_str,
                },
            ],
        )
        return response["message"]["content"]
