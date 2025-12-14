import json
import logging

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.agents.reasoning_agent import ReasoningAgent
from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class SingleTaskAgent(Agent):
    _prompt_path = "resources/prompts/v2/single_task_prompt_v2.jinja2"
    _tool_system_prompt_path = "resources/prompts/v2/tool_system_prompt.jinja2"
    _tool_user_prompt_path = "resources/prompts/v2/tool_user_prompt.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))

        tool_system_prompt_template = env.get_template(self._tool_system_prompt_path)
        self._tool_system_prompt_template = tool_system_prompt_template

        tool_user_prompt_template = env.get_template(self._tool_user_prompt_path)
        self._tool_user_prompt_template = tool_user_prompt_template

        self._model = DEFAULT_LLM_MODEL
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    async def run(self, **query) -> str | dict:
        task = query["task"]
        type_ = task["type"]
        task_description = task["description"]

        reasoning_agent = ReasoningAgent(client=self._ollama_client)

        if type_ == "tool":
            previous_task_results = query.get("previous_task_results", [])
            original_tasks = query.get("original_tasks", None)

            result = await self._tool_result(
                task_description=task_description,
                previous_task_results=previous_task_results,
                original_tasks=original_tasks,
            )
            logger.debug(f"Tool Result:\n{result}")
        elif type_ == "llm":
            result = await reasoning_agent.run(task=task)
            logger.debug(f"LLM Result:\n{result}")
        else:
            raise ValueError(f"Unknown task type: {type_}")

        return {
            "task": task,
            "result": result,
        }

    async def _tool_result(
        self, *, task_description, previous_task_results, original_tasks
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
            tool_system_prompt = self._tool_system_prompt_template.render()
            logger.debug(pretty_prompt_text("Tool System Prompt", tool_system_prompt))

            tool_user_prompt = self._tool_user_prompt_template.render(
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
                original_tasks=original_tasks,
                task=task_description,
            )
            logger.debug(pretty_prompt_text("Tool User Prompt", tool_user_prompt))

            response = await self._ollama_client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": tool_system_prompt},
                    {"role": "user", "content": tool_user_prompt},
                ],
                options={
                    "temperature": 0.2,
                },
            )

            try:
                task_result = json.loads(response["message"]["content"])
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse SingleTaskAgent response as JSON: {response['message']['content']}"
                    + f"\nThinking: {response['message']['thinking']}"
                )
                iteration_results.append(
                    {
                        "subtask": task_description,
                        "result": f"Error to parse result to JSON."
                        f"\nContent:{response['message']['content']}"
                        f"\nThinking: {response['message']['thinking']}",
                    }
                )
                continue

            logger.debug(f"Task Prompt Response:\n{task_result}")
            finished = task_result["finished"]

            if finished:
                break

            tool_result = await self._server_manager.call_tool(
                task_result["server_name"],
                task_result["tool_name"],
                task_result["tool_args"],
            )

            logger.debug(f"Tool result:\n{tool_result}")

            iteration_results.append(
                {
                    "subtask": task_result["subtask"],
                    "result": tool_result,
                }
            )
        return {
            "task": task_description,
            "iteration_results": iteration_results,
        }
