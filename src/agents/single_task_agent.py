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

    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = DEFAULT_LLM_MODEL
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    async def run(self, **query) -> str | dict:
        task = query["task"]

        tool_search_result = self._tool_collection.query(
            query_texts=[task],
            n_results=5,
        )

        metadatas = tool_search_result["metadatas"][0]
        reasoning_agent = ReasoningAgent(client=self._ollama_client)
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

        previous_task_results = query.get("previous_task_results", [])
        original_tasks = query.get("original_tasks", None)
        iteration_results = []

        if original_tasks is None:
            logger.warning(
                "original_tasks is not provided to SingleTaskAgent, defaulting to [task]"
            )
            original_tasks = []

        while True:
            prompt = self._prompt_template.render(
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
                original_tasks=original_tasks,
                task=task,
            )
            logger.debug(pretty_prompt_text("Task Prompt", prompt))

            response = await self._ollama_client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
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
                        "executed_task": task,
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

            if task_result["use_reasoning"] is True:
                # TODO: handle reasoning level
                tool_result = await reasoning_agent.run(
                    task=task_result["executed_task"]
                )
                logger.debug(f"Reasoning Agent Result:\n{tool_result}")
            else:
                tool_result = await self._server_manager.call_tool(
                    task_result["server_name"],
                    task_result["tool_name"],
                    task_result["tool_args"],
                )

                logger.debug(f"Tool result:\n{tool_result}")

            iteration_results.append(
                {
                    "executed_task": task_result["executed_task"],
                    "result": tool_result,
                }
            )

        return {
            "task": task,
            "result": iteration_results,
        }
