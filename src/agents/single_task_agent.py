import json
import logging

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.agents.reasoning_agent import ReasoningAgent
from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class SingleTaskAgentInput(BaseModel):
    original_tasks: list[dict]
    original_user_query: str
    task: dict
    previous_task_results: list[dict] = []


class SingleTaskAgentOutput(BaseModel):
    task: dict
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentInput, SingleTaskAgentOutput]):
    _tool_system_prompt_path = "resources/prompts/v2/tool_system_prompt.jinja2"
    _tool_user_prompt_path = "resources/prompts/v2/tool_user_prompt.jinja2"

    _tool_result_system_prompt_path = (
        "resources/prompts/v2/task_result_system_prompt.jinja2"
    )
    _tool_result_user_prompt_path = (
        "resources/prompts/v2/task_result_user_prompt.jinja2"
    )

    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""), undefined=StrictUndefined)

        tool_system_prompt_template = env.get_template(self._tool_system_prompt_path)
        self._tool_system_prompt_template = tool_system_prompt_template

        tool_user_prompt_template = env.get_template(self._tool_user_prompt_path)
        self._tool_user_prompt_template = tool_user_prompt_template

        task_result_system_prompt_template = env.get_template(
            self._tool_result_system_prompt_path
        )
        self._task_result_system_prompt_template = task_result_system_prompt_template

        task_result_user_prompt_template = env.get_template(
            self._tool_result_user_prompt_path
        )
        self._task_result_user_prompt_template = task_result_user_prompt_template

        self._model = DEFAULT_LLM_MODEL
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    async def run(self, input_: SingleTaskAgentInput) -> SingleTaskAgentOutput:
        task = input_.task
        type_ = task["type"]
        task_description = task["description"]

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
            reasoning_result = await reasoning_agent.run(task=task)
            iteration_results = [
                {
                    "subtask": task_description,
                    "result": reasoning_result,
                }
            ]
            logger.debug(f"LLM Result:\n{reasoning_result}")
        else:
            raise ValueError(f"Unknown task type: {type_}")

        result = await self._task_result(
            original_user_query=input_.original_user_query,
            task=task_description,
            iteration_results=iteration_results,
        )

        return SingleTaskAgentOutput(
            task=task,
            result=result,
        )

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

            if tool_result.isError:
                tool_result = f"Error occurred when calling tool: {tool_result.content}"
            else:
                if tool_result.structuredContent:
                    tool_result = tool_result.structuredContent
                else:
                    tool_result = tool_result.content

            iteration_results.append(
                {
                    "subtask": task_result["subtask"],
                    "result": tool_result,
                }
            )
        return {
            "iteration_results": iteration_results,
        }

    async def _task_result(
        self, original_user_query: str, task: str, iteration_results: list
    ) -> str:

        task_result_system_prompt = self._task_result_system_prompt_template.render()
        logger.debug(
            pretty_prompt_text("Task Result System Prompt", task_result_system_prompt)
        )

        task_result_user_prompt = self._task_result_user_prompt_template.render(
            original_user_query=original_user_query,
            task=task,
            iteration_results=iteration_results,
        )

        logger.debug(
            pretty_prompt_text("Task Result User Prompt", task_result_user_prompt)
        )

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": task_result_system_prompt},
                {"role": "user", "content": task_result_user_prompt},
            ],
            options={
                "temperature": 0.2,
            },
        )
        logger.debug(f"Task Result Response:\n{response['message']['content']}")

        return response["message"]["content"]
