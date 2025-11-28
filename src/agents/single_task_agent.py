import json
import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class SingleTaskAgent(Agent):
    _prompt_path = "resources/prompts/v2/single_task_prompt_v2.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = DEFAULT_LLM_MODEL

    async def run(self, **query) -> str | dict:
        task = query["task"]

        tool_candidates = query["tool_candidates"]
        previous_task_results = query.get("previous_task_results", [])
        iteration_results = query.get("iteration_results", [])
        original_tasks = query.get("original_tasks", None)

        if original_tasks is None:
            logger.warning(
                "original_tasks is not provided to SingleTaskAgent, defaulting to [task]"
            )
            original_tasks = []

        prompt = self._prompt_template.render(
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
            original_tasks=original_tasks,
        )
        logger.debug(pretty_prompt_text("Single Task Prompt", prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Current Task:\n{task}"},
            ],
            options={
                "temperature": 0.2,
                "max_tokens": 10000,
            },
        )
        try:
            result = json.loads(response["message"]["content"])
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse SingleTaskAgent response as JSON: {response['message']['content']}"
            )
            raise

        return result
