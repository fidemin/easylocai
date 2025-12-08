import json
import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class ReasoningAgent(Agent):
    _prompt_path = "resources/prompts/v2/reasoning_prompt.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
        prompt_path: str | None = None,
    ):
        if prompt_path is not None:
            self._prompt_path = prompt_path

        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = DEFAULT_LLM_MODEL

    async def run(self, **query) -> str | dict:
        task = query["task"]

        prompt = self._prompt_template.render(task=task)

        logger.debug(pretty_prompt_text("Reasoning Prompt", prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            options={"temperature": 0.5},
        )
        return json.loads(response["message"]["content"])
