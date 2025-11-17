import json
import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class ReplanAgent(Agent):
    _prompt_path = "resources/prompts/v2/replan_prompt.jinja2"

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
        original_user_query = query["original_user_query"]
        original_plan = query["original_plan"]
        step_results = query["step_results"]

        prompt = self._prompt_template.render(
            original_user_query=original_user_query,
            original_plan=original_plan,
            step_results=step_results,
        )
        logger.debug(pretty_prompt_text("Replan Prompt", prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            options={"temperature": 0.0},
        )
        return json.loads(response["message"]["content"])
