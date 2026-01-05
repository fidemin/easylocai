import json
import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from easylocai.core.agent import Agent
from easylocai.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class NextPlanAgent(Agent):
    _prompt_path = "resources/prompts/next_plan_prompt.txt"

    def __init__(
        self,
        *,
        client: AsyncClient,
        model: str,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    async def run(self, query: str | dict) -> str | dict:
        original_user_query = query["original_user_query"]
        previous_task_results: list[dict] = query["previous_task_results"]
        user_context_list: list[str] = query["user_context_list"]

        prompt = self._prompt_template.render(
            previous_task_results=previous_task_results,
            user_context_list=user_context_list,
        )
        logger.debug(pretty_prompt_text("Next Plan Prompt", prompt))
        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": original_user_query},
            ],
            options={"temperature": 0.5},
        )
        return json.loads(response["message"]["content"])
