import json
import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class PlanAgent(Agent):
    _system_prompt_path = "resources/prompts/v2/plan_system_prompt.jinja2"
    _user_prompt_path = "resources/prompts/v2/plan_user_prompt.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
        prompt_path: str | None = None,
    ):
        if prompt_path is not None:
            self._system_prompt_path = prompt_path

        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        system_prompt_template = env.get_template(self._system_prompt_path)
        self._system_prompt_template = system_prompt_template

        user_prompt_template = env.get_template(self._user_prompt_path)
        self._user_prompt_template = user_prompt_template

        self._model = DEFAULT_LLM_MODEL

    async def run(self, **query) -> str | dict:
        user_query = query["user_query"]

        system_prompt = self._system_prompt_template.render()
        logger.debug(pretty_prompt_text("Plan System Prompt", system_prompt))

        user_prompt = self._user_prompt_template.render(user_query=user_query)
        logger.debug(pretty_prompt_text("Plan User Prompt", user_prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.2},
        )
        return json.loads(response["message"]["content"])
