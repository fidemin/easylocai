import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class AnswerAgent(Agent):
    _system_prompt_path = "resources/prompts/answer_system_prompt.txt"

    def __init__(
        self,
        *,
        client: AsyncClient,
        model: str,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._system_prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    async def run(self, query: str | dict) -> str | dict:
        user_query = query["user_query"]
        task_results = query["task_results"]
        user_context_list = query["user_context_list"]
        prompt = self._prompt_template.render(
            user_query=user_query,
            task_results=task_results,
            user_context_list=user_context_list,
        )
        logger.debug(pretty_prompt_text("answer prompt", prompt))
        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        return response["message"]["content"]
