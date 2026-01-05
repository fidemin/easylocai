import logging

from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from easylocai.core.agent import Agent
from easylocai.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class ToolResultFilterAgent(Agent):
    _system_prompt_path = "resources/prompts/tool_result_prompt.txt"

    def __init__(
        self,
        *,
        client: AsyncClient,
        model: str,
    ):
        self._client = client
        self._model = model

        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._system_prompt_path)
        self._prompt_template = prompt_template

    async def run(self, **kwargs) -> str | dict:
        user_query = kwargs["user_query"]
        task = kwargs["task"]
        tool_result = kwargs["tool_result"]

        prompt = self._prompt_template.render(
            {"user_query": user_query, "task": task, "tool_result": tool_result}
        )

        logger.debug(pretty_prompt_text("Tool Filter Prompt", prompt))

        tool_result_str = (
            "TOOL RESULT:\n" + tool_result
            if isinstance(tool_result, str)
            else str(tool_result)
        )

        response = await self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": tool_result_str,
                },
            ],
        )
        return response["message"]["content"]
