from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL


class ReplanEvalAgent(Agent):
    _prompt_path = "resources/prompts/v2/replan_eval_prompt.jinja2"

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
        task_results = query["task_results"]
        tasks = query["tasks"]
        response = query["response"]

        prompt = self._prompt_template.render()

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"Original User Query: {original_user_query}\nOriginal Plan: {original_plan}\nPrevious Step Results: {task_results}\nNew Plan: {tasks}\nResponse: {response}",
                },
            ],
            options={"temperature": 0.3},
        )
        return response["message"]["content"]
