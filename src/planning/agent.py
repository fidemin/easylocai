from jinja2 import Environment, FileSystemLoader
from ollama import Client

from src.core.agent import Agent
from src.utlis.prompt import print_prompt


class PlanningAgent(Agent):
    _prompt_path = "resources/prompts/planning_prompt.txt"

    def __init__(
        self,
        *,
        client: Client,
        model: str,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    def _chat(self, text: str):
        prompt = self._prompt_template.render()
        print_prompt("Planning prompt", prompt)
        response = self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        return response["message"]["content"]
