from ollama import AsyncClient
from pydantic import BaseModel

from src.core.llm_call import LLMCall


class PlannerInput(BaseModel):
    user_query: str
    user_context: str | None


class PlannerOutput(BaseModel):
    tasks: list[str]


class Planner(LLMCall[PlannerInput, PlannerOutput]):
    def __init__(self, *, client: AsyncClient):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/planner_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/planner_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=PlannerOutput,
            options=options,
        )
