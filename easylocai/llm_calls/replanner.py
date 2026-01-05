from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.llm_call import LLMCall


class ReplannerInput(BaseModel):
    user_context: str | None
    original_user_query: str
    previous_plan: list[str]
    tool_candidates: list[dict]
    task_results: list[dict]


class ReplannerOutput(BaseModel):
    tasks: list[dict]
    response: str | None


class Replanner(LLMCall[ReplannerInput, ReplannerOutput]):
    def __init__(self, *, client: AsyncClient):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/replanner_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/replanner_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReplannerOutput,
            options=options,
        )
