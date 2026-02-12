from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class ReplannerInput(BaseModel):
    user_context: str | None
    original_user_query: str
    previous_plan: list[str]
    task_results: list[dict]


class ReplannerOutput(BaseModel):
    tasks: list[str] = Field(
        description="A list of remaining tasks to complete the user query"
    )
    response: str | None = Field(
        description="Final response to the user if all tasks are completed. None if tasks remain."
    )


class Replanner(LLMCallV2[ReplannerInput, ReplannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/replanner_system_prompt.jinja2"
        user_prompt_path = "prompts/replanner_user_prompt.jinja2"
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


class ReplannerV2(LLMCallV2[ReplannerInput, ReplannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/replanner_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/replanner_user_prompt_v2.jinja2"
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
