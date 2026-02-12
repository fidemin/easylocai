from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class PlannerInput(BaseModel):
    user_query: str = Field(
        title="User Query",
        description="The user's objective/query that needs to be planned.",
    )
    user_context: str | None = Field(
        title="User Context",
        description="Additional context provided by the user.",
    )


class PlannerOutput(BaseModel):
    tasks: list[str] = Field(
        title="Tasks",
        description="A list of atomic, independent, simple, and semantic tasks",
    )


class Planner(LLMCallV2[PlannerInput, PlannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/planner_system_prompt.jinja2"
        user_prompt_path = "prompts/planner_user_prompt.jinja2"
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


class PlannerV2(LLMCallV2[PlannerInput, PlannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/planner_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/planner_user_prompt_v2.jinja2"
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
