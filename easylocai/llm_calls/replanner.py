from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ReplannerV2Input(BaseModel):
    user_context: str | None
    original_user_query: str
    previous_plan: list[str]
    task_results: list[dict]


class ReplannerV2Output(BaseModel):
    tasks: list[str] = Field(
        description="A list of remaining tasks to complete the user query"
    )
    response: str | None = Field(
        description="Final response to the user if all tasks are completed. None if tasks remain."
    )


class ReplannerV2(LLMCallV2[ReplannerV2Input, ReplannerV2Output]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/replanner_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/v2/replanner_user_prompt_v2.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReplannerV2Output,
            options=options,
        )
