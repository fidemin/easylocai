from pydantic import BaseModel
from pydantic import RootModel, Field

from easylocai.core.llm_call import LLMCall


class TaskResultFilterInput(BaseModel):
    original_user_query: str
    task: str
    iteration_results: list[dict]
    user_context: str | None


class TaskResultFilterOutput(RootModel[str]):
    root: str = Field(description="Filtered task result")


class TaskResultFilter(LLMCall[TaskResultFilterInput, TaskResultFilterOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/task_result_filter_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/task_result_filter_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=TaskResultFilterOutput,
            options=options,
        )
