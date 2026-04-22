from pydantic import BaseModel
from pydantic import RootModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class TaskResultFilterInput(BaseModel):
    task: str
    subtask_results: list[dict]
    query_context: str | None


class TaskResultFilterOutput(RootModel[str]):
    root: str = Field(description="Filtered task result")


class TaskResultFilter(LLMCallV2[TaskResultFilterInput, TaskResultFilterOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/task_result_filter_system_prompt.jinja2"
        user_prompt_path = "prompts/task_result_filter_user_prompt.jinja2"
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
