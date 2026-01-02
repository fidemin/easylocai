from pydantic import BaseModel

from src.core.llm_call import LLMCall


class ToolSelectorInput(BaseModel):
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]
    original_tasks: list[dict]
    task: str


class ToolSelectorOutput(BaseModel):
    subtask: str | None
    server_name: str | None
    tool_name: str | None
    tool_args: dict | None
    finished: bool
    finished_reason: str | None


class ToolSelector(LLMCall[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "resources/prompts/v2/tool_selector_system_prompt.jinja2"
        user_prompt_path = "resources/prompts/v2/tool_selector_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ToolSelectorOutput,
            options=options,
        )
