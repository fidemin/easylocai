from pydantic import BaseModel

from easylocai.core.llm_call import LLMCall


class ToolInput(BaseModel):
    server_name: str
    tool_name: str
    tool_args: dict


class ToolSelectorInput(BaseModel):
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]
    original_tasks: list[dict]
    task: str
    user_context: str | None


class ToolSelectorOutput(BaseModel):
    subtask: str | None
    selected_tools: list[ToolInput] | None
    finished: bool
    finished_reason: str | None


class ToolSelector(LLMCall[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/tool_selector_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/tool_selector_user_prompt.jinja2"
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
