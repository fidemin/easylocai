from typing import Any

from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class ToolInput(BaseModel):
    server_name: str = Field(description="The server_name of the selected tool")
    tool_name: str = Field(
        description="The tool_name selected based on subtask requirements"
    )
    tool_args: dict[str, Any] = Field(
        description="Parameters for tool execution following the tool's input_schema. Do not include extra parameters not defined in the tool's input_schema."
    )


class ToolSelectorV2Input(BaseModel):
    subtask: str
    user_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorOutput(BaseModel):
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why no tool was selected. None if a tool was successfully selected."
    )


class ToolSelectorInput(BaseModel):
    subtask: str
    user_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorInputV2(BaseModel):
    subtasks: list[str]
    user_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class SubtaskWithTool(BaseModel):
    subtask: str = Field(description="The actual subtask for which tool is selected.")
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why tool can not be selected. None if a tool was successfully selected."
    )


class ToolSelectorOutputV2(BaseModel):
    results: list[SubtaskWithTool]


class ToolSelector(LLMCallV2[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/tool_selector_system_prompt.jinja2"
        user_prompt_path = "prompts/tool_selector_user_prompt.jinja2"
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


class ToolSelectorV2(LLMCallV2[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/tool_selector_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/tool_selector_user_prompt_v2.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ToolSelectorOutputV2,
            options=options,
        )
