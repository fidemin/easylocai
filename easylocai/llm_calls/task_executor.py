from typing import Literal

from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    server_name: str = Field(
        description="The name of the MCP server providing the tool."
    )
    tool_name: str = Field(description="The name of the tool to execute.")
    tool_args: dict = Field(description="Arguments to pass to the tool.")


class ReasoningInput(BaseModel):
    input: str = Field(
        description="The input query for the reasoning agent. Should be specific to the subtask."
    )
    level: str = Field(description="The reasoning level")


class TaskExecutorInput(BaseModel):
    task: str
    user_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class TaskExecutorOutput(BaseModel):
    subtask: str | None = Field(
        title="Subtask",
        description="The specific subtask to execute. None if finished=True",
    )
    subtask_type: Literal["tool", "reasoning"] | None = Field(
        title="Subtask Type",
        description="The type of subtask: 'tool' for tool-based execution, 'reasoning' for reasoning agent. None if finished=True",
    )
    tool_input: ToolInput | None = Field(
        title="Tool Input",
        description="tool input to execute when subtask_type is 'tool'. None if subtask_type='reasoning' or finished=True",
    )
    reasoning_input: ReasoningInput | None = Field(
        title="Reasoning Input",
        description="Reasoning input for subtask_type is 'reasoning'. None if subtask_type='tool' or finished=True. Should not be None when subtask is not None and subtask_type is 'reasoning'.",
    )
    finished: bool = Field(
        description="Whether the task is completed. If subtask is not None, finished is False."
    )
    finished_reason: str | None = Field(
        description="Explanation of why the task is finished. None if finished is False."
    )
