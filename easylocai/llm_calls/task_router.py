from typing import Literal

from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class TaskRouterInput(BaseModel):
    task: str
    user_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class SubTask(BaseModel):
    subtask: str = Field(
        description="The next subtask to execute. Do not refer to tool names directly here."
    )
    subtask_type: Literal["tool", "reasoning"] = Field(
        description="'tool' for tool-based execution, 'reasoning' for reasoning subtask (e.g. coding, problem solving, inference)."
    )


class TaskRouterOutput(BaseModel):
    subtask: str | None = Field(
        description="The next subtask to execute. None if finished=True. Do not refer to tool names directly here."
    )
    subtask_type: Literal["tool", "reasoning"] | None = Field(
        description="'tool' for tool-based execution, 'reasoning' for reasoning agent. None if finished=True"
    )
    finished: bool = Field(description="Whether the task is completed or failed.")
    finished_reason: str | None = Field(
        description="Explanation of why the task is finished. None if finished is False."
    )


class TaskRouterOutputV2(BaseModel):
    subtasks: list[SubTask] = Field(
        description="A list of next subtasks which can be executed in parallel to achieve the goal of task. Empty if finished=True."
    )
    finished: bool = Field(description="Whether the task is completed or failed.")
    finished_reason: str | None = Field(
        description="Explanation of why the task is finished. None if finished is False."
    )


class TaskRouter(LLMCallV2[TaskRouterInput, TaskRouterOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/task_router_system_prompt.jinja2"
        user_prompt_path = "prompts/task_router_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=TaskRouterOutput,
            options=options,
        )


class TaskRouterV2(LLMCallV2[TaskRouterInput, TaskRouterOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/task_router_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/task_router_user_prompt_v2.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=TaskRouterOutputV2,
            options=options,
        )
