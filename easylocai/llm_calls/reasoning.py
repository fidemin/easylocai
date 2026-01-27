from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ReasoningInput(BaseModel):
    task: str = Field(
        title="Task",
        description="The task to reason about.",
    )
    user_context: str | None = Field(
        title="User Context",
        description="Additional context provided by the user.",
    )
    previous_task_results: list[dict] = Field(
        title="Previous Task Results",
        description="Results from previous tasks.",
    )
    previous_subtask_results: list[dict] = Field(
        default_factory=list,
        title="Previous Subtask Results",
        description="Results from previous subtasks within the current task.",
    )


class ReasoningOutput(BaseModel):
    reasoning: str = Field(
        title="Reasoning",
        description="The reasoning process.",
    )
    final: str = Field(
        title="Final",
        description="The final answer.",
    )
    confidence: int = Field(
        title="Confidence",
        description="Confidence level of the answer.",
    )


class Reasoning(LLMCallV2[ReasoningInput, ReasoningOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/reasoning_system_prompt.jinja2"
        user_prompt_path = "prompts/reasoning_user_prompt.jinja2"
        options = {
            "temperature": 0.5,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
