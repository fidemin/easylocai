from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.context import ConversationHistory


class ReasoningInput(BaseModel):
    original_task: str = Field(
        title="Original Task",
        description="The parent task that this subtask belongs to.",
    )
    subtask: str = Field(
        title="Subtask",
        description="The subtask to reason about.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Preamble context extracted from the current query by QueryReformatter.",
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
    conversation_histories: list[ConversationHistory] = Field(
        default_factory=list,
        title="Conversation Histories",
        description="Prior conversation turns for multi-turn context.",
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
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
