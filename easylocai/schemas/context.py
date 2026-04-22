from pydantic import BaseModel, Field


class ConversationHistory(BaseModel):
    original_user_query: str
    reformatted_user_query: str
    query_context: str | None = None
    response: str


class GlobalContext(BaseModel):
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)


class ExecutedTaskResult(BaseModel):
    executed_task: str
    result: str


class SubtaskResult(BaseModel):
    subtask: str
    result: str


class WorkflowContext(BaseModel):
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = Field(default_factory=list)
    executed_task_results: list[ExecutedTaskResult] = Field(default_factory=list)


class SingleTaskAgentContext(BaseModel):
    # All WorkflowContext fields
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = Field(default_factory=list)
    executed_task_results: list[ExecutedTaskResult] = Field(default_factory=list)
    # SingleTaskAgent-specific fields
    original_task: str
    subtask_results: list[SubtaskResult] = Field(default_factory=list)
