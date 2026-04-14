from pydantic import BaseModel


class ConversationHistory(BaseModel):
    original_user_query: str
    reformatted_user_query: str
    query_context: str | None = None
    response: str


class GlobalContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []


class ExecutedTaskResult(BaseModel):
    executed_task: str
    result: str


class SubtaskResult(BaseModel):
    subtask: str
    result: str


class WorkflowContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = []
    executed_task_results: list[ExecutedTaskResult] = []


class SingleTaskAgentContext(BaseModel):
    # WorkflowContext 필드 전체
    conversation_histories: list[ConversationHistory] = []
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = []
    executed_task_results: list[ExecutedTaskResult] = []
    # SingleTaskAgent 전용
    original_task: str
    subtask_results: list[SubtaskResult] = []
