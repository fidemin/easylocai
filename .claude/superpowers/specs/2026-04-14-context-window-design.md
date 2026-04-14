# Context Window 개선 Design

**Date:** 2026-04-14  
**Ticket:** [Context Window 개선 #28](https://www.notion.so/3420b3fd6bcc8018b49fc2f917c673f4)  
**Status:** Approved

---

## 배경

현재 agent의 context가 각 Agent Input/Output에 개별 파라미터로 흩어져 있어 중복이 발생하고 관리가 어렵다. 계층적 Context 객체를 도입해 데이터 흐름을 명확히 하고 재사용성을 높인다.

---

## Context 스키마

위치: `easylocai/schemas/context.py` (신규)

### GlobalContext

conversation 간 persist되는 최상위 context. `main.py`가 생성하고 소유한다.

```python
class ConversationHistory(BaseModel):
    original_user_query: str
    reformatted_user_query: str
    query_context: str | None = None
    response: str

class GlobalContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []
```

### WorkflowContext

단일 user query의 workflow 전체에 걸쳐 사용되는 context. `workflow.run()` 시작 시 생성되고, `EasylocaiWorkflow`가 업데이트 책임을 가진다.

```python
class ExecutedTaskResult(BaseModel):
    executed_task: str
    result: str

class WorkflowContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []  # GlobalContext에서 복사
    original_user_query: str
    query_context: str | None = None           # PlanAgent 이후 채워짐
    reformatted_user_query: str | None = None  # PlanAgent 이후 채워짐
    task_list: list[str] = []                  # PlanAgent 이후 채워짐, Replanner가 업데이트 가능
    executed_task_results: list[ExecutedTaskResult] = []  # SingleTaskAgent 결과가 쌓임
```

### SingleTaskAgentContext

`SingleTaskAgent` 내부 전용 context. `WorkflowContext`의 모든 필드를 포함하고 agent 전용 내부 필드를 추가한다. Agent 결과 반환 시 외부에 노출되지 않는다.

```python
class SingleTaskAgentContext(BaseModel):
    # WorkflowContext 필드 전체 (workflow.py가 복사해서 전달)
    conversation_histories: list[ConversationHistory] = []
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = []
    executed_task_results: list[ExecutedTaskResult] = []

    # SingleTaskAgent 전용 내부 필드
    original_task: str
    subtask_results: list[SubtaskResult] = []

# SubtaskResult는 현재 {"subtask": str, "result": str} 구조를 따름
class SubtaskResult(BaseModel):
    subtask: str
    result: str
```

`subtask_tools`는 context에 포함하지 않는다. subtask 실행 시 매번 fresh fetch하며 로컬 변수로만 사용한다.

---

## 설계 원칙

### 독립적인 flat 모델

각 Context는 독립적인 Pydantic 모델이며 Python class 상속이나 composition(nested 필드)을 사용하지 않는다. "상속"은 개념적 관계이며, 실제로는 필요한 필드를 각 Context에 명시적으로 정의한다. 이는 향후 특정 필드를 특정 Context에서만 제외하거나 추가할 때 유연하게 대응하기 위함이다.

### Context 업데이트는 소유자가 담당

각 Context의 업데이트 책임은 해당 Context를 소유하는 계층이 가진다:
- `GlobalContext` 업데이트: `main.py` (또는 `workflow.run()` 완료 후 caller)
- `WorkflowContext` 업데이트: `EasylocaiWorkflow`
- `SingleTaskAgentContext` 업데이트: `SingleTaskAgent` 내부

Agent는 결과 데이터만 반환하며 context 업데이트 책임을 갖지 않는다.

### Jinja 프롬프트 방식 유지

LLM call의 user prompt는 현행 Jinja2 템플릿 방식을 유지하며, Context 객체의 필드를 템플릿 변수로 전달한다.

---

## 데이터 흐름

```
main.py
  global_context = GlobalContext()

  workflow.run(user_query, global_context)
    │
    ├─ WorkflowContext 생성
    │   ├─ conversation_histories ← global_context.conversation_histories
    │   └─ original_user_query ← user_query
    │
    ├─ PlanAgent.run(user_query, conversation_histories)
    │   └─ WorkflowContext 업데이트:
    │       query_context, reformatted_user_query, task_list
    │
    ├─ [loop]
    │   ├─ SingleTaskAgentContext 생성
    │   │   └─ WorkflowContext 필드 전체 복사 + original_task
    │   ├─ SingleTaskAgent.run(single_task_agent_context)
    │   │   └─ SingleTaskAgentOutput(executed_task, result) 반환
    │   └─ workflow_context.executed_task_results에 추가
    │
    ├─ ReplanAgent.run(workflow_context 관련 필드)
    │   └─ WorkflowContext.task_list 업데이트 or 최종 응답 반환
    │
    └─ GlobalContext 업데이트:
        conversation_histories에 ConversationHistory 추가
        (original_user_query, reformatted_user_query, query_context, response)
```

---

## 구현 전략

기존 `main` workflow를 수정하지 않는다. 새로운 workflow variant `contextimprove`를 별도로 생성해 적용한다.

- Workflow 파일: `easylocai/workflow_contextimprove.py`
- Runner 함수: `easylocai/main_contextimprove.py`
- 실행: `easylocai --flag=contextimprove`
- `workflow_registry`에 등록: `"contextimprove": run_agent_workflow_contextimprove`

---

## 변경 범위

| 파일 | 변경 내용 |
|------|----------|
| `easylocai/schemas/context.py` | 신규: `GlobalContext`, `WorkflowContext`, `SingleTaskAgentContext`, `ConversationHistory`, `ExecutedTaskResult`, `SubtaskResult` |
| `easylocai/workflow_contextimprove.py` | 신규: `GlobalContext` 수신, `WorkflowContext` 생성/업데이트 로직 포함한 새 workflow |
| `easylocai/main_contextimprove.py` | 신규: `contextimprove` runner 함수 |
| `easylocai/main.py` | `workflow_registry`에 `"contextimprove"` 등록 |
| `easylocai/agents/plan_agent_contextimprove.py` | 신규: `WorkflowContext` 기반 PlanAgent |
| `easylocai/agents/single_task_agent_contextimprove.py` | 신규: `SingleTaskAgentContext` 기반 SingleTaskAgent |
| `easylocai/agents/replan_agent_contextimprove.py` | 신규: `WorkflowContext` 기반 ReplanAgent |
