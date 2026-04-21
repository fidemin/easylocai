# Prompt Context Gap Analysis

**Date:** 2026-04-21
**Scope:** 각 LLM Call prompt template이 실제로 사용하는 context 변수 분석 및 gap 식별

---

## 섹션 1: Template 변수 vs 사용 가능한 Context

`WorkflowContext` / `SingleTaskAgentContext`에서 사용 가능한 필드:
- `conversation_histories`: 이전 대화 히스토리 (user query + assistant response)
- `original_user_query`: 원본 쿼리
- `query_context`: QueryReformatter가 추출한 현재 쿼리 preamble (대부분 null)
- `reformatted_user_query`: pronoun 해소된 쿼리
- `task_list`: 전체 plan task 목록
- `executed_task_results`: 완료된 task 결과들
- `original_task`: 현재 실행 중인 task (SingleTaskAgentContext)
- `subtask_results`: 현재 task의 subtask 결과들

| LLM Call | Template에서 실제 사용하는 변수 | 사용 가능하지만 미전달 (Gap) |
|---|---|---|
| QueryReformatter | `user_query`, `previous_conversations` | — |
| Planner | `user_query`, `user_context`(`query_context`) | **`conversation_histories`** |
| TaskRouter | `task`, `user_context`, `tool_candidates`, `previous_task_results`, `iteration_results` | `conversation_histories`, `task_list` |
| ToolSelector | `subtask`, `user_context`, `tool_candidates`, `previous_task_results`, `iteration_results` | `original_task`, `conversation_histories` |
| Reasoning | `subtask`(현재 `task`로 잘못 명명), `user_context`, `previous_task_results`, `previous_subtask_results` | `original_task`, `conversation_histories` |
| SubtaskResultFilter | `subtask`, `result` | — (의도적으로 최소화) |
| TaskResultFilter | `task`, `user_context`, `subtask_results` | — |
| Replanner | `user_context`, `original_user_query`, `previous_plan`, `task_results` | **`conversation_histories`** |

---

## 섹션 2: Gap 우선순위 및 영향도

### 🔴 High

**Planner — `conversation_histories` 누락**

멀티턴 대화에서 사용자가 이전 결과를 기반으로 후속 쿼리를 할 때 문제 발생. QueryReformatter는 pronoun만 해소하고 `query_context=null`로 설정하기 때문에, Planner는 이전 대화에서 무슨 일이 있었는지 모른 채 처음부터 plan을 작성하게 됨.

예: "alice.txt 찾아줘" → (찾았음) → "그 파일 요약해줘" → Planner가 "alice.txt 읽기" 태스크를 다시 생성

**Replanner — `conversation_histories` 누락**

최종 응답을 결정할 때 이전 대화 맥락이 없어서, 이전 턴에서 이미 답한 내용을 반영하지 못함.

---

### 🟡 Medium

**ToolSelector / Reasoning — `original_task` 누락**

subtask만 보고 tool args를 설정하거나 reasoning을 수행. 부모 task 맥락 없이 작동하므로 subtask가 모호할 때 판단이 어려울 수 있음. `user_context`와 `previous_task_results`로 어느 정도 커버되지만 완전하지 않음.

**내부 prompt 변수명 통일 — `user_context` → `query_context`**

모든 prompt template에서 `user_context`라는 변수명을 사용하지만, 실제로는 QueryReformatter가 추출한 `query_context` (현재 쿼리 preamble만, 대부분 null)임. "user context"라는 이름이 conversation history나 broader user intent를 포함하는 것처럼 오해를 유발.

변경 대상:
- `planner_user_prompt.jinja2`
- `task_router_user_prompt.jinja2`
- `tool_selector_user_prompt.jinja2`
- `reasoning_user_prompt.jinja2`
- `task_result_filter_user_prompt.jinja2`
- `replanner_user_prompt.jinja2`
- 대응하는 system prompt의 설명 텍스트
- 각 `LLMCallV2` input model의 `user_context` 필드

**Reasoning prompt 변수명 — `task` → `subtask`**

`reasoning_user_prompt.jinja2`의 `{{ task }}`는 실제로 subtask가 전달됨. ToolSelector의 `{{ subtask }}`와 일관성이 없고, template만 보면 parent task인지 subtask인지 구분 불가.

변경 대상:
- `reasoning_user_prompt.jinja2`: `{{ task }}` → `{{ subtask }}`
- `reasoning_system_prompt.jinja2`: 설명 텍스트 내 `TASK` → `SUBTASK`
- `ReasoningInput` 모델의 `task` 필드 → `subtask`
- `ReasoningAgent`에서 input 생성 시 필드명 변경

---

### 🟢 Low

**TaskRouter — `task_list` 누락**

현재 task가 전체 plan에서 어느 위치인지 모름. routing 결정에 영향을 줄 수 있으나 실제 오작동 사례는 제한적.

**TaskRouter / ToolSelector / Reasoning — `conversation_histories` 누락**

단일 턴에서는 영향 없음. 멀티턴에서도 이 레이어들은 `previous_task_results`로 충분히 커버되는 경우가 많아 실제 오작동은 제한적.

---

## 구현 우선순위 요약

| 우선순위 | 작업 | 관련 파일 |
|---|---|---|
| 1 | Planner에 `conversation_histories` 추가 | `planner_user_prompt.jinja2`, `planner.py`, `plan_agent.py` |
| 2 | Replanner에 `conversation_histories` 추가 | `replanner_user_prompt.jinja2`, `replanner.py`, `replan_agent.py` |
| 3 | 전체 prompt `user_context` → `query_context` 리네임 | 모든 prompt template + LLM call input models |
| 4 | Reasoning `task` → `subtask` 리네임 | `reasoning_*.jinja2`, `reasoning.py`, `reasoning_agent.py` |
| 5 | ToolSelector / Reasoning에 `original_task` 추가 | 해당 prompt template + input models |
