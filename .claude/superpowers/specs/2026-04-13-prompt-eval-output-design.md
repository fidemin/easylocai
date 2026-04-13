# PromptEvalWorkflow Output 개선 — Design Spec

**Date:** 2026-04-13
**Ticket:** #27
**Status:** Approved

## Context

`PromptEvalWorkflow`는 현재 각 테스트 케이스의 전체 Chat Input + Response를 그대로 출력하고, 전체 통계나 파일 저장 기능이 없다. prompt-eval 스킬이 이 출력을 보고 수동으로 scoring하는데, 가독성이 낮고 결과를 저장할 방법이 없다.

## Goals

1. stdout 출력을 Markdown 테이블로 변경 (Chat Input 생략)
2. 전체 통계를 stdout 마지막에 항상 출력
3. `--output <path>` 옵션으로 결과 파일 저장 (선택적)
4. `--format text|json` 으로 파일 형식 선택 (기본값: text/Markdown)

## CLI Interface

```bash
# stdout만 (기본)
python -m prompt_eval.run <config>

# Markdown 파일로 저장
python -m prompt_eval.run <config> --output results.md

# JSON 파일로 저장
python -m prompt_eval.run <config> --output results.json --format json
```

## Output Format

### stdout (항상 출력)

```
| ID | Response | Expected | Thinking |
|----|----------|----------|----------|
| 1  | ...      | ...      | ...      |

Config:        resources/prompt_eval/configs/plan_prompt_v2_config.json
System prompt: resources/prompts/planner_system_prompt.jinja2
User prompt:   resources/prompts/planner_user_prompt.jinja2
Input file:    resources/prompt_eval/inputs/plan_prompt_inputs.json
Output model:  easylocai.llm_calls.planner.PlannerOutput
Model:         gpt-oss:20b
Total:         8 cases
```

- Chat Input(프롬프트 전문)은 생략
- Thinking이 없는 경우 빈 칸
- 긴 텍스트는 적절히 truncate

### text 파일 (Markdown)

```markdown
## Eval Results — plan_prompt_v2_config — 2026-04-13

| ID | Response | Expected | Thinking |
|----|----------|----------|----------|
| 1  | ...      | ...      | ...      |

## Summary
Config:        resources/prompt_eval/configs/plan_prompt_v2_config.json
System prompt: resources/prompts/planner_system_prompt.jinja2
User prompt:   resources/prompts/planner_user_prompt.jinja2
Input file:    resources/prompt_eval/inputs/plan_prompt_inputs.json
Output model:  easylocai.llm_calls.planner.PlannerOutput
Model:         gpt-oss:20b
Total:         8 cases
```

### json 파일

```json
{
  "metadata": {
    "config": "resources/prompt_eval/configs/plan_prompt_v2_config.json",
    "system_prompt": "resources/prompts/planner_system_prompt.jinja2",
    "user_prompt": "resources/prompts/planner_user_prompt.jinja2",
    "input_file": "resources/prompt_eval/inputs/plan_prompt_inputs.json",
    "output_model": "easylocai.llm_calls.planner.PlannerOutput",
    "model": "gpt-oss:20b",
    "date": "2026-04-13"
  },
  "results": [
    {
      "id": "1",
      "response": "...",
      "expected": "...",
      "thinking": "..."
    }
  ],
  "summary": {
    "total": 8
  }
}
```

## Changes

### `prompt_eval/prompt_eval_workflow.py`

- `run(output_file, output_format)` 시그니처 추가
- `_print()` → `_print_table()` 로 교체 (Markdown 테이블 출력)
- `_print_summary()` 추가
- `_save(results, output_file, output_format)` 추가
  - `text`: Markdown 파일 저장
  - `json`: JSON 파일 저장

### `prompt_eval/run.py`

- `--output <path>` 인자 파싱 추가
- `--format text|json` 인자 파싱 추가 (기본값: `text`)
- `run()` 함수에 `output_file`, `output_format` 전달

## Non-Goals

- Match/Pass-fail 자동 판단 (semantic scoring은 prompt-eval 스킬이 담당)
- 기존 `run_and_collect()` 인터페이스 변경 없음
