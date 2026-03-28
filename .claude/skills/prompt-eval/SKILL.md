---
name: prompt-eval
description: Run prompt evaluation for an easylocai component, score each result against expected answers, and report a pass/fail summary. Use this skill whenever the user asks to "run prompt eval", "evaluate a prompt", "score a prompt", "test prompt quality", or mentions a specific component like "eval the planner prompt" or "run prompt-eval for task_router".
---

# Prompt Eval Skill

Runs the prompt evaluation test suite for a given component, compares each LLM output against the expected answer defined in the input JSON, scores every test case, and prints a summary report.

## Component Map

| Component | Eval module | Input JSON | System prompt | User prompt |
|-----------|-------------|------------|---------------|-------------|
| `plan` | `prompt_eval.plan_prompt_eval_v2` | `resources/prompt_eval/plan_prompt_inputs.json` | `resources/prompts/planner_system_prompt_v2.jinja2` | `resources/prompts/planner_user_prompt_v2.jinja2` |
| `task_router` | `prompt_eval.task_router_prompt_eval_v2` | `resources/prompt_eval/task_router_prompt_inputs_v2.json` | `resources/prompts/task_router_system_prompt_v2.jinja2` | `resources/prompts/task_router_user_prompt_v2.jinja2` |
| `tool_select` | `prompt_eval.tool_select_prompt_eval_v2` | `resources/prompt_eval/tool_selector_prompt_inputs_v2.json` | `resources/prompts/tool_selector_system_prompt_v2.jinja2` | `resources/prompts/tool_selector_user_prompt_v2.jinja2` |
| `replan` | `prompt_eval.replan_prompt_eval_v2` | `resources/prompt_eval/replan_prompt_inputs.json` | `resources/prompts/replanner_system_prompt_v2.jinja2` | `resources/prompts/replanner_user_prompt_v2.jinja2` |
| `query_reformatter` | `prompt_eval.query_reformatter_prompt_eval` | `resources/prompt_eval/query_reformatter_prompt_inputs.json` | `resources/prompts/query_reformatter_system_prompt.jinja2` | `resources/prompts/query_reformatter_user_prompt.jinja2` |
| `task_result_filter` | `prompt_eval.task_result_filter_prompt_eval` | `resources/prompt_eval/task_result_prompt_inputs.json` | `resources/prompts/task_result_filter_system_prompt.jinja2` | `resources/prompts/task_result_filter_user_prompt.jinja2` |
| `subtask_result` | `prompt_eval.subtask_result_prompt_eval` | `resources/prompt_eval/subtask_result_prompt_inputs.json` | `resources/prompts/subtask_result_filter_system_prompt.jinja2` | `resources/prompts/subtask_result_filter_user_prompt.jinja2` |

## Input JSON Format

Each test case in the input JSON supports:
```json
{
  "id": "1",
  "messages": [ ... ],
  "expected": { "tasks": ["..."] },
  "scoring_criteria": "Describe what to check when scoring this case."
}
```

- `expected`: mirrors the component's output model structure (e.g. `{"tasks": [...]}` for planner, `{"subtasks": [...], "finished": bool}` for task_router)
- `scoring_criteria`: natural language guidance for scoring edge cases
- Both fields are optional — if absent, score qualitatively based on general output quality

## Workflow

### Step 1 — Determine component

Map the user's argument to the table above. If unclear, list available components and ask.

### Step 2 — Run the eval

From the project root:
```bash
python -m <eval_module>
```
Example: `python -m prompt_eval.plan_prompt_eval_v2`

Capture the full stdout. Each test case is delimited by:
```
----- Prompt Eval Result (ID: <id>) -----
...
-------------------------
```

### Step 3 — Read input JSON for expected answers

Read the input JSON file for the component to get `expected` and `scoring_criteria` for each test case (matched by `id`).

### Step 4 — Score each test case

For each test case, compare the actual response to `expected`. Score on three dimensions:

| Dimension | What to check |
|-----------|---------------|
| **Correctness** (1–5) | Does the output match the expected answer semantically? Consider `scoring_criteria` when present. |
| **Completeness** (1–5) | Are all required fields present? Are all expected items included? |
| **Format** (pass/fail) | Does the output parse as valid JSON matching the output schema? |

A test case **passes** if Correctness ≥ 4 AND Completeness ≥ 4 AND Format = pass.

When `expected` is missing, score qualitatively: assess whether the output makes sense given the input.

### Step 5 — Print results

For each test case print:
```
[ID: <id>] <PASS|FAIL>
  Input: <brief summary of user input>
  Expected: <expected answer>
  Actual:   <actual response>
  Correctness: <score>/5 — <one-line reason>
  Completeness: <score>/5 — <one-line reason>
  Format: PASS/FAIL
```

Then print a summary:
```
=== Eval Summary: <component> ===
Passed: X / Y
Average correctness: X.X
Average completeness: X.X

Failed cases: <list of IDs with brief reason>
```

### Step 6 — Offer next steps

After the report, offer:
- "Add more test cases to the input JSON"
- "Edit the prompt to fix failing cases" (read the system prompt template, propose edits, apply with Edit tool, re-run eval)

## Notes

- Always run from the project root. Eval scripts use relative paths.
- `PromptEvalWorkflow.run_and_collect()` returns `{id, messages, response, thinking, expected, scoring_criteria}` — use this if you need to call the workflow from a Python snippet instead of running the script.
- Scoring is semantic: "2 tasks vs 1 task" matters; exact wording does not.