---
name: prompt-eval
description: Run prompt evaluation for an easylocai component, score each result against expected answers, and report a pass/fail summary. Use this skill whenever the user asks to "run prompt eval", "evaluate a prompt", "score a prompt", "test prompt quality", or mentions a specific component like "eval the planner prompt" or "run prompt-eval for task_router".
---

# Prompt Eval Skill

Runs the prompt evaluation test suite for a given component, compares each LLM output against the expected answer defined in the input JSON, scores every test case, and prints a summary report.

## Config Files

Each prompt set has its own config file in `resources/prompt_eval/configs/`. Input data lives separately in `resources/prompt_eval/inputs/`. Multiple configs can share the same input file (e.g. v1 and v2 variants).

| Config file | Input file |
|-------------|------------|
| `plan_prompt_config.json` | `plan_prompt_inputs.json` |
| `plan_prompt_v2_config.json` | `plan_prompt_inputs.json` |
| `task_router_prompt_config.json` | `task_router_prompt_inputs.json` |
| `task_router_prompt_v2_config.json` | `task_router_prompt_inputs_v2.json` |
| `tool_select_prompt_config.json` | `tool_selector_prompt_inputs.json` |
| `tool_select_prompt_v2_config.json` | `tool_selector_prompt_inputs_v2.json` |
| `replan_prompt_config.json` | `replan_prompt_inputs.json` |
| `replan_prompt_v2_config.json` | `replan_prompt_inputs.json` |
| `query_reformatter_prompt_config.json` | `query_reformatter_prompt_inputs.json` |
| `task_result_filter_prompt_config.json` | `task_result_prompt_inputs.json` |
| `subtask_result_prompt_config.json` | `subtask_result_prompt_inputs.json` |

Config file format:
```json
{
  "input_file": "resources/prompt_eval/inputs/plan_prompt_inputs.json",
  "prompt_info": {
    "system": "resources/prompts/planner_system_prompt_v2.jinja2",
    "user": "resources/prompts/planner_user_prompt_v2.jinja2"
  },
  "output_model": "easylocai.llm_calls.planner.PlannerOutput"
}
```
`model_info` is optional — defaults to `gpt-oss:20b` at `localhost:11434` with `temperature: 0.2`.

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
python -m prompt_eval.run <config_file>
```
Example: `python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json`

Capture the full stdout. Each test case is delimited by:
```
----- Prompt Eval Result (ID: <id>) -----
...
-------------------------
```

### Step 3 — Read input JSON for expected answers

Read the `input_file` from the config to get `expected` and `scoring_criteria` for each test case (matched by `id`).

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
- To add a new prompt set, create a config in `resources/prompt_eval/configs/` and input data in `resources/prompt_eval/inputs/`. No Python changes needed.
- Scoring is semantic: "2 tasks vs 1 task" matters; exact wording does not.