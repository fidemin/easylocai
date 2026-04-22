---
name: Replanner prompt v2 optimization
description: Optimization results for the Replanner prompt; v2 passes 5/5 cases; key fix is returning plan tasks verbatim instead of inventing new ones
type: project
---

## Result

Replanner prompt v2 (`replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2` + `replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`) passes all 5 eval cases.

**Why:** Baseline (v1) failed `replan-multi-source-task` by inventing per-URL retrieval tasks instead of returning the 3 remaining tasks verbatim from PREVIOUS_PLAN. The model was also rephrasing tasks (embedding computed values like "Save the result 60...").

## Key changes in v2 (vs v1)

1. **Decision-tree ordering (Rule 1: check USER_CONTEXT first)** — explicit numbered rule to check USER_CONTEXT before doing anything else. Fixes replan-004 reliably.
2. **"Return remaining tasks from original plan verbatim"** — v2 explicitly tells the model to return only tasks NOT yet completed from PREVIOUS_PLAN, not to invent or rephrase. Fixes `replan-multi-source-task`.
3. **Numbered PREVIOUS_PLAN in user prompt** — `{{ loop.index }}. {{ task }}` makes plan enumeration clear.
4. **REMAINING_TASKS_FROM_PLAN section in user prompt** — Jinja2 pre-computes which tasks are not yet done and injects them explicitly. This is a strong cue for gpt-oss:20b.
5. **`response` must be null when tasks remain** — explicit constraint "NEVER set both tasks (non-empty) and response (non-null)". Fixes the occasional leak where v1 would set both.

## Baseline failure modes (v1)

- `replan-multi-source-task`: model splits single plan task into per-URL tasks (over-planning)
- `replan-002`: task slightly rephrased by embedding computed value ("Save the result 60 to a file")
- Both caused by: no explicit instruction to return plan tasks verbatim

## Eval config

- Baseline config: `resources/prompt_eval/configs/replan_prompt_config.json`
- v2 config: `resources/prompt_eval/configs/replan_prompt_v2_config.json`
- Input: `resources/prompt_eval/inputs/replan_prompt_inputs.json` (now includes `expected` + `scoring_criteria` fields)

**How to apply:** When optimizing any "identify remaining steps" prompt for gpt-oss:20b, pre-compute the remaining items in Jinja2 and inject them explicitly rather than asking the model to diff two lists itself.
