---
name: Planner v2 prompt optimization results
description: Optimization history, failure modes, and winning rules for planner_system_prompt.jinja2
type: project
---

## Optimization run: planner_system_prompt_v2.jinja2

**Date:** 2026-03-28
**Best candidate:** `resources/prompts/planner_system_prompt_v2_candidate_v4.jinja2`
**Result:** 8/8 pass (converged at iteration 4)

## Baseline failures (original prompt, 6/8)

- **ID 4** (simple factual query): Model added a "present the answer" second task and split "Korea" into South/North Korea. Root cause: no rule against "present the answer" tasks; no guidance for single-task factual queries.
- **ID 6** (pure generation task): Model over-decomposed "Generate a weekly meal plan" into 4 sub-tasks (research + template + populate + format). Root cause: no rule preventing decomposition of pure generation tasks.

## Iteration history

| Iter | Candidate | Score | Change | New Failure |
|------|-----------|-------|--------|-------------|
| Baseline | original | 6/8 | — | ID 4 (factual over-split), ID 6 (generation over-decomposed) |
| 1 | candidate_v1 | 6/8 | Added factual=single-task + generation=single-task rules | ID 1 regression (read+summarize collapsed to 1), ID 8 regression (calc+save merged) |
| 2 | candidate_v2 | 7/8 | Added "pure generation" qualifier + "data-acquisition separate from processing" + "no present-answer task" | ID 2 regression (read task added despite USER_CONTEXT having data) |
| 3 | candidate_v3 | 7/8 | Added "skip data-acquisition if already in USER_CONTEXT" | ID 3 regression (Notion search over-split to 4 tasks) |
| 4 | candidate_v4 | 8/8 | Replaced open-ended "separate" rule with "exactly two tasks" rule specifying data-acquisition covers all fetching and processing covers all downstream work | None |

## Winning rules added in candidate_v4

1. **USER_CONTEXT skip rule** (explicit bullet): "If USER_CONTEXT already contains the data needed, skip the data-acquisition task entirely and proceed directly to the processing task."

2. **Calculation + save separation** (added explicit prohibition): "Do NOT merge the calculation and the file-save into a single task."

3. **Factual query = single task**: "If the user query is a simple factual or knowledge question, produce a single task to answer it. Do NOT add a separate 'present the answer' or 'retrieve more information' task."

4. **Pure generation = single task**: "If the user query is a pure generation or creative task with no required data-acquisition step and no required file I/O step, produce a single task. Do NOT decompose it into sub-tasks like 'research', 'create template', 'populate', 'format'."

5. **Two-task rule for external data**: "When a task requires fetching external data and that data is NOT already in USER_CONTEXT, produce exactly two tasks: one to acquire the data (covers all fetching — do NOT split further) and one to process/use it (covers all downstream work — do NOT split further)."

6. **No presentation task**: "Do NOT add a 'present the answer to the user' or 'format the output' task."

## Key gpt-oss:20b behavior patterns

- Model tends to add "present the answer to the user" tasks when not explicitly prohibited.
- Model over-decomposes creative/generation tasks into research + template + populate + format unless explicitly prohibited.
- Model applies "keep separate" rules too aggressively, splitting further than intended (e.g., "search" vs "read found docs" vs "summarize" vs "write").
- The "exactly N tasks" framing is more reliable than open-ended "keep separate" instructions.
- The USER_CONTEXT skip rule must be stated as a positive imperative ("skip the data-acquisition task entirely") not just implied.
- Few-shot examples in-rule (e.g., "X is task 1; Y is task 2") are highly effective at anchoring the model.

**Why:** Evidence from 4 iterations of eval showing how gpt-oss:20b misinterprets vague rules.
**How to apply:** When optimizing other planner-style prompts for gpt-oss:20b, lead with explicit prohibitions, use "exactly N tasks" framing, and add inline examples for every rule that involves task splitting.
