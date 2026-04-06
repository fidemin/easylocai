---
name: Query Reformatter prompt optimization
description: Optimization history and winning rules for the query_reformatter system prompt — converged at v3 (6/6 pass)
type: project
---

## Result

Converged at candidate v3 in 1 iteration (6/6 pass). The improved prompt was applied directly to the original file.

**Optimized file:** `resources/prompts/query_reformatter_system_prompt.jinja2`

## Key Failure Modes (History)

1. **Spurious query_context on self-contained queries** (original baseline): The model incorrectly treated the action itself as "new information". Fix: explicit null-trigger list + "null by default" framing.

2. **Statement-as-preamble confusion** (v1 → v2): When the user states a claim and asks "Is this true?", a naive preamble rule caused the model to extract the statement into query_context. Fix: decision tree step 2 — "statement to verify" keeps the claim in reformed_query.

3. **Schema/data preamble not recognized as setup context** (v2 → v3): Decision Rule 3 said "factual environment/system information" — too narrow, priming the model toward infrastructure facts only. A DB schema ("My database has tables...") was treated as self-contained and query_context was set to null. Fix: broadened to "factual setup information of any kind (environment, configuration, data structures, schemas, or other context)". Also broadened the pattern description from "[facts about my environment/system]" to "[facts about the user's setup, data, or context]".

## Winning Rules (v3)

1. **Decision tree ordering matters**: Check pronoun resolution first, then "statement to verify", then "factual preamble + separate action". This prevents false positives at each step.

2. **Concrete pattern + counter-examples**: Showing `[facts about the user's setup, data, or context]. [action].` as the only non-null pattern, with explicit null counter-examples, is far more effective than abstract rules.

3. **"statement to verify" rule**: If the user is asking to verify/explain/evaluate a claim, the claim IS the subject of reformed_query — do not extract it to query_context.

4. **null is the default**: Explicitly labeling query_context as "null in most cases" with an enumerated null-trigger list reduces false positives dramatically.

5. **Broad preamble language**: Decision Rule 3 must enumerate diverse setup types to avoid the model narrowing to a single domain (e.g. infrastructure). "Factual setup information of any kind (environment, configuration, data structures, schemas, or other context)" generalizes reliably.

## Iteration Summary

| Iteration | Score | Key change |
|-----------|-------|-----------|
| Baseline | 3/5 | Original prompt |
| v1 | 4/5 | Decision tree + concrete examples + null-by-default framing |
| v2 | 5/5 | Added "statement-to-verify" rule (rule 2) to prevent claim extraction |
| v3 | 6/6 | Broadened Decision Rule 3 preamble pattern to cover all factual setup types, not just "environment/system" |
