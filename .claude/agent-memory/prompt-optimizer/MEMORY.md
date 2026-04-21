# Prompt Optimizer Memory Index

- [Planner prompt v2 optimization](planner_v2_optimization.md) — Best candidate is v4 (8/8 pass); key failure modes and winning rules for gpt-oss:20b
- [Query Reformatter prompt optimization](query_reformatter_optimization.md) — Converged at v2 (5/5 pass); decision-tree ordering + statement-to-verify rule are the key fixes
- [Replanner prompt v2 optimization](replanner_v2_optimization.md) — v2 already passes 5/5; key fix is pre-computing remaining plan tasks in Jinja2 + verbatim-return rule
