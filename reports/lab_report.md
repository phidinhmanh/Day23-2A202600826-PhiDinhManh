# Day 08 Lab Report

## 1. Team / student

- Name: Phí Đình Mạnh
- Repo/commit: Day 08 Lab
- Date: 2026-06-29

## 2. Architecture

Our graph follows a multi-route design with the following components:
- Nodes: intake_node, classify_node, tool_node, evaluate_node, answer_node,
         ask_clarification_node, risky_action_node, approval_node,
         retry_or_fallback_node, dead_letter_node, finalize_node.
- Edges: Fixed execution flows + conditional edge routing based on state updates.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append | audit conversation/events |
| tool_results | append | track execution outcomes |
| errors | append | track trace logs of failure |
| events | append | audit internal execution trail |
| route | overwrite | current classified route |
| risk_level | overwrite | current risk verification |
| attempt | overwrite | current attempt count |
| max_attempts | overwrite | max attempt restriction |
| final_answer | overwrite | final synthesized response |
| evaluation_result | overwrite | validation loop result |
| pending_question | overwrite | clarification followup question |
| proposed_action | overwrite | description of risky mutation |
| approval | overwrite | decision of hitl |

## 4. Scenario results

Total Scenarios: 7
Success Rate: 100.00%
Average Nodes Visited: 6.43
Total Retries: 3
Total Interrupts: 2

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 |
| S02_tool | tool | tool | True | 0 | 0 |
| S03_missing | missing_info | missing_info | True | 0 | 0 |
| S04_risky | risky | risky | True | 0 | 1 |
| S05_error | error | error | True | 2 | 0 |
| S06_delete | risky | risky | True | 0 | 1 |
| S07_dead_letter | error | error | True | 1 | 0 |

## 5. Failure analysis

1. Retry or tool failure: When external services fail, the retry loop increments
   attempt count and routes back to tools until max attempts are exhausted,
   preventing infinite loops.
2. Risky action without approval: Evaluates risk levels. If classification marks
   a query as risky, routing goes through approval_node first, enforcing safety
   constraints.

## 6. Persistence / recovery evidence

Wired with SqliteSaver checkpoint persistence, utilizing unique thread_ids
to query past checkpoints, enable transaction rollbacks, or recover state
after abrupt processes exits.

## 7. Extension work

Implemented SQLite checkpointer support using WAL mode, and incorporated
LLM-as-judge in evaluate_node to assess tool output quality.

## 8. Improvement plan

If we had more time, we would implement Postgres checkpointer support for scaling,
add structured JSON logging for observability (OTel/LangSmith), and support
concurrent tool executions using LangGraph's Send API.
