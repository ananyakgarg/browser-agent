# Kafka PR Audit

## Instruction
For each PR, extract: a screenshot of the PR page, the CI/check status (look at the merge status area and any checks — identify what passed, failed, and whether required checks were met), whether the PR was merged despite any failed required checks, and if the PR title or description references a Jira ticket (KAFKA-XXXXX pattern), navigate to the Jira ticket and screenshot it.

## Model
claude-opus-4-6

## Config
- Workers: 10
- Timeout: 600s
- Max iterations: 30

## Results
- 60 PRs total
- 46 completed (77%)
- 14 failed (all timeouts — Opus latency × iteration count exceeded 600s)
