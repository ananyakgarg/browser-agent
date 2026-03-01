# Kafka PR Audit — Results Analysis

## Overview

| Metric | Count |
|--------|-------|
| Total PRs | 60 |
| Completed | 46 (77%) |
| Failed (timeout) | 14 (23%) |

## Findings

### CI Check Status Breakdown (46 completed PRs)

| Category | Count | Notes |
|----------|-------|-------|
| All checks passed | 24 | Clean merges with full CI green |
| No CI checks ran | 14 | Older PRs from Jenkins era — "Workflow runs completed with no jobs" |
| Merged despite failures | 9 | Maintainers overrode failing checks (see details below) |

### Jira Ticket Coverage

- **35 of 46** PRs referenced a Jira ticket (KAFKA-XXXXX pattern)
- Agent navigated to each Jira ticket and captured a screenshot
- 11 PRs used `MINOR:` prefix or had no Jira reference

### PRs Merged Despite Failed Checks

| PR | Failed Check | Reason |
|----|-------------|--------|
| #18450 | JUnit tests Java 17 | Flaky test — reviewer noted it was unrelated |
| #18500 | JUnit tests Java 23 + Java 17 | Reviewer confirmed failures were pre-existing flaky tests |
| #18550 | JUnit tests Java 17 | Exit code 1 — reviewer approved despite failure |
| #18700 | JUnit tests Java 17 + Java 25 | 7 of 9 passed — test failures deemed unrelated |
| #18994 | Collate Test Catalog | Infrastructure issue, not a code problem |
| #19000 | JUnit tests Java 23 | Failed after 126min — likely timeout/flaky |
| #19150 | Jenkins pr-merge | Jenkins check failed, maintainer asked about test relevance |
| #19312 | Jenkins pr-merge | All Jenkins checks failed, merged anyway by maintainer |
| #19550 | PR Linter | "No Reviewers found in commit body" — not a required check |

### Key Patterns

1. **Flaky Java tests are the #1 override reason.** JUnit tests for Java 17 and Java 23 fail intermittently. Maintainers routinely merge after confirming failures are unrelated.

2. **Jenkins → GitHub Actions migration.** Older PRs (pre-#18400) have zero GitHub Actions checks. CI ran through Jenkins (`continuous-integration/jenkins/pr-merge`) which shows in the commit status area, not the Checks tab.

3. **PR Linter failures are non-blocking.** The PR Linter check ("No Reviewers found in commit body") fails but is not configured as a required check.

## Failures

All 14 failures were timeouts (600s). Root cause: Opus model latency (100-190s per iteration) × multi-step task (navigate PR → screenshot → check CI → find Jira → navigate Jira → screenshot) = exceeded timeout with only 6-9 iterations completed.

## Model & Config

- **Model**: claude-opus-4-6
- **Workers**: 10 concurrent
- **Timeout**: 600s per sample
- **Max iterations**: 30
