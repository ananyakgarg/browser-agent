# Wikipedia Citation Verification — v2 Results

## TL;DR

**38% → 100% completion. 7x faster. 75% fewer tokens.**

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Completed | 3/8 | **8/8** | +167% |
| Failed | 5/8 | **0** | Eliminated |
| Total time | >4800s | **670s** | ~7x faster |
| Avg tokens/sample | 1.1–1.4M | **249K** | ~75% reduction |
| Avg iterations | 25+ (hit limit) | **13** | ~50% fewer |

## Head-to-Head: v1 vs v2

| Article | v1 | v2 | Tokens (v1 → v2) |
|---------|----|----|-------------------|
| Basel III | Completed (25 iters) | Completed (10 iters) | ~1.1M → 140K |
| Black-Scholes | **Failed** (30 iter limit) | Completed (14 iters) | blew up → 196K |
| CRISPR | **Failed** (600s timeout) | Completed (15 iters) | 1.4M+ → 281K |
| Monte Carlo | **Failed** (600s timeout) | Completed (19 iters) | blew up → 399K |
| PageRank | Completed (25 iters) | Completed (7 iters) | ~1.2M → 92K |
| RSA | Completed (20 iters) | Completed (10 iters) | ~1.1M → 170K |
| Sarbanes-Oxley | **Failed** (30 iter limit) | Completed (19 iters) | blew up → 452K |
| Transformer | **Failed** (600s timeout) | Completed (12 iters) | 1.1M+ → 260K |

## What Changed

### 1. Context Compaction (biggest impact)
Server-side compaction via `compact-2026-01-12` beta. Triggers at 50K input tokens — Claude auto-summarizes old conversation turns. In v1, every tool call appended ~3K tokens of page state to the message history, which accumulated to 1M+ tokens and either hit the context limit or caused quality degradation.

### 2. execute_js (replaced scroll/get_text)
Instead of scrolling through a page 13+ times to find references (v1 pattern), agents now run one JS expression like:
```js
[...document.querySelectorAll('.references li')].map(li => li.innerText.substring(0, 300))
```
This extracts all references in a single call vs 13+ scroll iterations.

### 3. http_request (URL verification without browser)
Checking if a citation URL is live now takes one `http_request` call (~100ms) instead of navigating the browser to the URL, waiting for it to load, and reading the page state (~5-10s + context bloat).

### 4. Tool Deprecation
Removed 5 tools that caused degenerate agent behavior:
- `find_element` → `find_element` loops: **84 occurrences** in v1 Kafka traces
- `scroll` → `scroll` loops: **25 occurrences** in v1 traces
- Also removed: `get_text`, `go_back`, `select_option`

All subsumed by `execute_js` which is more precise and doesn't dump page state.

### 5. Pioneer-Follower Pattern
Basel III ran as the pioneer, generating a detailed playbook with exact JS expressions, URL patterns, and edge cases. The remaining 7 articles followed this playbook, reducing exploration time.

## Results

| Article | Oldest Citation | Year | URL | Accuracy |
|---------|----------------|------|-----|----------|
| Basel III | Basel II First Pillar doc | 2005 | live | partially accurate |
| Black-Scholes | Bachelier's Théorie de la Spéculation | **1900** | live | accurate |
| CRISPR | Woolf et al., PNAS | 1995 | live | accurate |
| Monte Carlo | Fermi & Richtmyer, Los Alamos | 1948 | live | accurate |
| PageRank | Landau, chess tournament ranking | **1895** | no URL (1895 paper) | accurate |
| RSA | Cocks, GCHQ classified paper | 1973 | **dead → Wayback** | accurate |
| Sarbanes-Oxley | 18 USC § 1350 | 2002 | live | accurate |
| Transformer | von der Malsburg, brain function | 1981 | live | accurate |

### Notable Behaviors
- **RSA**: Source URL was dead (GCHQ took it down). Agent automatically queried Wayback Machine API and found an archived snapshot from Nov 2019.
- **PageRank**: 1895 German mathematics paper — no URL exists. Agent correctly reported "no_url" instead of hallucinating a link.
- **Basel III**: Rated "partially accurate" — the citation references a Basel II document used in context of Basel III, which is a nuanced but correct assessment.
- **Black-Scholes**: Found Bachelier's 1900 thesis — the earliest known application of Brownian motion to finance.

## Per-Sample Audit

| Article | Iters | Time (s) | Tokens | Role |
|---------|-------|----------|--------|------|
| Basel III | 10 | 88 | 140K | Pioneer |
| Black-Scholes | 14 | 82 | 196K | Follower |
| CRISPR | 15 | 543 | 281K | Follower |
| Monte Carlo | 19 | 454 | 399K | Follower |
| PageRank | 7 | 301 | 92K | Follower |
| RSA | 10 | 320 | 170K | Follower |
| Sarbanes-Oxley | 19 | 251 | 452K | Follower |
| Transformer | 12 | 376 | 260K | Follower |

## Config
- **Model**: claude-sonnet-4-6
- **Workers**: 8 concurrent
- **Timeout**: 600s per sample
- **Max iterations**: 30
- **Compaction trigger**: 50K input tokens
- **Pioneer-follower**: enabled
