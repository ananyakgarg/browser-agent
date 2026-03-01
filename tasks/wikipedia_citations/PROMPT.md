# Wikipedia Citation Verification

## Instruction
For each Wikipedia article, find the claim with the oldest citation. Click through to that citation's source URL. Check if the source still exists (not 404). If it does, read the source and verify whether the Wikipedia claim accurately represents what the source actually says. If the source is dead, check the Wayback Machine. Rate each claim as: accurate, misleading, unsupported, or source dead.

## Model
claude-sonnet-4-5-20250929

## Config
- Workers: 3
- Timeout: 600s
- Max iterations: 30

## Results
- 8 articles total (Black-Scholes, Monte Carlo, PageRank, RSA, Transformer, CRISPR, Sarbanes-Oxley, Basel III)
- 3 completed (Basel III: accurate, PageRank: unsupported, RSA: accurate)
- 5 failed (3 timeouts, 2 hit 30 iteration limit — context growth was the root cause)
