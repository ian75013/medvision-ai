# 01 - Core Principles

## Mandatory Principles
1. Safety over speed.
2. Small, reversible changes over large risky batches.
3. Explicit ownership for every production-impacting change.
4. Testability and observability are first-class requirements.
5. No undocumented behavior in production systems.

## Non-Negotiables
- No destructive operations in routine workflows.
- No release without rollback strategy.
- No merge without minimum quality gates.
- No incident closure without root cause and validation evidence.
- **No remote redeploy loop to debug a behavior change** — reproduce and fix locally first, deploy once when the test passes.
- **No deploy from unpushed local state** — push target branch first, then deploy from remote source of truth.

## Decision Rules
- If uncertainty is high: reduce blast radius.
- If data is missing: fail safe.
- If monitoring is missing: do not ship.
