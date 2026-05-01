# 01 - Core Principles

## Mandatory Principles
1. Safety over speed.
2. Small, reversible changes over large risky batches.
3. Explicit ownership for every production-impacting change.
4. Testability and observability are first-class requirements.
5. No undocumented behavior in production systems.
6. Minimize side effects: any shared infra/auth/network change must include an explicit impact map and containment plan.
7. Plan before action: for any non-trivial task, define a short execution plan, review relevant code paths first, then execute.

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
- If a change can affect multiple products: run cross-product integrity checks before and after deployment.
- If root cause is not established: pause execution, read relevant code and logs, then revise the plan before continuing.
