# 06 - Observability and Operations

## Minimum Runtime Signals
- Service health endpoints.
- Error rate and latency metrics.
- Key business KPI signal.
- Structured logs for critical decisions.

## Operational Readiness
- Dashboards for critical flows.
- Alerting with on-call ownership.
- Runbook for top failure scenarios.

## Terminal and Process Hygiene
- After each validation or debug run, stop terminal sessions and background processes that are no longer needed.
- Never leave duplicate long-running test commands in parallel unless explicitly required and documented.
- Keep only the minimum active terminals needed for the current task (for example: one app server, one test runner).
- If multiple runs were started accidentally, terminate redundant processes first and re-run a single controlled command.
- For incidents caused by hidden parallel runs, add a preventive check to scripts or runbooks (timeout, lock file, or explicit PID cleanup step).

## Incident Response Baseline
- Capture timeline and impact.
- Identify technical root cause.
- Apply fix and verify with tests/checks.
- Add preventive action and owner.

## Post-Incident Rule
- Every high-severity incident must result in at least one guardrail or test improvement.
