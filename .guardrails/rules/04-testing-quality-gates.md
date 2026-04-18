# 04 - Testing and Quality Gates

## Required Test Layers
1. Unit tests for core logic and edge cases.
2. Integration tests for service boundaries.
3. End-to-end tests for critical user and cross-service flows.

## Merge Blockers
- New behavior without tests.
- Failing unit or integration tests.
- Missing regression test for a fixed bug.
- Missing schema contract tests for API changes.

## GO/NOGO Baseline
- GO:
  - All required tests pass.
  - Health checks pass.
  - Contract checks pass.
- NOGO:
  - Any blocker above fails.
  - Key endpoint returns invalid schema.
  - Critical monitoring signal missing.

## Testing Discipline
- For bug fixes: reproduce with failing test first.
- Keep deterministic tests for production-critical paths.
- Treat flaky tests as production risk and fix or quarantine with explicit owner.
- Run end-to-end tests frequently on critical flows: at least on each PR before merge, after deployment, and before release GO/NOGO.
- For auth, ingress, reverse-proxy, or callback-path changes, add a framework-based integration smoke test that validates the public protected route, redirect behavior, and recent ingress/auth logs after deployment.
