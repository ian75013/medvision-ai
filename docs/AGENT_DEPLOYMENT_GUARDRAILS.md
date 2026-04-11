# Agent Deployment Guardrails

## Purpose
Prevent recurring deployment incidents across repositories by enforcing non-destructive, testable, and reversible operations.

## Mandatory Rules
1. Never delete data volumes in routine deploy
- Forbidden in deploy scripts: `down -v`, `--volumes`, `docker volume rm`, `docker system prune`.

2. No multi-axis production changes
- In one release, change only one of:
  - infra/deploy mechanics
  - provider/business logic
  - frontend build/runtime config

3. Preflight before any deploy
- Confirm target env file.
- Confirm frontend API base URL is public and correct.
- Confirm credentials/tokens for any new provider.

4. Post-deploy smoke tests are blocking
- Frontend root returns 200.
- JS bundle returns 200.
- Bundle does NOT contain localhost/private API origin.
- API health returns 200.
- Functional endpoint check returns valid payload.

5. Rollback-ready policy
- Every prod deploy must have a clear rollback commit/tag.
- No irreversible DB operation without explicit, approved migration plan.

6. Communication policy
- First message: customer impact + scope + ETA + rollback status.
- Do not minimize incidents with vague wording (example to avoid: "just cold start").

## Deployment Checklist (copy/paste)
1. `git status` is clean.
2. Correct branch/commit selected.
3. Correct env file selected.
4. Build arguments validated (especially frontend API URL).
5. Deploy command executed once.
6. Smoke tests pass.
7. If failure: rollback immediately, then analyze.

## Incident Record Format
1. Summary
2. Impact
3. Timeline
4. Root cause(s)
5. Corrective actions
6. Preventive actions
7. Owners and deadlines
