# 05 - Release and Change Management

## Change Scope Rules
- Avoid multi-axis risky releases (architecture + infra + business logic at once).
- Prefer incremental rollout when possible.

## Release Requirements
- Release plan with owner and rollback path.
- Versioned artifact and reproducible build.
- Pre-release smoke checks documented.

## Rollback Requirements
- Define rollback trigger conditions before release.
- Keep rollback command/procedure tested and documented.
- Record rollback outcome and follow-up actions.

## Deployment Safety
- No destructive cleanup commands in standard deploy path.
- Confirm target environment and config before execution.
- Push branch before deploy (`git push`) and deploy only from remote tracked commit.
- Verify commit parity after deploy: target runtime commit SHA must match pushed SHA.
- Stop release immediately on failed smoke checks.

## Multi-Compose and Env Topology (Mandatory)

### Single source of truth
- Each repository must define and maintain one deployment topology table (in docs) with:
	- stack name,
	- compose file,
	- env file,
	- compose project name,
	- exposed ports,
	- bind interface (`127.0.0.1`, VPN IP, or public).
- No deploy is allowed if this mapping is missing or outdated.

### Explicit deploy command only
- In multi-stack environments, always run deploy commands with explicit flags:
	- `docker compose -p <project> --env-file <env-file> -f <compose-file> ...`
- Do not rely on implicit defaults (`.env`, default compose file, or default project name).

### Stack isolation requirements
- Every stack must use a unique compose project name (`-p`).
- Every stack must have non-overlapping host ports.
- Every stack must have explicit bind intent:
	- VPN-only services bind to VPN IP,
	- localhost-only services bind to `127.0.0.1`.

### Preflight gate (blocking)
- Before `up`/`restart`, run and record:
	1. target stack identity (`-p`, `-f`, `--env-file`),
	2. port occupancy check (`ss -tln` or equivalent),
	3. service identity check (`curl`/health response proves which service answers on target port).
- If any preflight check is ambiguous, stop and fix topology first.

### Post-deploy smoke gate (blocking)
- Validate health endpoint on the intended interface and port.
- Validate that response content matches the intended service (not just open port).
- Validate expected containers for the selected project only.

### Anti-pattern: Cross-stack drift
**Symptom**: multiple compose files and env files coexist, commands mix defaults, and services collide on ports or project scope.

**Why it is wrong**:
- Causes false diagnostics (port open but wrong service behind it).
- Produces inconsistent state between stacks.
- Increases MTTR and operational risk.

**Required discipline**:
1. Identify stack using explicit `-p`, `-f`, and `--env-file`.
2. Verify ports and bind interfaces before deploy.
3. Deploy only the targeted stack/services.
4. Validate endpoint identity and health.

## Anti-Pattern: Deploy Loop
**Symptom**: A bug appears in production. Agent/developer redeploys multiple times with small adjustments to observe behavior in live logs.

**Why it is wrong**:
- Burns rate limits and external API quotas on every restart (e.g. Tradovate 5 auth/hour).
- Introduces noise into production logs and metrics.
- Does not produce a reproducible test — the fix is invisible and unverified.
- Violates the principle: *small, reversible changes with known outcomes only*.

**Required discipline**:
1. Reproduce the failure locally (unit test or integration test with mocked external).
2. Write a failing test that pins the bug.
3. Fix the code until the test passes.
4. Deploy **once** with confidence.
5. Validate with a single smoke check post-deploy.
