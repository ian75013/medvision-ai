# Copilot Instructions

These rules apply to all coding tasks in medvision-ai.

## Chat Guardrails
- On the first chat of a session, read project documentation before any significant action (at minimum the `doc` or `docs` folder when present, plus `README.md`, `INFRASTRUCTURE.md` if present, `SKILLS.md` if present, and `ROADMAP.md` if present).
- Read `SKILLS.md`, `ROADMAP.md`, and `INFRASTRUCTURE.md` (when present) before any substantial task.
- Align the execution plan with the active phase in `ROADMAP.md` when it exists.
- Apply a mandatory Plan -> Analyze -> Read relevant code -> Act workflow for any non-trivial task.
- Before any edits, briefly summarize assumptions and verify they match observed code.
- Provide short, actionable progress updates during execution.
- Prefer minimal, verifiable changes.

## Technical Guardrails
- Fix root causes rather than patching symptoms.
- Preserve repository style and avoid out-of-scope refactors.
- Maintain Ubuntu + Windows compatibility for automation scripts.
- Never hardcode secrets in files, scripts, or logs.
- When behavior changes, update related documentation.

## Quality
- Validate changes (targeted validation script, dry-run, or equivalent checks).
- Reduce speculative actions: prefer fewer high-confidence actions over many low-confidence attempts.
- If a blocker persists, document the cause and propose the simplest practical alternative.

## Execution Contract (Mandatory)
- Treat each user request as an end-to-end execution task until the explicit done criteria are met.
- Do not stop at analysis or planning only: execute code changes, validation, and deployment steps when requested.
- Provide factual progress updates every 3 to 5 meaningful actions, with the next concrete step.
- On blocker, report: root cause, what was attempted, one practical workaround, then continue unless explicit approval is required.
- A task is done only when results are verifiable: changed files, validations run, deployment status (if requested), and rollback steps documented in ROADMAP.md.
- Prefer deterministic actions over speculative iterations; keep changes minimal and reversible.


## Code Documentation Standard (Mandatory)
- Every public function, class, and module MUST have a docstring/JSDoc/XMLDoc in the language's canonical format:
  - Python: Google-style docstrings (Args, Returns, Raises, Example sections).
  - TypeScript/JavaScript: JSDoc with `@param`, `@returns`, `@throws`, `@example`.
  - C#: XML summary with `<summary>`, `<param>`, `<returns>`, `<exception>`.
  - Shell scripts: header block with Purpose, Usage, Arguments, Exit codes.
- When modifying a function, update its docstring to reflect the new behavior.
- New modules must include a module-level docstring describing purpose, dependencies, and usage example.
- Run `pydoc`, `typedoc`, or equivalent after doc changes to verify output is valid.

## Secret Management Protocol (Mandatory)
- Never hardcode secrets, tokens, or credentials in any file, script, log, or commit.
- Use environment variables or a secrets manager (Vault, AWS Secrets Manager, k8s Secret) for all sensitive values.
- **ECR / Docker Registry tokens rotate every 12 hours.** Before any Docker pull/push or k8s image operation:
  1. Refresh the token: `aws ecr get-login-password --region <region>`.
  2. Recreate the k8s pull secret: `kubectl delete secret <name> --ignore-not-found && kubectl create secret docker-registry <name> ...`.
  3. Restart affected pods if already in `ImagePullBackOff`.
- Document token rotation commands in `INFRASTRUCTURE.md` under the **Operations** section.
- If a secret is accidentally committed, rotate it immediately and document the incident.
- See `.github/agents/ecr-secrets-agent.agent.md` for the full runbook.

## Terminal Sessions (Mandatory for Critical Operations)
- **Always use a named tmux session** for long-running, deployment, or destructive operations:
  ```bash
  tmux new-session -A -s <task-name>   # start or reattach
  # Naming: deploy-<service>, build-<tag>, k3s-ops, ecr-refresh
  # Detach: Ctrl+B D  |  Reattach: tmux attach -t <task-name>
  ```
- This prevents work loss on SSH disconnection and provides a recoverable audit trail.
- On Windows (PowerShell): use Windows Terminal tabs or `Start-Process` with logging to a file as equivalent.
- On macOS: same tmux convention (iTerm2 or Terminal).
