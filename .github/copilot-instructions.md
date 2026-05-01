# Copilot Instructions

These rules apply to all coding tasks in medvision-ai.

## Guardrails
- Always read and follow `SKILLS.md` and `ROADMAP.md` before significant edits.
- Apply a mandatory Plan-Analyze-Read-Act workflow on non-trivial tasks:
	1) plan concise steps, 2) analyze likely failure modes, 3) read all relevant code paths/files, 4) execute.
- Keep changes minimal and directly tied to user request.
- Prefer root-cause fixes over superficial patches.
- Include validation steps for implemented changes.
- Avoid side effects: for shared infra/auth/routing changes, list impacted products and validate each one.

## Chat Behavior
- In each substantial request, propose a short execution plan aligned to `ROADMAP.md` phases.
- Before first code edit, summarize assumptions and confirm the plan still matches observed code.
- Keep progress updates concise and action-oriented.
- If blocked, state blocker clearly and propose one practical workaround.

## Code & Quality
- Preserve existing style and architecture.
- Avoid unrelated refactors.
- Do not introduce secrets in code or logs.
- Update documentation when behavior changes.
- Prefer fewer, higher-confidence edits over many speculative changes.
- Require non-regression evidence beyond the changed service when common dependencies are touched (ingress, DNS, auth, registry, shared APIs).
