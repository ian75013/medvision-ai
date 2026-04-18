# Copilot Instructions

This repository uses the Guardrails Kit.

## Mandatory Rule Source

Before making changes, follow the guardrail files installed in:

- `.guardrails/rules/01-core-principles.md`
- `.guardrails/rules/02-engineering-standards.md`
- `.guardrails/rules/03-security-privacy.md`
- `.guardrails/rules/04-testing-quality-gates.md`
- `.guardrails/rules/05-release-change-management.md`
- `.guardrails/rules/06-observability-operations.md`
- `.guardrails/rules/07-documentation-knowledge.md`

If project-specific instructions exist, apply both sets of rules. If there is a conflict, use the stricter safety rule.

## Operational Expectations

- Prefer test-first changes for behavior fixes.
- Do not ship changes without appropriate tests.
- Avoid destructive commands unless explicitly requested.
- Keep changes minimal, reversible, and documented.
