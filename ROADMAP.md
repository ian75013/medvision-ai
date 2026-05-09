# ROADMAP

## Last Update
2026-05-08

## Active Phase
Phase 1 — Baseline stability and documentation

## Goals
- Maintain service availability
- Keep dependencies and documentation up to date
- Enforce guardrails (code documentation, secret management, deployment standards)

## Execution Log — 2026-05-08

### Guardrails Propagation

- Added `INFRASTRUCTURE.md` with operational runbook
- Added `CLAUDE.md` and `GEMINI.md` for multi-AI assistant compatibility
- Updated `.github/copilot-instructions.md` with:
  - Code documentation standard (pydoc/JSDoc/XML/shell headers)
  - Secret management protocol (ECR 12h rotation, k8s pull secrets)
  - tmux sessions mandatory for critical operations
- ECR secrets agent available at `.github/agents/ecr-secrets-agent.agent.md`

## Rollback
- All changes above are documentation-only and non-breaking.
- Revert any file via `git checkout HEAD~1 -- <file>`.
