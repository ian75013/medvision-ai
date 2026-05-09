# CLAUDE.md

This file defines Claude Code execution rules for `medvision-ai`.

## Mandatory Sources of Truth
- `README.md`
- `INFRASTRUCTURE.md` (if present)
- `SKILLS.md` (if present)
- `ROADMAP.md` (if present)
- `.github/copilot-instructions.md`

## Project Context
medvision-ai — medical image AI pipeline (DVC, Python, Docker)

## Execution Principles
1. Read sources of truth before any non-trivial task.
2. Align every change with the active phase in `ROADMAP.md`.
3. Fix root causes before any workaround.
4. Keep changes minimal, targeted, and verifiable.
5. Update documentation when behavior changes.

## Code Documentation Standard (Mandatory)
- Every public function, class, and module MUST have a docstring in the language's canonical format:
  - Python: Google-style docstrings (Args, Returns, Raises, Example).
  - TypeScript/JavaScript: JSDoc with @param, @returns, @throws, @example.
  - C#: XML summary with <summary>, <param>, <returns>.
  - Shell: header block with Purpose, Usage, Arguments, Exit codes.
- When modifying a function, update its docstring to reflect the new behavior.
- Run `pydoc`, `typedoc`, or equivalent after doc changes to verify output.

## Secret Management Protocol (Mandatory)
- Never hardcode secrets, tokens, or credentials in any file, script, log, or commit.
- ECR / Docker Registry tokens rotate every 12 hours. Before any image operation:
  1. `aws ecr get-login-password --region <region>` to refresh.
  2. Recreate the k8s pull secret via `kubectl delete/create secret`.
  3. Restart pods stuck in `ImagePullBackOff`.
- Document rotation commands in `INFRASTRUCTURE.md` under Operations.

## Terminal Sessions (Mandatory for Critical Operations)
- Always use a named tmux session for long-running, deployment, or destructive operations:
  ```bash
  tmux new-session -A -s <task-name>   # start or reattach
  # Naming: deploy-<service>, build-<tag>, k3s-ops, ecr-refresh
  ```
- On Windows: use Windows Terminal tabs or Start-Process with file logging.
- On macOS: same tmux convention applies.

## Environment Policy
- Use a single standard environment per repo when possible.
- Parallel environments allowed only if technically unavoidable — document reason, limits, naming, activation.

## Platform Compatibility
- Critical automation scripts must provide Ubuntu (Bash) and Windows (PowerShell) execution.
- macOS (zsh/bash) compatibility required when using POSIX-standard commands.
- Any exception must be documented with an operational alternative.
