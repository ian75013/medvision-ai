# 02 - Engineering Standards

## Code Standards
- Keep modules cohesive and interfaces explicit.
- Prefer pure functions for critical business logic where possible.
- Validate all external input at boundaries.
- Avoid hidden global state for decision logic.

## Review Standards
- Every PR must include: scope, risk, test evidence, rollback note.
- Review must focus first on correctness, regressions, and failure modes.
- High-risk changes require at least one additional reviewer.

## Dependency Standards
- Pin dependencies in production builds.
- Track license and known vulnerabilities.
- Do not add a dependency if standard library or existing dependency can solve it safely.

## YAML Manifest Standards
- Do not embed large executable application code directly in YAML manifests.
- Keep YAML focused on declarative infrastructure and configuration; application logic must live in versioned source files.
- For Kubernetes, mount code via image builds or file-backed ConfigMaps generated from repository files, not inline heredocs.
- Any temporary inline script in YAML must be short, documented with rationale, and tracked for removal in TODO.
