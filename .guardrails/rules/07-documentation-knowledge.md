# 07 - Documentation and Knowledge

## Documentation is Mandatory
Every new code addition or behavior change must update documentation in the same change.

## Required Documentation Fields
- What changed.
- Why it changed.
- How it works (inputs, outputs, failure mode).
- How to validate it.
- How to rollback it.

## Knowledge Transfer
- Keep architecture and runbook documents current.
- Link all new docs from a central index.
- Avoid undocumented tribal knowledge for production-critical behavior.

## NOGO Documentation Cases
- New feature without docs.
- New env variable not documented.
- Release without updated validation steps.
