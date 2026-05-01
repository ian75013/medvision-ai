# 03 - Security and Privacy

## Secrets and Credentials
- Never commit secrets or tokens.
- Use environment variables or secret managers.
- Rotate credentials after suspected leakage.

## Data Protection
- Classify data: public, internal, sensitive.
- Minimize sensitive data retention.
- Log only what is needed for operations.

## Runtime Security
- Default-deny network exposure.
- Restrict admin endpoints.
- Use least-privilege service accounts.

## Security Gates
- Run dependency vulnerability scan before release.
- Block release on critical vulnerabilities unless explicit exception is approved.
