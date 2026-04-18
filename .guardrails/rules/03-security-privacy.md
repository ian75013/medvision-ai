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

## Platform Security Assurance
- Every exposed software platform must have an explicit security posture: authentication, abuse controls, and observability.
- Security-sensitive platforms must provide operational visibility (dashboard or equivalent) for blocked IPs, intrusion signals, and incident timelines.
- When auto-blocking is enabled, provide a controlled unblock path with audit trail.
- Security controls must be tested after deployment (health checks + representative malicious input simulation).
- Document the security operating model (what is blocked, how to unblock, where to observe signals).

## Security Gates
- Run dependency vulnerability scan before release.
- Block release on critical vulnerabilities unless explicit exception is approved.

## Auto-Blocking Safety Rules (learned from incident 2026-04-17)

Auto-blocking mechanisms that can modify cluster-level network configuration carry systemic risk. The following rules are **mandatory** for any auto-blocking service deployed on k3s or equivalent infrastructure:

- **Never block private/internal IPs.** Any auto-blocker must exclude RFC1918 ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`), and all cluster-internal CIDRs (k3s: `10.42.0.0/16` pod CIDR, `10.43.0.0/16` service CIDR) before any blocking action.
- **Whitelist administrator IPs.** Expose a `ADMIN_WHITELIST_CIDRS` configuration variable. IPs in this list must never be blocked, regardless of score.
- **Never test an auto-blocker from the admin IP.** Use a dedicated test IP or a VPN endpoint you can afford to lose access from.
- **Scope the blast radius.** A service that can modify `ingress-nginx-controller` ConfigMap has cluster-wide impact. Validate the change is scoped to only the intended key.
- **Log every block and unblock action** with: IP, reason, score, timestamp, operator.
- **Review rate-limit values for admin UIs.** SPAs that load many parallel JS modules (Keycloak admin, Grafana…) require at least `limit-rps: 50` and `limit-connections: 50`. Default `20` is too low and causes module load failures.

### Admin Whitelist Baseline

- Current administrator public IPs to keep whitelisted (captured on 2026-04-17):
  - `90.65.229.231/32` (IP principale)
  - `150.228.15.196/32` (Starlink)
  - `146.70.253.134/32` (VPN Windscribe)
- This IP must be present in `ADMIN_WHITELIST_CIDRS` for any auto-blocking service.
- If this IP changes, update the whitelist before running security tests.

### Procedure — Moving House / Changing ISP (Public IP rotation)

Use this procedure whenever your public IP changes.

1. Detect new IP from your workstation:

```bash
curl -s https://ifconfig.me
```

2. Temporarily disable auto-blocker before changes:

```bash
kubectl scale deployment/security-intake -n security --replicas=0
```

3. Update whitelist config (replace OLD_IP and NEW_IP):

```bash
# Example target value format
ADMIN_WHITELIST_CIDRS="NEW_IP/32,150.228.15.196/32,146.70.253.134/32"
```

4. Remove old public IP from global block list if present:

```bash
kubectl get cm ingress-nginx-controller -n ingress-nginx -o jsonpath='{.data.block-cidrs}'
# If OLD_IP/32 exists, remove it from block-cidrs and patch the ConfigMap.
```

5. Re-enable auto-blocker:

```bash
kubectl scale deployment/security-intake -n security --replicas=1
```

6. Validate access from new IP:

```bash
curl -k -I https://keycloak.doctumconsilium.com
curl -k -I https://products.doctumconsilium.com/tools/security/dashboard
```

7. Update documentation and audit trail:
- Record the old and new IP in incident/change log.
- Update this guardrail baseline entry with the new `/32`.
- Keep one emergency access path (OVH KVM console) documented and tested.

## Incident Response — Security Incident Checklist

When a security incident is suspected (403 on critical services, auth loops, unexpected blocks):

**Step 1 — Isolate**
```bash
kubectl scale deployment/<auto-blocker> -n <ns> --replicas=0
```

**Step 2 — Diagnose**
- Check `block-cidrs` in `ingress-nginx-controller` ConfigMap for unexpected entries.
- Check if your public IP or any internal IP (`10.42.x.x`, `10.43.x.x`) is listed.
- Check service logs for the source of the block decision.

**Step 3 — Unblock**
```bash
kubectl patch configmap -n ingress-nginx ingress-nginx-controller \
  --type='json' -p='[{"op":"remove","path":"/data/block-cidrs"}]'
```

**Step 4 — Fix and redeploy**
- Identify root cause in code.
- Write a failing test reproducing the issue.
- Fix until test passes.
- Redeploy **once**.

**Step 5 — Document**
- Write a postmortem: timeline, root causes, corrections, recommendations.
- Update `docs/security-incident-runbook.md`.
- Update these guardrails if a new risk category is identified.

> Reference postmortem: `k3s-fromOVHVps/docs/security-incident-runbook.md`
