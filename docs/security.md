# PixLint — Security Guide

## Threat Model

This MCP server runs **on the user's machine** with direct filesystem access. The LLM client (Claude, Cursor, etc.) sends tool calls over stdio — a compromised or malicious LLM could attempt:

| Attack Vector | Impact | Mitigation |
|---------------|--------|------------|
| Path traversal (`../../../etc/passwd`) | Arbitrary file read/write | `PathValidator` resolves & checks against allowed bases on **both reads and writes** |
| Credential injection (AWS keys in params) | Cloud account compromise | `CredentialManager` — creds ONLY from env vars |
| Pipeline code execution (`__import__('os').system(...)`) | RCE | `PipelineSecurityValidator` — AST blocks dangerous nodes |
| Resource exhaustion (infinite loops, huge outputs) | DoS | `RateLimiter` + `ResourceLimiter` — enforced per tool call (sliding-window rate limit, max-concurrency slot) |
| Decompression bomb (tiny file, huge decoded image) | Memory exhaustion / DoS | Pixel-count guard on image decode (`CV_MAX_IMAGE_PIXELS`, default ~64MP) |
| Data exfiltration (LLM reads all files) | Privacy breach | Output sanitization redacts credentials; paths restricted; exceptions returned as sanitized strings (no stack traces / secrets) |
| Audit evasion (no trace of malicious calls) | Non-repudiation loss | `SecurityAuditor` — immutable append-only log |
| XML entity expansion (billion laughs / XXE) | DoS / SSRF / file read | XML parsed with `defusedxml` |

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TOOL CALL ENTRY POINT                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  InputValidator.validate_*()                                 │
│  • dataset_id: alphanumeric, 200 chars max                   │
│  • format: whitelist (coco, voc, yolo, kitti, ...)           │
│  • model: whitelist (resnet50, clip-vit-base, ...)           │
│  • ratios: sum to 1.0, positive floats                       │
│  • bbox: [x1,y1,x2,y2], x2>x1, y2>y1, non-negative          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PathValidator.validate_path()  (reads AND writes)           │
│  • Blocks: .., ~, ${}, `, |, ;, &, \x00                      │
│  • Resolves: Path(path).resolve()                            │
│  • Checks: dangerous extensions (.exe, .sh, .py, .so)        │
│  • Verifies: path within CV_DATA_DIR / CV_WORKSPACE          │
│  • Output/write paths confined to the same allowed bases     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  RateLimiter.is_allowed(key)                                 │
│  • 100 req/min per dataset_id                                │
│  • 50 req/min global                                         │
│  • Sliding window                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ResourceLimiter.acquire(tool_name)                          │
│  • Max 4 concurrent tools                                    │
│  • Max 300s execution time                                   │
│  • Max 2GB RAM, 10GB disk                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  TOOL EXECUTES                                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  CredentialFilter.sanitize()                                 │
│  • Redacts: aws_secret, password, token, api_key, conn_str  │
│  • Applied to ALL tool responses                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  SecurityAuditor.log_tool_call()                             │
│  • Timestamp, tool, dataset_id, success/error                │
│  • Violations logged as CRITICAL                             │
│  • Also logs resource access for resources/prompts           │
└─────────────────────────────────────────────────────────────┘
```

**Note**: The security pipeline applies to all 103 MCP components (67 tools, 23 resources, 13 prompts).

**Enforced, not just available**: Every one of the 67 tools is wrapped at a single choke point so that each call is (a) rate-limited per `(tool, dataset)` via a sliding-window limiter, (b) bounded by a max-concurrency slot, (c) audit-logged on both success and violation via `SecurityAuditor`, and (d) exception-guarded — any error is caught and returned as a sanitized string, so stack traces and secrets never leak to the client. (Previously these primitives existed but were not wired into the tool call path.)

---

## Configuration

### Required Environment Variables

```bash
# Data directories (MUST be absolute paths)
export CV_DATA_DIR="/absolute/path/to/datasets"
export CV_WORKSPACE="/absolute/path/to/outputs"

# Audit log (recommended)
export CV_SECURITY_LOG_FILE="/var/log/cv-mcp-security.log"
# or for local dev:
export CV_SECURITY_LOG_FILE="$HOME/.cv-mcp-security.log"
```

### Optional Overrides

```bash
# Allow ANY path (DANGEROUS — disables path confinement; NEVER set in production)
export CV_ALLOW_ALL_PATHS="false"

# Decompression-bomb guard: max decoded image pixels (default ~64MP)
export CV_MAX_IMAGE_PIXELS="67108864"

# Resource limits
export CV_MAX_CONCURRENT="4"
export CV_MAX_EXECUTION_TIME="300"
export CV_MAX_MEMORY_MB="2048"
export CV_MAX_DISK_MB="10240"

# Rate limits
export CV_TOOL_RATE_LIMIT="100"      # req/min per dataset
export CV_DATASET_RATE_LIMIT="50"    # req/min global
```

### Environment Variable Reference

| Variable | Purpose | Default / Notes |
|----------|---------|-----------------|
| `CV_DATA_DIR` | Allowed base directory for datasets (reads) | Absolute path; required |
| `CV_WORKSPACE` | Allowed base directory for outputs (writes) | Absolute path; required |
| `CV_ALLOW_ALL_PATHS` | Disables path confinement | `false` — **NEVER set `true` in production** |
| `CV_MAX_IMAGE_PIXELS` | Decompression-bomb guard on image decode | ~64MP (`67108864`) |
| `CV_SECURITY_LOG_FILE` | Path to the append-only security audit log | Recommended |
| `CV_MCP_AUTH_TOKEN` | Bearer token for HTTP transport auth | Required when hosting over HTTP |
| `MCP_TRANSPORT` | Transport mode (`stdio` or `streamable-http`) | `stdio` |
| `CV_MAX_CONCURRENT` | Max concurrent tool executions | `4` |
| `CV_TOOL_RATE_LIMIT` | Rate limit, req/min per dataset | `100` |
| `CV_DATASET_RATE_LIMIT` | Rate limit, req/min global | `50` |

---

## Network Hosting (HTTP Transport)

By default the server runs over **stdio** and is only reachable by the local LLM client. To host it over the network, set `MCP_TRANSPORT=streamable-http`. In that mode:

```bash
export MCP_TRANSPORT="streamable-http"
export CV_MCP_AUTH_TOKEN="$(openssl rand -hex 32)"   # required
```

- **Bearer-token auth**: every request must present `Authorization: Bearer <CV_MCP_AUTH_TOKEN>`. The token is checked with a **constant-time comparison** to resist timing attacks.
- **Health probe**: `GET /health` is intentionally **unauthenticated** so load balancers / orchestrators can check liveness without the token.
- **TLS**: TLS is **not** built in — terminate TLS at a reverse proxy (e.g. nginx, Caddy, an ingress controller) in front of the server.
- **Least privilege**: run the server as a **non-root** user inside a container.

See `docs/mcp_client_setup.md` for the remote-server hosting walkthrough.

### Cloud Credentials (Never in Tool Params!)

```bash
# AWS
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# GCS
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

# Azure
export AZURE_STORAGE_CONNECTION_STRING=...
```

---

## Audit Log Format

Each line is JSON. Example:

```json
{
  "timestamp": 1700000000.123,
  "event_type": "TOOL_CALL",
  "tool_name": "load_dataset",
  "dataset_id": "coco_train_12345",
  "user_id": null,
  "details": {"success": true, "error": null},
  "severity": "INFO"
}
```

```json
{
  "timestamp": 1700000001.456,
  "event_type": "SECURITY_VIOLATION",
  "tool_name": "load_dataset",
  "dataset_id": null,
  "user_id": null,
  "details": {
    "violation_type": "PATH_TRAVERSAL",
    "attempted_path": "../../../etc/passwd",
    "allowed_bases": ["/home/user/datasets", "/home/user/workspace"]
  },
  "severity": "CRITICAL"
}
```

### Log Rotation (Recommended)

```bash
# /etc/logrotate.d/cv-mcp
/var/log/cv-mcp-security.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 640 root root
}
```

---

## Credential Handling

**NEVER** pass credentials as tool parameters. The server:

1. **Only reads from environment** at startup (via `CredentialManager`)
2. **Caches in memory** (process lifetime only)
3. **Redacts from all outputs** (via `CredentialFilter`)
4. **Never logs** full credentials (only key names in audit)

### Correct Usage

```bash
# User sets once
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...

# LLM calls (no creds in params!)
load_cloud_dataset_tool(provider="s3", bucket="my-data", prefix="train/")
```

### Incorrect (Blocked)

```python
# This will fail — tool signature doesn't accept creds
load_cloud_dataset_tool(
    provider="s3",
    bucket="my-data",
    aws_access_key_id="AKIA...",  # ERROR: not a parameter
    aws_secret_access_key="..."   # ERROR: not a parameter
)
```

---

## Pipeline Safety

Pipelines are JSON definitions executed by the server. The `PipelineSecurityValidator` uses AST parsing to block:

| Blocked Pattern | Example |
|-----------------|---------|
| `import os` / `import subprocess` | `import os; os.system('rm -rf /')` |
| `__import__` | `__import__('os').system('...')` |
| `eval` / `exec` | `eval('malicious_code')` |
| `getattr` / `setattr` on dangerous modules | `getattr(__import__('os'), 'system')` |
| Attribute access on `os`, `sys`, `subprocess`, `shutil` | `os.system`, `subprocess.run` |
| `open` with write mode | `open('/etc/passwd', 'w')` |

Only whitelisted operations allowed: math, string, list/dict manipulation, whitelisted library calls.

---

## Deployment Checklist

- [ ] `CV_DATA_DIR` and `CV_WORKSPACE` set to dedicated directories
- [ ] `CV_ALLOW_ALL_PATHS="false"` (default) — never `true` in production
- [ ] `CV_MAX_IMAGE_PIXELS` set appropriately for your workload
- [ ] `CV_SECURITY_LOG_FILE` pointing to persistent, rotated log
- [ ] Cloud credentials in environment (not config files)
- [ ] Log rotation configured
- [ ] Monitoring on `SECURITY_VIOLATION` events
- [ ] Regular `bandit -r src/` scans in CI
- [ ] Dependency updates via `pip-audit`
- [ ] **If hosting over HTTP**: `CV_MCP_AUTH_TOKEN` set to a strong random value
- [ ] **If hosting over HTTP**: TLS terminated at a reverse proxy; server runs as non-root in a container

---

## Incident Response

1. **Check audit log**: `grep SECURITY_VIOLATION /var/log/cv-mcp-security.log`
2. **Identify tool & path**: `violation_type`, `attempted_path`
3. **Revoke compromised credentials** (if any leaked)
4. **Review LLM client config** for injection vectors
5. **Rotate logs** and archive for forensics