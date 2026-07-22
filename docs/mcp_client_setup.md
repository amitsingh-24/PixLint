# MCP Client Setup Guide

## Overview

This server (v1.0) uses the **Model Context Protocol (MCP)**. By default it runs over **stdio** as a subprocess of your MCP client — no network, no HTTP, no auth tokens. It can also be hosted over the network (`streamable-http` / `sse`) with bearer-token auth for remote clients — see [Remote/HTTP Server](#remotehttp-server) below.

## Supported Clients

| Client | Config File | Status |
|--------|-------------|--------|
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` | ✅ Full |
| **Cursor** | `.cursor/mcp.json` or `~/.cursor/mcp.json` | ✅ Full |
| **VS Code (MCP extension)** | `.vscode/mcp.json` | ✅ Full |
| **Windsurf** | `~/.codeium/windsurf/mcp_config.json` | ✅ Full |
| **Custom Python** | See below | ✅ Full |

---

## 1. Claude Desktop (macOS)

**Config:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": {
        "CV_DATA_DIR": "/Users/YOURNAME/datasets",
        "CV_WORKSPACE": "/Users/YOURNAME/cv-workspace",
        "CV_SECURITY_LOG_FILE": "/Users/YOURNAME/.cv-mcp-security.log"
      }
    }
  }
}
```

**Restart Claude Desktop** (Cmd+Q → reopen). Tools appear in chat.

---

## 2. Cursor

**Config:** `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global)

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": {
        "CV_DATA_DIR": "${workspaceFolder}/datasets",
        "CV_WORKSPACE": "${workspaceFolder}/cv-workspace"
      }
    }
  }
}
```

Reload: `Cmd+Shift+P` → "MCP: Reload Servers"

---

## 3. VS Code (MCP Extension)

**Config:** `.vscode/mcp.json`

```json
{
  "servers": {
    "pixlint": {
      "command": "pixlint",
      "env": {
        "CV_DATA_DIR": "${workspaceFolder}/data",
        "CV_WORKSPACE": "${workspaceFolder}/output"
      }
    }
  }
}
```

---

## 4. Windsurf

**Config:** `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": {
        "CV_DATA_DIR": "/home/user/datasets",
        "CV_WORKSPACE": "/home/user/cv-workspace"
      }
    }
  }
}
```

---

## 5. Custom Python Client

```python
import asyncio
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

async def main():
    # Start the server
    async with stdio_client("pixlint", env={
        "CV_DATA_DIR": "/path/to/data",
        "CV_WORKSPACE": "/path/to/work"
    }) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {len(tools.tools)}")
            
            # Call a tool
            result = await session.call_tool("load_dataset", {
                "path": "/path/to/data",
                "format": "yolo"
            })
            print(result.content[0].text)

asyncio.run(main())
```

---

## 6. Docker (Advanced)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install pixlint
ENV CV_DATA_DIR=/data
ENV CV_WORKSPACE=/workspace
VOLUME ["/data", "/workspace"]
ENTRYPOINT ["pixlint"]
```

```bash
docker run -it --rm \
  -v ~/datasets:/data \
  -v ~/cv-workspace:/workspace \
  -e CV_DATA_DIR=/data \
  -e CV_WORKSPACE=/workspace \
  pixlint
```

---

## Remote/HTTP Server

The server can also be hosted over the network for internet-accessible deployments instead of running as a local subprocess. Set the transport via environment variables when starting the server:

| Variable | Values | Description |
|----------|--------|-------------|
| `MCP_TRANSPORT` | `stdio` (default) / `streamable-http` / `sse` | Transport protocol |
| `MCP_HOST` | e.g. `0.0.0.0` | Bind address (network transports) |
| `MCP_PORT` | e.g. `8000` | Bind port (network transports) |
| `CV_MCP_AUTH_TOKEN` | any secret string | Bearer token required on all requests |
| `CV_DATA_DIR` | path | Datasets are read only from allowed base dirs |

Start a hosted server:

```bash
export MCP_TRANSPORT=streamable-http
export MCP_HOST=0.0.0.0
export MCP_PORT=8000
export CV_MCP_AUTH_TOKEN="your-secret-token"
export CV_DATA_DIR="/srv/datasets"
pixlint
```

An unauthenticated `GET /health` endpoint is available for readiness/liveness checks.

Point an MCP client at the hosted server (`streamable-http`):

```json
{
  "mcpServers": {
    "pixlint": {
      "url": "http://host:8000/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-token"
      }
    }
  }
}
```

For production hosting, terminate TLS at a reverse proxy, run as a non-root user, set `CV_MCP_AUTH_TOKEN`, and restrict datasets to `CV_DATA_DIR`. See the [Security Guide](security.md) for the full threat model and checklist.

---

## Verifying Connection

After setup, restart your MCP client and ask:

> "What MCP tools are available?"

You should see **103 MCP components**:
- **67 tools** including: `load_dataset`, `list_datasets`, `dataset_info`, `find_duplicates`, `analyze_quality`, `check_integrity`, `analyze_distribution`, `compute_embeddings`, `semantic_search`, `augment_dataset`, `resize_dataset`, `normalize_dataset`, `split_dataset`, `export_dataset` (pytorch/tensorflow/ultralytics/hdf5), `execute_pipeline`, `register_pipeline`, `detect_outliers`, `merge_datasets`, `load_cloud_dataset`, `uncertainty_sampling`, `query_strategy`, plus newer additions such as `compute_statistics_tool`, `sample_dataset_tool`, `unload_dataset_tool`, `filter_dataset_tool` (subset), `clean_dataset_tool` (fix corrupt/out-of-bounds/degenerate/duplicates), `remap_classes_tool`, `auto_label_tool` (torchvision COCO-80, needs `[torch]`), `export_huggingface_tool` (needs `[huggingface]`), `find_label_errors_tool` (mislabel detection), `query_dataset_tool` (NL→predicate query), `dataset_readiness_report_tool` ("Dataset Doctor" with executable remediation), `discover_slices_tool` (weak-slice/bias discovery), and more
- **23 resources** including: `dataset://{id}/info`, `dataset://{id}/images`, `dataset://{id}/quality`, `dataset://{id}/health`, `dataset://{id}/class_stats`, `dataset://{id}/statistics`, `dataset://list`, etc.
- **13 prompts** including: `prepare_yolo_training_prompt`, `active_learning_prompt`, `dataset_versioning_prompt`, `export_pipeline_prompt`, `dataset_curation_prompt`, `publish_dataset_prompt`, etc.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Command not found" | Ensure `pixlint` is in PATH (run `which pixlint`) |
| "Connection closed" | Check server logs: `CV_SECURITY_LOG_FILE=/tmp/log pixlint` |
| "Path not allowed" | Set `CV_DATA_DIR` to parent of your dataset |
| "Tools not showing" | Restart client fully (not just reload) |
| "Permission denied" | Ensure `CV_WORKSPACE` is writable by your user |

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CV_DATA_DIR` | ✅ | - | Root for input datasets |
| `CV_WORKSPACE` | ✅ | - | Root for all outputs |
| `CV_SECURITY_LOG_FILE` | ❌ | `/tmp/pixlint_security.log` | Audit log path |
| `CV_ALLOW_ALL_PATHS` | ❌ | `false` | **Dev only** - disable path checks |
| `AWS_*`, `GOOGLE_*`, `AZURE_*` | ❌ | - | Cloud credentials (env only!) |

---

## What's Exposed via MCP

| Component Type | Count | Description |
|----------------|-------|-------------|
| **Tools** | 67 | Callable operations with full validation |
| **Resources** | 23 | Read-only data endpoints (dataset info, quality scores, class stats, statistics, etc.) |
| **Prompts** | 13 | Reusable prompt templates for common CV workflows |
| **Total** | **103** | All with input validation, path checking, rate limiting, resource limits, audit logging |