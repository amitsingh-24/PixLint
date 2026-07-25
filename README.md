<p align="center">
  <img src="https://raw.githubusercontent.com/amitsingh-24/PixLint/main/assets/pixlint-logo.png" alt="PixLint" width="128" height="128" />
</p>

# PixLint

**Lint, curate, and prepare computer-vision datasets — right from your AI assistant.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-orange.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-purple.svg)](https://modelcontextprotocol.io)

PixLint is an [MCP](https://modelcontextprotocol.io) server that gives AI assistants — Claude, Cursor, VS Code, and any MCP client — direct, conversational access to a complete computer-vision dataset toolkit: analyze quality, find duplicates and label errors, clean and curate, split, augment, convert formats, and export to every major training framework.

It runs locally over stdio, or self-hosted on the internet over authenticated HTTP.

---

## Why PixLint

Most dataset tooling is either a paid SaaS or a heavy GUI app. PixLint is a single, open-source, self-hostable server an AI agent can drive end to end — and it does things others keep behind paid tiers:

- **🩺 Dataset Doctor** — one call runs a full diagnostic and returns a prioritized, *executable* fix plan.
- **Label-error detection** — automatically surface images that are probably mislabeled.
- **Natural-language query** — *"find blurry images with a person on the left"*, answered over your data.
- **Weak-slice discovery** — find under-represented or low-quality slices to collect or augment next.
- **Curation that writes a new dataset** — clean / filter / remap, not just report.
- **Auto-labeling** with a pretrained detector, and **one-command Hugging Face publishing**.

---

## Features

**103 operations** — 67 tools, 23 resources, 13 prompts.

| Category | What you get |
|----------|--------------|
| **Load** | COCO · VOC · YOLO · KITTI · folder, plus cloud (S3 / GCS / Azure) |
| **Analyze** | Duplicates · quality (blur/exposure/noise/contrast) · integrity · class distribution · embeddings · semantic search · outliers · health score |
| **Data intelligence** | Dataset Doctor readiness report · label-error detection · natural-language query · weak-slice / bias discovery |
| **Curate** | Filter to a subset · clean (corrupt / out-of-bounds / degenerate / duplicates) · remap classes — each produces a new dataset |
| **Augment & transform** | YOLO/classification/segmentation pipelines · resize · normalize · format conversion |
| **Split** | Stratified / random / temporal / grouped · k-fold · data-leakage detection |
| **Auto-label** | Pretrained COCO-80 detector → pre-annotated dataset |
| **Export & publish** | PyTorch · TensorFlow · Ultralytics · HDF5 · WebDataset · FiftyOne · CVAT · LabelMe · Hugging Face Hub |
| **Pipelines** | Compose multi-step workflows and reuse pre-built templates |

---

## Quick Start

### 1. Install

```bash
pip install pixlint
```

Optional extras add heavier capabilities:

```bash
pip install "pixlint[torch]"        # embeddings, auto-labeling, label-error detection
pip install "pixlint[huggingface]"  # Hugging Face export + publishing
pip install "pixlint[all]"          # everything
```

### 2. Connect your AI assistant

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": { "CV_DATA_DIR": "/path/to/your/datasets" }
    }
  }
}
```

**Cursor / VS Code** — `.cursor/mcp.json` or `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "env": { "CV_DATA_DIR": "/path/to/your/datasets" }
    }
  }
}
```

`CV_DATA_DIR` is the directory PixLint is allowed to read datasets from.

### 3. Just ask

> *"Load my dataset at `/data/coco_person`, give it a readiness report, then clean it and export for YOLO."*

Your assistant calls the right PixLint tools in sequence — diagnose, clean, split, export — and hands back a training-ready dataset.

---

## Security

PixLint touches the filesystem and can be exposed to a network, so protections run on **every** tool call:

- Paths are confined to your configured data directory (reads **and** writes).
- Credentials come only from environment variables, never tool inputs.
- Per-call rate limiting, concurrency limits, and audit logging.
- Decompression-bomb protection on image decode.
- Optional bearer-token authentication for the HTTP transport.

See the [Security Guide](docs/security.md) for the full threat model and the recommended production checklist.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting_started.md) | Installation, configuration, first steps |
| [MCP Client Setup](docs/mcp_client_setup.md) | Claude, Cursor, VS Code, and remote/HTTP hosting |
| [API Reference](docs/api_reference.md) | All 67 tools with parameters |
| [Security Guide](docs/security.md) | Threat model, configuration, hosting |
| [Pipeline Templates](docs/pipeline_templates.md) | Pre-built and custom pipelines |

Runnable scripts live in [`examples/`](examples/). See [CHANGELOG.md](CHANGELOG.md) for release notes.

---

## License

PixLint is source-available under the **PolyForm Strict License 1.0.0** — see [LICENSE](LICENSE).
You may use it for permitted (noncommercial) purposes; commercial use, redistribution, or
modification requires a separate license from the copyright holder. Contributions are welcome
via pull request.

---

<sub>mcp-name: io.github.amitsingh-24/pixlint</sub>
