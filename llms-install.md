# Installing PixLint (guide for AI agents / Cline)

PixLint is a **local stdio MCP server** for computer-vision dataset management.
No API keys, accounts, or network services are required — it runs entirely on the
user's machine. Follow these exact steps to install and register it.

## 1. Install the package

PixLint requires **Python 3.10+**.

```bash
pip install pixlint
```

This installs the `pixlint` command, which is the MCP server entry point (stdio transport).

## 2. Register PixLint in the MCP settings

Add this entry to the `mcpServers` object in Cline's MCP settings file
(`cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "pixlint": {
      "command": "pixlint",
      "args": [],
      "env": {
        "CV_DATA_DIR": "/absolute/path/to/your/datasets"
      }
    }
  }
}
```

- `CV_DATA_DIR` — directory PixLint may read datasets from. Set it to the folder that
  contains the user's image datasets. Optional; if omitted, safe default directories are used.
- `CV_WORKSPACE` — (optional) directory for outputs such as exports and curated datasets.

## 3. Verify

After saving the settings, PixLint's tools become available. Confirm by listing tools —
you should see ~67 tools, including `load_dataset`, `find_duplicates_tool`,
`analyze_quality_tool`, `check_integrity_tool`, `dataset_readiness_report_tool`,
`clean_dataset_tool`, `split_dataset_tool`, and `export_dataset_tool`.

## Notes

- Optional features need extras: `pip install "pixlint[torch]"` (embeddings, label-error
  detection, auto-labeling), `pip install "pixlint[huggingface]"` (Hugging Face export).
- PixLint is local-only; it never sends data anywhere.
