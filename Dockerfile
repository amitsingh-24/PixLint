# =============================================================================
# pixlint — production image
# =============================================================================
# Runtime environment variables to set with `docker run -e ...`:
#
#   CV_MCP_AUTH_TOKEN   Bearer token required to authenticate MCP requests.
#   CV_DATA_DIR         Absolute path (inside the container) to the allowed
#                       dataset directory. Mount your data here, e.g.
#                       `-v /host/datasets:/data -e CV_DATA_DIR=/data`.
#   HF_TOKEN            Optional. Hugging Face access token, required only for
#                       publishing datasets via push_to_hub / the [huggingface]
#                       extra.
#
# Example:
#   docker run --rm -p 8000:8000 \
#     -e CV_MCP_AUTH_TOKEN=change-me \
#     -e CV_DATA_DIR=/data \
#     -e HF_TOKEN=hf_xxx \
#     -v /host/datasets:/data \
#     pixlint
# =============================================================================

FROM python:3.11-slim

# Install curl for the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the server.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

# Copy the project and install it (plus uvicorn for the HTTP transport).
COPY . /app
RUN pip install --no-cache-dir . uvicorn

# Default MCP transport / networking configuration.
ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

EXPOSE 8000

# Drop privileges.
USER app

# Health check against the app's /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["pixlint"]
