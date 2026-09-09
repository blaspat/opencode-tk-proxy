# Hermes LLM Proxy

An OpenAI-compatible proxy for Hermes Agent that sits between Hermes and LLM providers, injecting session headers and compressing input to reduce token costs.

## Features

- **Session Header Injection** — Automatically injects `x-opencode-session` header for opencode-go compatibility
- **Content-Type Aware Compression** — Detects and compresses different content types appropriately
- **Tools-Array Compression** — Truncates verbose tool/function descriptions and parameter schema text (names/types/enums preserved)
- **Structural JSON Compression** — Large JSON tool results keep all keys/numbers/bools; long string values are truncated with a marker
- **Smart Compressors**:
  - Terminal output (ANSI strip, noise removal)
  - Log files (timestamp strip, level filtering)
  - Diff output (header removal, context collapse)
  - HTML content (tag stripping, link preservation)
  - Tool schemas (description truncation)
  - System prompts (section-aware compression)
- **Recovery Store** — SQLite-based storage for lossy transforms with recovery handles
- **Query-Aware Compression** — Optimizes compression based on search context
- **Streaming SSE Support** — Full streaming support for real-time responses
- **Dashboard** — Real-time compression stats at `/dashboard`
- **Stats API** — JSON stats endpoint at `/stats`

## Quick Start

```bash
# Clone the repo
git clone https://github.com/blaspat/hermes-proxy.git
cd hermes-proxy

# Install dependencies
pip install fastapi uvicorn httpx python-dotenv

# Create .env file
cat > .env << 'EOF'
UPSTREAM_URL=https://opencode.ai/zen/go
UPSTREAM_KEY=your-api-key-here
PROXY_PORT=8787
SESSION_ID=
VERBOSE=true
COMPRESS=true
EOF

# Run the proxy
python3 proxy.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `https://opencode.ai/zen/go/v1` | LLM provider endpoint |
| `UPSTREAM_KEY` | - | API key for upstream |
| `PROXY_PORT` | `8787` | Proxy listen port |
| `SESSION_ID` | md5 of UPSTREAM_KEY | Fixed session identifier |
| `VERBOSE` | `true` | Enable debug logging |
| `COMPRESS` | `true` | Enable compression |

## Endpoints

- `/health` — Health check
- `/stats` — Compression statistics JSON (per-request skip-reason breakdown, tools savings, upstream `usage`/`cost` when reported, and local pre/post token estimates via tiktoken)
- `/dashboard` — Real-time dashboard UI
- `/recovery/{handle}` — Retrieve original content
- `/{path}` — Catch-all proxy to upstream

## Usage with Hermes

Configure your Hermes profile to use the proxy:

```yaml
# config.yaml
model:
  provider: custom
```

```bash
# .env
CUSTOM_BASE_URL=http://127.0.0.1:8787/v1
CUSTOM_API_KEY=your-api-key
```

## Use from the opencode CLI

The proxy is an OpenAI-compatible endpoint upstream of opencode's own gateway
(`UPSTREAM_URL` defaults to `https://opencode.ai/zen/go/v1`) and injects the
`x-opencode-session` header opencode expects, so the opencode CLI can be pointed
at it directly and pick up compression automatically. Incoming API keys are
ignored — the proxy always authenticates upstream with its own `UPSTREAM_KEY`.

Add a provider in `opencode.json`:

```json
{
  "provider": {
    "hermes-proxy": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        "apiKey": "***"
      }
    }
  },
  "model": "hermes-proxy/mimo-v2.5"
}
```

Notes:
- The proxy binds `127.0.0.1`, so the CLI must run on the same host (remote
  use requires exposing the port).
- The model id must be one the upstream accepts (the default config's
  `mimo-v2.5` is what the test harness uses).
- CLI traffic appears in `/stats` and the recovery store like any other client.

## Install as Service

```bash
sudo cp hermes-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hermes-proxy
sudo systemctl start hermes-proxy
```

## Compression Ratios

Typical compression ratios based on content type:

| Content Type | Average Savings |
|-------------|-----------------|
| System prompts | 40-50% |
| Terminal output | 50-70% |
| Log files | 30-50% |
| Tool schemas | 20-30% |
| HTML content | 60-80% |

## License

MIT
