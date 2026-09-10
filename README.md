# Opencode LLM Proxy

An OpenAI-compatible proxy for Opencode, injecting session headers and compressing input to reduce token costs.

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
git clone https://github.com/blaspat/opencode-tk-proxy.git
cd opencode-tk-proxy

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << 'EOF'
UPSTREAM_URL=https://opencode.ai/zen/go
UPSTREAM_KEY=
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

Configure your Hermes profile to use the proxy by overriding the `OPENCODE_GO_BASE_URL` or `OPENCODE_ZEN_BASE_URL`:

```bash
# .env
OPENCODE_GO_BASE_URL=http://127.0.0.1:8787/v1
OPENCODE_ZEN_BASE_URL=http://127.0.0.1:8787/v1
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
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "opencode-go": {
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1"
      }
    }
  }
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
sudo cp opencode-tk-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable opencode-tk-proxy
sudo systemctl start opencode-tk-proxy
```

## Compression Ratios

Measured on the proxy's own tests (N=5 A/B harness + real recovery-store
traffic, Sep 2026). Compression acts on the request **input only** — never
responses; tool definitions are never removed (only long `description`
strings are trimmed).

| Content type | Measured savings |
|-------------|-----------------|
| Chat + tool-result JSON (real traffic) | ~50-65% (whole real chat payload 54.5%) |
| Git / diff output | ~48% |
| Documentation text | ~49% |
| Log files | ~42% |
| Shell / terminal output | ~44% |
| Tool schemas (description strings only) | ~25% |
| Short prose (<100 chars) / verbatim code | 0% — untouched by design |

Production average on live traffic (`/stats` `avg_compression_ratio`): ~43%.

## Install as systemd Service

```bash
# Clone the repo
git clone https://github.com/blaspat/hermes-proxy.git
cd hermes-proxy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your config
cat > .env <<EOF
UPSTREAM_URL=https://your-llm-provider.com/v1
UPSTREAM_KEY=
PROXY_PORT=8787
EOF

# Create the service file
sudo tee /etc/systemd/system/hermes-proxy.service > /dev/null <<EOF
[Unit]
Description=Hermes LLM Proxy
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python3 $(pwd)/proxy.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable hermes-proxy
sudo systemctl start hermes-proxy

# Check status
sudo systemctl status hermes-proxy
```

## License

MIT
