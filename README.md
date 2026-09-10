# Opencode LLM Proxy

OpenAI-compatible (and Anthropic Messages-compatible) proxy that filters out the garbage: redirects clients to upstream, injects session headers, fixes common smells and runs lossless input compression before sending.

Here's the pricing model: tokens cost money. Session context and tool metadata are repeated on every call. That's where the money leaks. This proxy shrinks that repeat-tax and keeps a full, recoverable record of anything it truncates.

## Features

- **Session Header Injection** — automatically injects `x-opencode-session`
- **Compression by API surface**:
  - **OpenAI Chat Completions** (`POST /v1/chat/completions`) — messages + tools
  - **OpenAI Responses API** (`POST /v1/responses`) — `input` items (message + `function_call_output`), flat tool format, `instructions`; `reasoning`/`function_call` items always untouched
  - **Anthropic Messages** (`POST /v1/messages`) — `tool_result` blocks (string or content list), large text parts, top-level `system`; `tool_use`/`thinking` blocks never touched
- **Smart compressors per content type** — terminal output (ANSI strip), logs, diffs, HTML, tool schemas, system prompts, JSON (structural: keys/numbers/bools preserved, long string values truncated, keys sorted for byte-stable re-serialization → upstream prompt-cache friendly)
- **Query-Aware Compression** — threads the latest user message through as an anchor; tool-result compression keeps lines matching the query keywords instead of blind head/tail truncation
- **Age-Based History Compression** — `AGE_SPLIT_TURNS` env (default 4, 0 disables): tool results older than N turns compress at ratio 0.15 (aggressive) instead of 0.50
- **Recovery Store** — anything compressed is stored verbatim in SQLite (`/tmp/opencode-tk-proxy-recovery.db`, 1h TTL) with a `[ccr:<handle>]` marker appended; fetch originals back via `/recovery/{handle}`
- **Tool schemas preserved** — names, types, enums, and `required` always survive; only verbose description strings trimmed
- **Stats API** — JSON at `/stats`: per-request entries (with skip-reason breakdown, tool savings, upstream `usage`/`cost`, local tiktoken pre/post estimates), aggregated totals, and `requests_by_path`
- **Dashboard** — live UI at `/dashboard`: aggregate cards, per-endpoint request counts, latest-20 request table
- **Streaming SSE support** — usage/cost parsed from `data:` lines (incl. nested `response.completed` usage)

## Quick Start

```bash
git clone https://github.com/blaspat/opencode-tk-proxy.git
cd opencode-tk-proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cat > .env << 'EOF'
UPSTREAM_URL=https://opencode.ai/zen/go/v1
UPSTREAM_KEY=your-key-here
PROXY_PORT=8787
EOF

python3 proxy.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `https://opencode.ai/zen/go/v1` | LLM provider endpoint |
| `UPSTREAM_KEY` | — | API key for upstream (only injected when the client didn't send one) |
| `PROXY_PORT` | `8787` | Proxy listen port (binds 127.0.0.1) |
| `SESSION_ID` | sha256(UPSTREAM_KEY)[:16] | Fixed session identifier |
| `VERBOSE` | `true` | Debug logging |
| `COMPRESS` | `true` | Enable compression |
| `AGE_SPLIT_TURNS` | `4` | Turns from end of conversation before which tool results compress at 0.15 ratio. `0` disables |

## Endpoints

- `/health` — health check
- `/stats` — compression statistics JSON (`requests_by_path`, skip-reason totals, tools savings, upstream usage/cost, local token estimates)
- `/dashboard` — real-time dashboard UI (requests by endpoint, savings, latest 20 requests)
- `/recovery/{handle}` — retrieve the original content for a `[ccr:<handle>]` marker
- `/{path}` — catch-all proxy to upstream (rejects `..` traversal; forwards unrecognized JSON untouched)

## Use with the opencode CLI

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

The proxy is an OpenAI-compatible endpoint upstream of opencode's own gateway (`UPSTREAM_URL` defaults to `https://opencode.ai/zen/go/v1`) and injects the `x-opencode-session` header opencode expects. Incoming API keys are passed through when present; the proxy injects its own `UPSTREAM_KEY` Authorization only when the header is missing.

Notes:
- The proxy binds `127.0.0.1`, so clients must run on the same host (remote use requires exposing the port).
- When overriding base URLs in clients, include the `/v1` suffix (e.g. `http://127.0.0.1:8787/v1`) — most clients only auto-append `/v1` for well-known hostnames.
- Traffic appears in `/stats` and the recovery store regardless of which surface the client uses.

---

## For AI Agents — Automated Install

**You are an AI agent (Claude Code, Cursor, Codex, Hermes, opencode, etc.) tasked with installing this proxy for your user.** Follow exactly; no human judgment required.

### Step 1 — Clone and set up

```bash
git clone https://github.com/blaspat/opencode-tk-proxy.git
cd opencode-tk-proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Configure

Ask the user (or read from your environment) for:
- `UPSTREAM_URL` — the LLM provider endpoint (default `https://opencode.ai/zen/go/v1`; any OpenAI-compatible `/v1` base works, Anthropic `/v1/messages` is forwarded + compressed too)
- `UPSTREAM_KEY` — the API key

Then:

```bash
cat > .env << EOF
UPSTREAM_URL=<url from user>
UPSTREAM_KEY=<key from user>
PROXY_PORT=8787
EOF
```

### Step 3 — Smoke test

```bash
python3 proxy.py &          # or run in your own long-lived process manager
sleep 2
curl -s localhost:8787/health
```

Pass: `{"status":"ok","compress":true,...}`. If not, check the port is free (`ss -tlnp | grep 8787`) and the `.env` values are correct.

### Step 4 — Wire your client

Point any OpenAI-compatible client's base URL at `http://127.0.0.1:8787/v1`.
- **Hermes**: set `OPENCODE_GO_BASE_URL=http://127.0.0.1:8787/v1` in the profile's `.env` (must include `/v1`).
- **opencode CLI**: see the JSON snippet above in this README.
- **Anthropic-format clients**: point base URL at `http://127.0.0.1:8787` — `/v1/messages` is handled directly.

Send one test request, then `curl -s localhost:8787/stats | python3 -m json.tool | head -20` — pass when the last entry has a nonzero `chars_saved`.

### Step 5 — Install as a systemd service (production)

```bash
sudo cp opencode-tk-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now opencode-tk-proxy
```

The service file already points at `~/.hermes/artifacts/claire/opencode-tk-proxy` — if you cloned elsewhere, edit `WorkingDirectory` and `ExecStart` in the unit file first.

### Hard rules for agents

- **Never change compression logic** without running `python3 -m unittest -v test_proxy` (all green required).
- **Never log or echo `UPSTREAM_KEY`.**
- Recovery handles: canonical marker is `[ccr:<hex>]` appended to compressed content — split on it before re-parsing JSON.
- Do not add Authorization overrides beyond the existing inject-when-missing behavior.

## Install as systemd Service (manual)

```bash
git clone https://github.com/blaspat/opencode-tk-proxy.git
cd opencode-tk-proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo cp opencode-tk-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now opencode-tk-proxy
sudo systemctl status opencode-tk-proxy
```

## Compression Ratios

Measured on the proxy's own tests (N=5 A/B harness + real recovery-store traffic, Sep 2026). Compression acts on the request **input only** — never responses; tool definitions are never removed (only long `description` strings are trimmed).

| Content type | Measured savings |
|-------------|-----------------|
| Chat + tool-result JSON (real traffic) | ~50–65% (whole real chat payload 54.5%) |
| Git / diff output | ~48% |
| Documentation text | ~49% |
| Log files | ~42% |
| Shell / terminal output | ~44% |
| Tool schemas (description strings only) | ~25% |
| Short prose (<100 chars) / verbatim code | 0% — untouched by design |

Production average on live traffic: ~40–60% depending on tool-heavy vs prose-heavy sessions.

## License

MIT
