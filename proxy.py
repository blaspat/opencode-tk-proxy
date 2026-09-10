#!/usr/bin/env python3
"""
OpenAI-compatible proxy for Hermes → opencode-go.
Handles x-opencode-session header injection so Hermes can use provider: custom.
Compresses tool results and compressible content before forwarding upstream.

Env:
  UPSTREAM_URL   — opencode-go endpoint (default: https://opencode.ai/zen/go/v1)
  UPSTREAM_KEY   — API key for upstream
  PROXY_PORT     — port (default: 8787)
  SESSION_ID     — fixed x-opencode-session value (default: sha256 of UPSTREAM_KEY)
  VERBOSE        — logging (default: true)
  COMPRESS       — enable compression (default: true)
"""

import os
import hashlib
from collections import Counter
import re
import json
import time
import sqlite3
import logging
import threading
import asyncio
import uuid
from collections import deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response, JSONResponse

# ─── Config ────────────────────────────────────────────────────────────────────

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "https://opencode.ai/zen/go/v1").rstrip("/")
UPSTREAM_KEY = os.getenv("UPSTREAM_KEY", "")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8787"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"
COMPRESS_ENABLED = os.getenv("COMPRESS", "true").lower() == "true"
# Turn-of-conversation before which tool results are treated as deep history
# and compressed hard (ratio 0.15 vs normal 0.50). 0 disables.
AGE_SPLIT_TURNS = int(os.getenv("AGE_SPLIT_TURNS", "4"))

# Fix #5: Use SHA256 instead of MD5 for SESSION_ID
SESSION_ID = os.getenv("SESSION_ID", "")
if not SESSION_ID and UPSTREAM_KEY:
    SESSION_ID = hashlib.sha256(UPSTREAM_KEY.encode()).hexdigest()[:16]
elif not SESSION_ID:
    SESSION_ID = "opencode-tk-proxy"

# ─── Recovery Store (SQLite) ─────────────────────────────────────────────────

_RECOVERY_DB = "/tmp/opencode-tk-proxy-recovery.db"
_recovery_local = threading.local()

def _get_recovery_conn() -> sqlite3.Connection:
    """Thread-safe SQLite connection for recovery store."""
    if not hasattr(_recovery_local, "conn") or _recovery_local.conn is None:
        conn = sqlite3.connect(_RECOVERY_DB, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS originals (
                id TEXT PRIMARY KEY,
                content BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON originals(created_at)")
        conn.commit()
        _recovery_local.conn = conn
    return _recovery_local.conn

def _recovery_store(content: str) -> str:
    """Store original content, return handle like 'ccr_<8hex>'."""
    if not content:
        return ""
    content_bytes = content.encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
    handle = f"ccr_{content_hash[:8]}"
    try:
        conn = _get_recovery_conn()
        conn.execute(
            "INSERT OR REPLACE INTO originals (id, content, created_at) VALUES (?, ?, ?)",
            (handle, content_bytes, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except Exception as e:
        log.warning("Recovery store failed: %s", e)
    return handle

def _recovery_get(handle: str) -> str | None:
    """Retrieve original content by handle."""
    try:
        conn = _get_recovery_conn()
        row = conn.execute("SELECT content FROM originals WHERE id = ?", (handle,)).fetchone()
        return row[0].decode("utf-8") if row else None
    except Exception as e:
        log.warning("Recovery get failed: %s", e)
        return None

def _recovery_cleanup():
    """Delete entries older than 1 hour."""
    try:
        conn = _get_recovery_conn()
        cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() - 3600))
        deleted = conn.execute("DELETE FROM originals WHERE created_at < ?", (cutoff,)).rowcount
        conn.commit()
        if deleted:
            log.debug("Recovery cleanup: deleted %d old entries", deleted)
    except Exception as e:
        log.warning("Recovery cleanup failed: %s", e)

def _recovery_stats() -> dict:
    """Return recovery store stats."""
    try:
        conn = _get_recovery_conn()
        row = conn.execute("SELECT COUNT(*) FROM originals").fetchone()
        total = row[0] if row else 0
        total_size = conn.execute("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM originals").fetchone()[0]
        return {"entries": total, "total_size_bytes": total_size}
    except Exception:
        return {"entries": 0, "total_size_bytes": 0}

# ─── Stats tracking ──────────────────────────────────────────────────────────
REQUEST_STATS: deque[dict] = deque(maxlen=100)

# ── Token estimation (best-effort, local) ────────────────────────────────────
_TOKEN_ENC = None

def _get_encoder():
    """Lazily load tiktoken o200k encoder; returns None if unavailable."""
    global _TOKEN_ENC
    if _TOKEN_ENC is None:
        try:
            import tiktoken
            _TOKEN_ENC = tiktoken.get_encoding("o200k_base")
        except Exception:
            _TOKEN_ENC = False
    return _TOKEN_ENC or None


def _estimate_tokens(text: str) -> int:
    """Estimate token count: tiktoken o200k when available, else chars/4 heuristic."""
    if not text:
        return 0
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _find_entry(req_id: str) -> dict | None:
    """Find a stats entry by its request id (stats deque is small)."""
    for e in REQUEST_STATS:
        if e.get("req_id") == req_id:
            return e
    return None


async def _backfill_token_stats(req_id: str, original: bytes, compressed: bytes):
    """Compute pre/post-compression token estimates off the event loop,
    then attach them to the request's stats entry."""
    try:
        orig_tok = await asyncio.to_thread(_estimate_tokens, original.decode("utf-8", errors="replace"))
        comp_tok = await asyncio.to_thread(_estimate_tokens, compressed.decode("utf-8", errors="replace"))
    except Exception:
        return
    entry = _find_entry(req_id)
    if entry is not None:
        entry["tokens_est_original"] = orig_tok
        entry["tokens_est_compressed"] = comp_tok
        entry["tokens_est_saved"] = orig_tok - comp_tok
        entry["tokens_est_method"] = "tiktoken" if _get_encoder() is not None else "chars4"

logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("opencode-tk-proxy")

def _extract_query(payload: dict) -> str | None:
    """Latest user text from Chat/Responses input for query-aware compression."""
    sequences = [payload.get("messages"), payload.get("input")]
    for items in sequences:
        if isinstance(items, str) and len(items) >= 20:
            return items[:500]
        if not isinstance(items, list):
            continue
        for item in reversed(items):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str) and len(content) >= 20:
                return content[:500]
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                )
                if len(text) >= 20:
                    return text[:500]
    return None


def _canonical_path(path: str) -> str:
    """Normalize a routed API path without changing its upstream endpoint name."""
    return path.rstrip("/")


def _upstream_url(path: str, query: str = "") -> str:
    """Join an upstream base and API path without duplicating the /v1 prefix."""
    base = UPSTREAM_URL.rstrip("/")
    if base.endswith("/v1") and (path == "v1" or path.startswith("v1/")):
        base = base[:-3]
    url = f"{base}/{path}"
    return f"{url}?{query}" if query else url


def _compress_recoverable_text(text: str, role: str, query: str | None = None,
                               msg_age: int = 0) -> tuple[str, int]:
    """Compress one text block and retain its original behind a recovery marker."""
    if not text or len(text) <= 200:
        return text, 0
    label = _classify_message(text, role)
    if label != "compressible" and not (role == "tool" and _maybe_json_content(text)):
        return text, 0
    compressed = _compress_for_role(text, role, query, msg_age)
    handle = _recovery_store(text)
    marker = f"\n[ccr:{handle}]"
    if len(compressed) + len(marker) >= len(text):
        return text, 0
    return compressed + marker, len(text) - len(compressed) - len(marker)


# ─── Classifier (inlined from context-bridge/classifier.py) ───────────────────

_CODE_PATTERNS = [
    re.compile(r"```"),                     # code blocks
    re.compile(r"^\s*\{"),                  # JSON objects
    re.compile(r"^\s*\["),                  # JSON arrays
    re.compile(r"Traceback\s+\(most recent"), # stack traces
    re.compile(r"Error:\s"),                # error messages
    re.compile(r"^\s*def\s+\w+\("),         # Python function defs
    re.compile(r"^\s*class\s+\w+"),          # class definitions
    re.compile(r"^\s*(import|from)\s+\w+"),  # imports
    re.compile(r"tool_use_id"),             # tool identifiers
    re.compile(r'"type"\s*:\s*"function"'), # tool schemas
    re.compile(r'"parameters"\s*:\s*\{'),   # JSON schema
    re.compile(r"<thinking>"),              # thinking blocks
    re.compile(r"^\s*curl\s"),              # curl commands
    re.compile(r"HTTP/\d"),                 # HTTP responses
]

def _is_verbatim(content: str) -> bool:
    """Check if content contains code/JSON/structured data that must not be altered."""
    if not content or len(content) < 50:
        return True
    for pat in _CODE_PATTERNS:
        if pat.search(content):
            return True
    return False

def _classify_message(content: str, role: str, has_tool_calls: bool = False) -> str:
    """Classify a message: 'verbatim' or 'compressible'."""
    if not content:
        return "verbatim"
    if has_tool_calls:
        return "verbatim"
    if role == "tool":
        if _is_verbatim(content):
            return "verbatim"
        return "compressible"
    if role == "system":
        return "compressible"
    if _is_verbatim(content):
        return "verbatim"
    return "compressible"

# ─── Compressor (inlined from context-bridge/compressor.py) ────────────────────

def _compress_text(text: str, ratio: float = 0.50) -> str:
    """Compress text to approximately ratio of original size.
    Deduplicates lines, collapses whitespace, truncates middle if needed."""
    if not text or len(text) < 100:
        return text

    original_len = len(text)
    target_len = int(original_len * ratio)

    # Step 1: remove exact duplicate lines (preserving order)
    lines = text.split("\n")
    seen = set()
    deduped = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            deduped.append(line)
            continue
        if stripped not in seen:
            seen.add(stripped)
            deduped.append(line)
    text = "\n".join(deduped)

    # Step 2: collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 3: if still over target, truncate by removing middle paragraphs
    if len(text) > target_len:
        text = _truncate_to_target(text, target_len)

    return text

def _truncate_to_target(text: str, target_len: int) -> str:
    """Truncate text to target length, keeping beginning and end."""
    if len(text) <= target_len:
        return text

    paragraphs = re.split(r"\n\n+", text)

    if len(paragraphs) <= 2:
        return text[:target_len - 20] + "\n[...truncated...]"

    n = len(paragraphs)
    keep_head = max(1, n // 3)
    keep_tail = max(1, n // 3)

    head = paragraphs[:keep_head]
    tail = paragraphs[-keep_tail:]

    result = "\n\n".join(head) + "\n\n[...compressed: {} paragraphs reduced to {}...]\n\n".format(
        n, keep_head + keep_tail
    ) + "\n\n".join(tail)

    if len(result) > target_len:
        result = result[:target_len - 20] + "\n[...truncated...]"

    return result

def _compress_tool_result(text: str, ratio: float = 0.50) -> str:
    """Specialized compression for tool results."""
    if not text or len(text) < 200:
        return text

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("[") or "```" in text:
        return text

    return _compress_text(text, ratio)

# ─── Content-type auto-detection ────────────────────────────────────────────

_JSON_RE = re.compile(r"^\s*[\[{]")
_LOG_LEVEL_RE = re.compile(r"\b(?:INFO|WARN(?:ING)?|ERROR|DEBUG|FATAL|CRITICAL|NOTICE|TRACE)\b")
_LOG_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
_DIFF_HEADER_RE = re.compile(r"^@@\s+-\d+.*\+\d+.*\s+@@|^---\s+a/|^\+\+\+\s+b/")
_CODE_FUNC_RE = re.compile(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+|^\s*(?:import|from)\s+\w+|^\s*async\s+def\s+|^\s*try:|^\s*except\s|^\s*for\s+\w+\s+in\s+|^\s*return\s|^\s*yield\s|^\s*if\s+__name__")
_ANSI_FULL_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")
_PROGRESS_RE = re.compile(r"\[=>+\-# ]*\s*\d+%|\█|▓|░")
_TOOL_SCHEMA_RE = re.compile(r'"type"\s*:\s*"function"|"parameters"\s*:\s*\{|"properties"\s*:\s*\{|"required"\s*:\s*\[')
_HTML_TAG_RE = re.compile(r"<(?:html|div|table|tr|td|th|p|span|body|head|section|article|header|footer|nav|main|form|input|button|img|a |link |script|style)\b", re.IGNORECASE)

def _detect_content_type(text: str) -> str:
    """Classify content into: json, log, terminal, code, diff, tool_schema, html, text. <1ms."""
    if not text:
        return "text"
    trimmed = text.strip()

    # diff: check first (very specific pattern)
    if _DIFF_HEADER_RE.search(trimmed[:500]):
        return "diff"

    # terminal: ANSI codes or progress bars
    if _ANSI_FULL_RE.search(trimmed[:2000]) or _PROGRESS_RE.search(trimmed[:2000]):
        return "terminal"

    # html: has HTML tags
    if _HTML_TAG_RE.search(trimmed[:500]):
        return "html"

    # tool_schema: JSON with function/parameters structure
    if _JSON_RE.match(trimmed) and _TOOL_SCHEMA_RE.search(trimmed[:2000]):
        return "tool_schema"

    # json: starts with { or [, has key:value
    if _JSON_RE.match(trimmed) and ('"' in trimmed or ':' in trimmed):
        return "json"

    # log: timestamps + log levels
    sample = trimmed[:4000]
    has_ts = bool(_LOG_TS_RE.search(sample))
    has_level = bool(_LOG_LEVEL_RE.search(sample))
    if has_ts and has_level:
        return "log"
    # Also: many lines with log levels → log
    if has_level:
        level_count = len(_LOG_LEVEL_RE.findall(sample))
        line_count = sample.count('\n') + 1
        if line_count > 3 and level_count / max(line_count, 1) > 0.3:
            return "log"

    # code: function defs, class defs, imports
    code_hits = 0
    for line in trimmed.split('\n')[:30]:
        if _CODE_FUNC_RE.match(line):
            code_hits += 1
            if code_hits >= 2:
                return "code"

    return "text"

# ─── Log compressor ──────────────────────────────────────────────────────────

_LOG_LINE_TS_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*")
_KEEP_LEVEL_RE = re.compile(r"\b(?:ERROR|WARN(?:ING)?|FATAL|CRITICAL|PANIC)\b", re.IGNORECASE)
_DROP_LEVEL_RE = re.compile(r"\b(?:DEBUG|TRACE)\b", re.IGNORECASE)
_PATH_LN_RE = re.compile(r"(/[^\s:]+):(\d+)")

def _compress_log(text: str, ratio: float = 0.40) -> str:
    """Compress log files: drop DEBUG, collapse repeats, keep errors, truncate."""
    if not text or len(text) < 100:
        return text

    original_len = len(text)
    lines = text.split('\n')

    # Step 1: Strip timestamps, drop DEBUG/TRACE lines
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        # Always keep ERROR/WARN/FATAL
        is_important = bool(_KEEP_LEVEL_RE.search(stripped))

        # Drop DEBUG and TRACE
        if _DROP_LEVEL_RE.search(stripped) and not is_important:
            continue

        # Strip timestamp prefix
        stripped = _LOG_LINE_TS_RE.sub("", stripped)

        # Truncate long lines (>200 chars), preserving file:line
        path_match = _PATH_LN_RE.search(stripped)
        if len(stripped) > 200:
            prefix = f"[{path_match.group(1)}:{path_match.group(2)}] " if path_match else ""
            # Keep first 100 + last 50 chars
            stripped = stripped[:100] + f"... [...{len(stripped)} chars total] ..." + stripped[-50:]
            if prefix and prefix not in stripped:
                stripped = prefix + stripped

        kept.append(stripped)

    # Step 2: Collapse repeated identical lines
    collapsed = []
    i = 0
    while i < len(kept):
        current = kept[i]
        if current == "":
            collapsed.append(current)
            i += 1
            continue
        count = 1
        while i + count < len(kept) and kept[i + count] == current:
            count += 1
        if count > 1:
            collapsed.append(f"{current} [×{count}]")
        else:
            collapsed.append(current)
        i += count

    # Step 3: Remove consecutive blank lines
    deduped = []
    prev_blank = False
    for line in collapsed:
        if line == "":
            if not prev_blank:
                deduped.append(line)
            prev_blank = True
        else:
            prev_blank = False
            deduped.append(line)

    result = "\n".join(deduped).strip()

    # Step 4: If still too large, keep first/last 20% BUT always keep ERROR/WARN/FATAL
    target_len = int(original_len * ratio)
    if len(result) > target_len:
        result_lines = result.split('\n')
        n = len(result_lines)
        head_count = max(1, n // 5)
        tail_count = max(1, n // 5)
        
        # Always collect important lines (ERROR/WARN/FATAL)
        important_indices = set()
        important_lines = []
        for idx, line in enumerate(result_lines):
            if _KEEP_LEVEL_RE.search(line):
                important_indices.add(idx)
                important_lines.append(line)
        
        # Build kept lines: head + important (not in head/tail) + tail
        head = result_lines[:head_count]
        tail = result_lines[-tail_count:]
        tail_indices = set(range(n - tail_count, n))
        
        # Important lines that aren't in head or tail
        mid_important = []
        for idx, line in enumerate(result_lines):
            if idx in important_indices and idx not in set(range(head_count)) and idx not in tail_indices:
                mid_important.append(line)
        
        all_kept = head + mid_important + tail
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for line in all_kept:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        
        result = "\n".join(deduped) + f"\n[...compressed: {n} lines → kept {len(deduped)} with {len(mid_important)} important lines...]"

    return result

# ─── Terminal compressor ──────────────────────────────────────────────────────

# Regex patterns for terminal output
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07")  # ANSI escape codes
_PROGRESS_BAR_RE = re.compile(r"[\[>=.]\s*\d+%\s*[\]>]|█|▓|░|\r")  # progress bars
_SPINNER_RE = re.compile(r"[|/\\-]{2,}\s*")  # spinner characters repeated
_DOWNLOAD_RE = re.compile(r"\d+\.\d+\s*(MB|KB|GB|bytes|B)/s")  # download speeds
_PERCENT_RE = re.compile(r"\d+%")  # standalone percentages

# Noise lines: build output, compilations, etc.
_NOISE_PATTERNS = [
    re.compile(r"^\s*Compiling\s+\w+", re.IGNORECASE),
    re.compile(r"^\s*Building\s+", re.IGNORECASE),
    re.compile(r"^\s*Downloading\s+", re.IGNORECASE),
    re.compile(r"^\s*Extracting\s+", re.IGNORECASE),
    re.compile(r"^\s*Installing\s+", re.IGNORECASE),
    re.compile(r"^\s*Fetching\s+", re.IGNORECASE),
    re.compile(r"^\s*Resolving\s+", re.IGNORECASE),
    re.compile(r"^\s*Generating\s+", re.IGNORECASE),
    re.compile(r"^\s*Processing\s+", re.IGNORECASE),
    re.compile(r"^\s*Reading\s+", re.IGNORECASE),
    re.compile(r"^\s*Writing\s+", re.IGNORECASE),
    re.compile(r"^\s*Copying\s+", re.IGNORECASE),
    re.compile(r"^\s*Removing\s+", re.IGNORECASE),
    re.compile(r"^\s*Cleaning\s+", re.IGNORECASE),
    re.compile(r"^\s*Loading\s+", re.IGNORECASE),
    re.compile(r"^\s*Scanning\s+", re.IGNORECASE),
    re.compile(r"^\s*Checking\s+", re.IGNORECASE),
    re.compile(r"^\s*Verifying\s+", re.IGNORECASE),
    re.compile(r"^\s*Computing\s+", re.IGNORECASE),
    re.compile(r"^\s*Downloading\s+.*\.\.\.", re.IGNORECASE),
]

# Patterns to ALWAYS keep (errors, warnings, important)
_KEEP_PATTERNS = [
    re.compile(r"error[:\s]", re.IGNORECASE),
    re.compile(r"warning[:\s]", re.IGNORECASE),
    re.compile(r"fatal[:\s]", re.IGNORECASE),
    re.compile(r"panic[:\s]", re.IGNORECASE),
    re.compile(r"failed", re.IGNORECASE),
    re.compile(r"fail\b", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"exit\s+code[:\s]", re.IGNORECASE),
    re.compile(r"returned\s+\d+", re.IGNORECASE),
    re.compile(r"\bPASS\b"),
    re.compile(r"\bFAIL\b"),
    re.compile(r"\bOK\b"),
    re.compile(r"test.*passed", re.IGNORECASE),
    re.compile(r"test.*failed", re.IGNORECASE),
    re.compile(r"all\s+\d+\s+passed", re.IGNORECASE),
    re.compile(r"completed", re.IGNORECASE),
    re.compile(r"done\b", re.IGNORECASE),
    re.compile(r"finished\b", re.IGNORECASE),
]

# File paths with line numbers (keep for debugging)
_PATH_LINE_RE = re.compile(r"(?:[/\\]\w+[\w./\\-]+)\.\w+:\d+")

def _looks_like_terminal(text: str) -> bool:
    """Check if text contains terminal/shell output indicators."""
    # Check for ANSI codes
    if _ANSI_RE.search(text):
        return True
    # Check for progress bars
    if re.search(r"\[\s*=>?\s*\d+%", text):
        return True
    # Check for spinner-like repeated chars
    if re.search(r"[|/\\-]{3,}", text):
        return True
    # Check for download speed patterns
    if _DOWNLOAD_RE.search(text):
        return True
    return False

def _compress_terminal(text: str, ratio: float = 0.50) -> str:
    """Compress terminal/shell output: strip ANSI, remove noise, collapse repeats."""
    if not text or len(text) < 100:
        return text

    original_len = len(text)

    # Step 1: Strip ANSI escape codes
    cleaned = _ANSI_RE.sub("", text)

    # Step 2: Split into lines
    lines = cleaned.split("\n")

    # Step 3: Remove noise lines and progress bars, keep important ones
    important_lines = []
    for line in lines:
        stripped = line.strip()

        # Skip empty lines (but preserve single blank)
        if not stripped:
            if important_lines and important_lines[-1] != "":
                important_lines.append("")
            continue

        # Skip progress bars
        if re.search(r"\[\s*=>?\s*\d+%", stripped):
            continue
        # Skip lines that are JUST a percentage
        if re.match(r"^\s*\d+%\s*$", stripped):
            continue
        # Skip spinner-like lines
        if re.match(r"^[|/\\-]{2,}\s*$", stripped):
            continue
        # Skip download speed lines
        if _DOWNLOAD_RE.search(stripped) and len(stripped) < 60:
            continue
        # Skip common noise patterns
        noise = False
        for pat in _NOISE_PATTERNS:
            if pat.match(stripped):
                noise = True
                break
        if noise:
            continue

        # Remove inline \r (carriage return) artifacts
        stripped = re.sub(r"\r+", "", stripped)

        important_lines.append(stripped)

    # Step 4: Collapse repeated identical lines
    collapsed = []
    i = 0
    while i < len(important_lines):
        current = important_lines[i]
        count = 1
        while i + count < len(important_lines) and important_lines[i + count] == current:
            count += 1
        if count > 1:
            collapsed.append(f"{current} [×{count} repeated]")
            i += count
        else:
            collapsed.append(current)
            i += 1

    # Step 5: Remove consecutive blank lines
    result_lines = []
    prev_blank = False
    for line in collapsed:
        if line == "":
            if not prev_blank:
                result_lines.append(line)
            prev_blank = True
        else:
            prev_blank = False
            result_lines.append(line)

    result = "\n".join(result_lines).strip()

    # Step 6: If still too large, truncate middle (like _compress_text)
    target_len = int(original_len * ratio)
    if len(result) > target_len:
        result = _truncate_to_target(result, target_len)

    return result

def _compress_system_prompt(text: str, ratio: float = 0.50) -> str:
    """System prompt compression preserving section headers."""
    if not text or len(text) < 200:
        return text

    target_len = int(len(text) * ratio)
    sections = re.split(r"(?m)^(#{1,3}\s+.+)$", text)

    if len(sections) <= 1:
        return _compress_text(text, ratio)

    result = []
    for section in sections:
        if re.match(r"^#{1,3}\s+", section):
            result.append(section)
        else:
            result.append(_compress_text(section, ratio))

    return "\n".join(result)

# ─── Tool Schema compressor ──────────────────────────────────────────────────

def _compress_schema_obj(obj):
    """Recursively compress a JSON schema object: keep keys/types/structure,
    truncate verbose description strings. Used by schema and tools compression."""
    if isinstance(obj, dict):
        compressed = {}
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                # Truncate verbose descriptions to first 100 chars
                # Remove code blocks (```...```)
                v = re.sub(r"```[\s\S]*?```", "[code example removed]", v)
                # Remove inline examples
                v = re.sub(r"(?:e\.g\.|example|for example)[^.]*\.", "eg.", v, flags=re.IGNORECASE)
                if len(v) > 100:
                    v = v[:97] + "..."
                compressed[k] = v
            elif isinstance(v, dict):
                compressed[k] = _compress_schema_obj(v)
            elif isinstance(v, list):
                compressed[k] = _compress_schema_obj(v)
            else:
                # Keep scalar values (type, name, enum, format, default, etc.)
                compressed[k] = v
        return compressed
    elif isinstance(obj, list):
        return [_compress_schema_obj(item) for item in obj]
    return obj


def _compress_tools(tools: list) -> tuple[list, int]:
    """Compress an OpenAI tools/functions array: truncate long descriptions,
    compress parameter-schema descriptions. Names/types/enums/required preserved.
    Returns (new_tools, chars_saved)."""
    new_tools = []
    saved = 0
    for tool in tools:
        if not isinstance(tool, dict):
            new_tools.append(tool)
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            # Responses API flat format: {type:"function", name, description, parameters}
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                function = tool
            else:
                new_tools.append(tool)
                continue
        new_function = dict(function)
        desc = function.get("description")
        if isinstance(desc, str) and len(desc) > 120:
            new_function["description"] = desc[:117] + "..."
            saved += len(desc) - len(new_function["description"])
        params = function.get("parameters")
        if isinstance(params, dict):
            before = json.dumps(params, separators=(",", ":"))
            new_function["parameters"] = _compress_schema_obj(params)
            after = json.dumps(new_function["parameters"], separators=(",", ":"))
            if len(after) < len(before):
                saved += len(before) - len(after)
        if function is tool:  # flat Responses format — keep flat
            new_tools.append(new_function)
        else:
            new_tools.append(dict(tool, function=new_function))
    return new_tools, saved


# Structural JSON compression for large JSON tool results
_MAX_JSON_STRING_LEN = 600

def _maybe_json_content(text: str) -> bool:
    """True when text is large enough to matter and parses as a JSON doc (dict/list)."""
    if not text or len(text) < 1000:
        return False
    if text.lstrip()[:1] not in ("{", "["):
        return False
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    return isinstance(obj, (dict, list))


def _compress_json_obj(obj):
    """Recursively shrink long strings in a parsed JSON value; keep keys/types.
    Sorts dict keys → structurally identical tool results re-serialize to the
    same bytes → upstream prompt-caching sees a stable token tail."""
    if isinstance(obj, dict):
        return {k: _compress_json_obj(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_compress_json_obj(v) for v in obj]
    if isinstance(obj, str) and len(obj) > _MAX_JSON_STRING_LEN:
        return obj[:450] + f"\n[...truncated: {len(obj)} chars...]" + obj[-150:]
    return obj


def _compress_json_content(text: str, ratio: float = 0.50) -> str:
    """Structural JSON compression: preserve all keys/numbers/bools, truncate long strings.
    Returns original text when JSON is unparseable or there is no size gain."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if not isinstance(obj, (dict, list)):
        return text
    try:
        result = json.dumps(_compress_json_obj(obj), separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return text
    if len(result) / max(len(text), 1) > ratio:
        return text
    return result


def _compress_tool_schema(text: str, ratio: float = 0.50) -> str:
    """Compress tool schemas: truncate verbose descriptions, keep structure.
    - Truncate verbose tool descriptions to first 100 chars
    - Keep function names and parameter names
    - Keep required parameters marked
    - Remove example code blocks from descriptions
    - Keep type information
    """
    if not text or len(text) < 100:
        return text

    original_len = len(text)
    target_len = int(original_len * ratio)

    # Try to parse as JSON to do structural compression
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Not valid JSON — fall back to text compression
        return _compress_text(text, ratio)

    try:
        compressed = _compress_schema_obj(data)
        result = json.dumps(compressed, separators=(",", ":"))
    except Exception:
        return _compress_text(text, ratio)

    if len(result) > target_len:
        result = _truncate_to_target(result, target_len)
    return result

# ─── Diff compressor ─────────────────────────────────────────────────────────

_DIFF_HUNK_RE = re.compile(r"^@@\s+(-\d+(?:,\d+)?)\s+\+(\d+(?:,\d+)?)\s+@@")
_DIFF_FILE_RE = re.compile(r"^(---|\+\+\+)\s+[\w/.\-]+")
_DIFF_INDEX_RE = re.compile(r"^index\s+")
_DIFF_MODE_RE = re.compile(r"^(?:old|new)\s+mode\s+\d+")

def _compress_diff(text: str, ratio: float = 0.50) -> str:
    """Compress diffs: keep changed lines, collapse unchanged context.
    - Strip diff headers (--- a/, +++ b/, index, mode lines)
    - Keep only changed lines (+ and - lines)
    - Collapse unchanged context lines
    - Keep file paths from headers
    - Keep line numbers
    """
    if not text or len(text) < 100:
        return text

    original_len = len(text)
    target_len = int(original_len * ratio)
    lines = text.split("\n")

    output_files = []
    current_file = ""
    current_hunk_start_new = 0
    current_hunk_lines = 0
    kept_lines = []
    unchanged_count = 0

    for line in lines:
        # Track file paths from diff headers
        file_match = _DIFF_FILE_RE.match(line)
        if file_match:
            path = file_match.group(1).strip()
            # Strip b/ prefix for +++ lines
            path = re.sub(r"^b/", "", path)
            if line.startswith("+++"):
                current_file = path
                kept_lines.append(f"--- {current_file}")
            else:
                current_file = path
                kept_lines.append(f"--- {current_file}")
            continue

        # Skip index and mode lines
        if _DIFF_INDEX_RE.match(line):
            continue
        if _DIFF_MODE_RE.match(line):
            continue

        # Track hunk headers for line numbers
        hunk_match = _DIFF_HUNK_RE.match(line)
        if hunk_match:
            # Flush accumulated unchanged lines
            if unchanged_count > 0:
                kept_lines.append(f"... {unchanged_lines}")
                unchanged_count = 0
                unchanged_lines = 0
            current_hunk_start_new = int(hunk_match.group(2).split(",")[0])
            current_hunk_lines = 0
            continue

        # Changed lines (+ or -)
        if line.startswith("+") or line.startswith("-"):
            # Flush unchanged buffer
            if unchanged_count > 0:
                kept_lines.append(f"... {unchanged_lines} unchanged lines")
                unchanged_count = 0
                unchanged_lines = 0
            current_hunk_lines += 1
            kept_lines.append(line)
            continue

        # Unchanged context line
        unchanged_count += 1
        unchanged_lines = unchanged_count

    # Flush trailing unchanged buffer
    if unchanged_count > 0:
        kept_lines.append(f"... {unchanged_lines} unchanged lines")

    result = "\n".join(kept_lines).strip()

    if len(result) > target_len:
        result = _truncate_to_target(result, target_len)
    return result

# ─── HTML compressor ─────────────────────────────────────────────────────────

_HTML_SCRIPT_RE = re.compile(r"<script[\s\S]*?</script>", re.IGNORECASE)
_HTML_STYLE_RE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
_HTML_TAG_RE2 = re.compile(r"<[^>]+>")
_HTML_ATTR_HREF_RE = re.compile(r'\bhref\s*=\s*"([^"]+)"', re.IGNORECASE)
_HTML_ATTR_SRC_RE = re.compile(r'\bsrc\s*=\s*"([^"]+)"', re.IGNORECASE)
_HTML_ATTR_ALT_RE = re.compile(r'\balt\s*=\s*"([^"]+)"', re.IGNORECASE)
_HTML_WS_RE = re.compile(r"\s+")
_HTML_TABLE_OPEN = re.compile(r"<(?:table|tr|td|th|thead|tbody)\b", re.IGNORECASE)
_HTML_TABLE_CLOSE = re.compile(r"</(?:table|tr|td|th|thead|tbody)\b", re.IGNORECASE)
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_P_RE = re.compile(r"<(?:p|div|h[1-6]|li|blockquote)\b[^>]*>", re.IGNORECASE)

def _compress_html(text: str, ratio: float = 0.50) -> str:
    """Compress HTML content: strip tags, keep text, preserve links/images.
    - Strip HTML tags, keep text content
    - Keep href/src attributes as [link:URL]
    - Keep alt text from images
    - Strip scripts and styles completely
    - Collapse whitespace
    - Keep table structure indicators
    """
    if not text or len(text) < 100:
        return text

    original_len = len(text)
    target_len = int(original_len * ratio)

    result = text

    # Step 1: Strip scripts and styles completely
    result = _HTML_SCRIPT_RE.sub("", result)
    result = _HTML_STYLE_RE.sub("", result)

    # Step 2: Extract href/src/alt before stripping tags
    links = _HTML_ATTR_HREF_RE.findall(result)
    sources = _HTML_ATTR_SRC_RE.findall(result)
    alts = _HTML_ATTR_ALT_RE.findall(result)

    # Step 3: Replace table tags with indicators
    result = _HTML_TABLE_OPEN.sub("\n", result)
    result = _HTML_TABLE_CLOSE.sub("\n", result)
    result = _HTML_BR_RE.sub("\n", result)
    result = _HTML_P_RE.sub("\n", result)

    # Step 4: Strip all remaining HTML tags
    result = _HTML_TAG_RE2.sub("", result)

    # Step 5: Add link/image indicators
    link_parts = []
    for url in links[:20]:  # limit to first 20
        link_parts.append(f"[link:{url}]")
    for src in sources[:10]:  # limit to first 10
        link_parts.append(f"[src:{src}]")
    for alt in alts[:10]:  # limit to first 10
        link_parts.append(f"[alt:{alt}]")

    if link_parts:
        result = result + "\n" + " ".join(link_parts)

    # Step 6: Collapse whitespace
    result = _HTML_WS_RE.sub(" ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    if len(result) > target_len:
        result = _truncate_to_target(result, target_len)
    return result

# ─── Query-aware compression ──────────────────────────────────────────────────

def _compress_aware(text: str, query: str) -> str:
    """Query-aware compression: keep lines/sections matching query terms.
    Falls back to regular compression if no query or text is small."""
    if not query or not text:
        return text
    if len(text) < 300:
        return text

    # Extract query keywords (split on whitespace, lowercase, filter short)
    keywords = [w.lower().strip('",.:;!?') for w in query.split() if len(w) > 2]
    if not keywords:
        return _compress_text(text)

    lines = text.split("\n")

    # Score each line by keyword matches
    scored = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            scored.append((line, 0))
            continue
        lower = stripped.lower()
        score = sum(1 for kw in keywords if kw in lower)
        scored.append((line, score))

    # Keep lines with score > 0, plus first 5 and last 5 lines for context
    head_count = min(5, len(lines))
    tail_count = min(5, len(lines))
    head_indices = set(range(head_count))
    tail_indices = set(range(max(0, len(lines) - tail_count), len(lines)))

    result_lines = []
    kept_relevant = 0
    for i, (line, score) in enumerate(scored):
        if score > 0 or i in head_indices or i in tail_indices:
            result_lines.append(line)
            if score > 0:
                kept_relevant += 1
        elif result_lines and result_lines[-1] != "":
            # Add blank separator when skipping content
            result_lines.append("")

    result = "\n".join(result_lines)
    # Collapse multiple blanks
    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    if kept_relevant == 0:
        # No matches found — fall back to regular compression
        return _compress_text(text, 0.50)

    # Add indicator
    result += f"\n[query-aware: {kept_relevant}/{len([s for s in scored if s[1] > 0])} matching lines kept]"

    # Final size check — if still too large, apply text compression
    if len(result) > len(text) * 0.5:
        result = _compress_text(result, 0.50)

    return result

# ─── Compression orchestrator ──────────────────────────────────────────────────

def _compress_for_role(text: str, role: str, query: str | None = None, msg_age: int = 0) -> str:
    """Choose the right compressor for a given role using content-type detection.
    msg_age = turns from end of conversation (0 = latest); deep-history tool
    results get the aggressive 0.15 ratio."""
    ratio = 0.15 if (AGE_SPLIT_TURNS and role == "tool" and msg_age >= AGE_SPLIT_TURNS) else 0.50
    # If query provided and text is large, use query-aware compression first
    if query and role == "tool" and len(text) > 500:
        text = _compress_aware(text, query)
        # After query-aware, might still be large — do a final pass
        if len(text) > 500:
            content_type = _detect_content_type(text)
            if content_type == "terminal":
                text = _compress_terminal(text)
            elif content_type == "log":
                text = _compress_log(text)
            elif content_type != "json":
                text = _compress_tool_result(text)
        return text

    if role == "system":
        return _compress_system_prompt(text)
    elif role == "tool":
        content_type = _detect_content_type(text)
        if content_type == "terminal":
            return _compress_terminal(text)
        elif content_type == "log":
            return _compress_log(text)
        elif content_type == "json":
            # Structural JSON compression: keep keys/numbers, truncate long strings
            return _compress_json_content(text, ratio)
        elif content_type == "diff":
            return _compress_diff(text)
        elif content_type == "tool_schema":
            return _compress_tool_schema(text)
        elif content_type == "html":
            return _compress_html(text)
        elif content_type == "code":
            return _compress_text(text, ratio=ratio)
        else:
            return _compress_tool_result(text, ratio)
    elif role == "user":
        content_type = _detect_content_type(text)
        if content_type == "log":
            return _compress_log(text)
        if content_type == "html":
            return _compress_html(text)
        return _compress_text(text)
    return _compress_text(text)

def _try_compress_message(msg: dict, skipped: dict | None = None, query: str | None = None,
                          msg_age: int = 0) -> tuple[dict, int]:
    """Try to compress a message. Returns (possibly modified msg, chars saved).
    Stores original content in recovery store before compression.
    `skipped` (optional dict of reason->count) accumulates messages NOT compressed.
    `msg_age` = turns from end of conversation (0 = latest)."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    has_tool_calls = bool(msg.get("tool_calls"))

    # Extract text content from string or list-of-parts format
    text_content = ""
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        # Fix #3: compress each text part individually, not all with same string
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
        text_content = "\n".join(text_parts)

    if not text_content or len(text_content) <= 200:
        if skipped is not None:
            skipped["too_small"] = skipped.get("too_small", 0) + 1
        return msg, 0

    if has_tool_calls:
        if skipped is not None:
            skipped["tool_calls"] = skipped.get("tool_calls", 0) + 1
        return msg, 0

    label = _classify_message(text_content, role, has_tool_calls)
    # Large JSON tool results: structurally compress instead of skipping as verbatim
    if label != "compressible" and role == "tool" and _maybe_json_content(text_content):
        label = "compressible"
    if label != "compressible":
        if skipped is not None:
            skipped["verbatim"] = skipped.get("verbatim", 0) + 1
        return msg, 0

    # Store original in recovery store before lossy compression
    recovery_handle = _recovery_store(text_content)

    # Fix #3: Build new message — compress each text part individually
    new_msg = dict(msg)
    if isinstance(content, str):
        compressed = _compress_for_role(content, role, query, msg_age)
        saved = len(content) - len(compressed)
        if saved <= 0:
            if skipped is not None:
                skipped["no_saving"] = skipped.get("no_saving", 0) + 1
            return msg, 0
        # Append recovery handle as comment
        new_msg["content"] = compressed + f"\n[ccr:{recovery_handle}]"
        saved += len(f"\n[ccr:{recovery_handle}]")
    elif isinstance(content, list):
        total_saved = 0
        new_parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                part_text = p.get("text", "")
                if part_text and len(part_text) > 200:
                    compressed_part = _compress_for_role(part_text, role, query, msg_age)
                    part_saved = len(part_text) - len(compressed_part)
                    if part_saved > 0:
                        total_saved += part_saved
                        new_parts.append(dict(p, text=compressed_part))
                    else:
                        new_parts.append(p)
                else:
                    new_parts.append(p)
            else:
                new_parts.append(p)
        if total_saved <= 0:
            if skipped is not None:
                skipped["no_saving"] = skipped.get("no_saving", 0) + 1
            return msg, 0
        # Append recovery handle to the last text-bearing part
        for part_idx in range(len(new_parts) - 1, -1, -1):
            part = new_parts[part_idx]
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                handle_text = part.get("text", "")
                marker = f"\n[ccr:{recovery_handle}]"
                new_parts[part_idx] = dict(part, text=handle_text + marker)
                total_saved -= len(marker)
                break
        new_msg["content"] = new_parts
        saved = total_saved
    else:
        return msg, 0

    return new_msg, saved


def _compress_responses_items(items: list, skipped: dict | None = None, query: str | None = None,
                              messages_total: int = 0) -> tuple[list, int]:
    """Compress a Responses API `input` item list.

    Reusable shapes are mapped onto chat messages and routed through
    _try_compress_message; everything else (function_call, reasoning,
    item_reference, mcp_*, ...) is forwarded untouched.
    Returns (new_items, chars saved)."""
    new_items = []
    saved_total = 0
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            new_items.append(item)
            continue
        itype = item.get("type")
        if itype in (None, "message"):
            # EasyInputMessage / Message: role + content — same shape as chat
            msg, saved = _try_compress_message(item, skipped, query, len(items) - 1 - idx)
            saved_total += saved
            new_items.append(msg)
        elif itype == "function_call_output":
            # Tool result: {type, call_id, output}. Map onto a synthetic
            # chat tool message so the result compressor + recovery path apply.
            output = item.get("output")
            text = output if isinstance(output, str) else (
                json.dumps(output) if output is not None else "")
            synthetic = {"role": "tool", "content": text}
            compressed_msg, saved = _try_compress_message(synthetic, skipped, query,
                                                          len(items) - 1 - idx)
            if saved > 0:
                saved_total += saved
                item = dict(item, output=compressed_msg["content"])
            new_items.append(item)
        else:
            new_items.append(item)  # function_call, reasoning, etc. — never touch
    return new_items, saved_total


# ─── Anthropic Messages API (POST /v1/messages) ────────────────────────────────

_ANTHROPIC_BLOCK_RE = re.compile(r"tool_result|tool_use|thinking")


def _is_anthropic_payload(payload: dict, path: str) -> bool:
    """Anthropic /v1/messages: messages with tool_result/tool_use/thinking content blocks."""
    if path not in ("v1/messages", "messages"):
        return False
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return False
    for msg in msgs:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            continue
        for p in msg["content"]:
            if isinstance(p, dict) and _ANTHROPIC_BLOCK_RE.search(str(p.get("type", ""))):
                return True
    return False


def _compress_anthropic_message(msg: dict, skipped: dict | None, query: str | None,
                                msg_age: int) -> tuple[dict, int]:
    """Compress one Anthropic /v1/messages message.

    - tool_result blocks (string or content-list) → synthetic tool message
      through the full result-compressor + recovery path.
    - text parts → role-based compression.
    - tool_use / thinking blocks are NEVER touched.
    """
    role = msg.get("role", "user")
    content = msg.get("content")
    new_parts = []
    saved_total = 0

    if isinstance(content, str):
        text = content
        if role == "user":
            # User text: compress if large (conversation history user turns)
            compressed = _compress_for_role(text, role, query, msg_age)
            saved = len(text) - len(compressed)
            if saved > 0:
                handle = _recovery_store(text)
                saved -= len(f"\n[ccr:{handle}]")
                if saved > 0:
                    return dict(msg, content=compressed + f"\n[ccr:{handle}]"), saved
        return msg, 0

    if not isinstance(content, list):
        return msg, 0

    for p in content:
        if not isinstance(p, dict):
            new_parts.append(p)
            continue
        ptype = p.get("type", "")
        if ptype == "tool_result":
            inner = p.get("content")
            if isinstance(inner, str):
                synthetic = {"role": "tool", "content": inner}
                comp_msg, saved = _try_compress_message(synthetic, skipped, query, msg_age)
                if saved > 0:
                    saved_total += saved
                    p = dict(p, content=comp_msg["content"])
            elif isinstance(inner, list):
                compressed_inner = []
                for ip in inner:
                    if isinstance(ip, dict) and ip.get("type") == "text":
                        text, saved = _compress_recoverable_text(
                            ip.get("text", ""), "tool", query, msg_age)
                        if saved > 0:
                            saved_total += saved
                            compressed_inner.append(dict(ip, text=text))
                            continue
                    compressed_inner.append(ip)
                if any(a is not b for a, b in zip(inner, compressed_inner)):
                    p = dict(p, content=compressed_inner)
        elif ptype == "text":
            if role != "assistant":
                text, saved = _compress_recoverable_text(
                    p.get("text", ""), role, query, msg_age)
                if saved > 0:
                    saved_total += saved
                    p = dict(p, text=text)
        # tool_use, thinking, image, document, etc. — never touch
        new_parts.append(p)

    if saved_total <= 0:
        return msg, 0
    if skipped is not None and not any(
        isinstance(p, dict) and p.get("type") == "tool_result" for p in new_parts):
        pass  # keep counters simple; skip reasons already accumulated per block
    return dict(msg, content=new_parts), saved_total


def _compress_anthropic_messages(payload: dict, skipped: dict | None,
                                 query: str | None) -> tuple[int, int, int]:
    """Compress an Anthropic /v1/messages payload in place.
    Returns (total_saved, messages_compressed, messages_total)."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0, 0, 0
    total_saved = 0
    count = 0
    n = len(messages)
    new_messages = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue
        comp_msg, saved = _compress_anthropic_message(msg, skipped, query, n - 1 - idx)
        total_saved += saved
        if saved > 0:
            count += 1
        new_messages.append(comp_msg)
    if isinstance(payload.get("system"), str):
        text, saved = _compress_recoverable_text(payload["system"], "system", query)
        if saved > 0:
            total_saved += saved
            payload["system"] = text
    payload["messages"] = new_messages
    return total_saved, count, n

# ─── Lifespan ──────────────────────────────────────────────────────────────────

_http: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    # Initialize recovery store
    _recovery_cleanup()
    # Fix #7: Log only first 8 chars of session ID
    log.info("Proxy starting: upstream=%s port=%s session=%s compress=%s",
             UPSTREAM_URL, PROXY_PORT, SESSION_ID[:8] + "...", COMPRESS_ENABLED)
    yield
    await _http.aclose()

app = FastAPI(lifespan=lifespan)

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_URL, "session_id": SESSION_ID[:8] + "...", "compress": COMPRESS_ENABLED}


@app.get("/stats")
async def stats():
    """Return compression stats as JSON."""
    entries = list(REQUEST_STATS)
    total_requests = len(entries)
    total_chars_saved = sum(e["chars_saved"] for e in entries)
    avg_ratio = sum(e["compression_ratio"] for e in entries) / total_requests if total_requests else 0
    skipped_totals = {}
    for e in entries:
        for k, v in (e.get("skipped") or {}).items():
            skipped_totals[k] = skipped_totals.get(k, 0) + v
    return {
        "total_requests": total_requests,
        "requests_by_path": dict(Counter(e["path"] for e in entries)),
        "total_chars_saved": total_chars_saved,
        "avg_compression_ratio": round(avg_ratio, 1),
        "recent": entries[-20:],
        "skipped_totals": skipped_totals,
        "tools_saved_total": sum(e.get("tools_saved_chars", 0) for e in entries),
        "recovery_store": _recovery_stats(),
        "tokens_est_saved_total": sum(e.get("tokens_est_saved", 0) for e in entries),
        "upstream_prompt_tokens_total": sum(e.get("upstream_prompt_tokens") or 0 for e in entries),
    }


@app.get("/recovery/{handle:path}")
async def recovery(handle: str):
    """Retrieve original content by recovery handle."""
    _recovery_cleanup()  # lazy cleanup
    original = _recovery_get(handle)
    if original is None:
        return JSONResponse(
            {"error": "not_found", "handle": handle},
            status_code=404,
        )
    return {"handle": handle, "content": original}


@app.get("/dashboard")
async def dashboard():
    """Serve the compression stats dashboard."""
    dash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    try:
        with open(dash_path, "r") as f:
            html = f.read()
        return Response(content=html, media_type="text/html")
    except FileNotFoundError:
        return Response(content="<h1>Dashboard not found</h1>", media_type="text/html", status_code=404)


# Fix #4: Path traversal protection — reject paths containing '..' segments
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """Catch-all: forward to upstream with x-opencode-session injection + compression."""
    path = _canonical_path(path)
    # Fix #4: Reject path traversal attempts
    if ".." in path.split("/"):
        return Response(
            json.dumps({"error": "invalid_path"}),
            status_code=400,
            media_type="application/json",
        )

    url = _upstream_url(path, request.url.query)

    headers = dict(request.headers)
    for h in ("host", "transfer-encoding", "connection", "content-length"):
        headers.pop(h, None)

    headers_lower = {k.lower() for k in headers}
    if "x-opencode-session" not in headers_lower:
        headers["x-opencode-session"] = SESSION_ID
        log.debug("Injected x-opencode-session: %s", SESSION_ID[:8] + "...")

    if UPSTREAM_KEY and "authorization" not in headers_lower:
        headers["authorization"] = f"Bearer {UPSTREAM_KEY}"
        log.debug("Injected Authorization header")

    body = await request.body()
    original_body = body
    req_id = uuid.uuid4().hex

    # ── Compression pass ────────────────────────────────────────────────────
    total_saved = 0
    original_size = len(body)
    compressed_size = len(body)
    messages_compressed = 0
    messages_total = 0
    is_streaming = False
    skipped = {"too_small": 0, "tool_calls": 0, "verbatim": 0, "no_saving": 0}
    tools_compressed = 0
    tools_saved_chars = 0
    if COMPRESS_ENABLED and body and request.method in ("POST", "PUT", "PATCH"):
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                query = _extract_query(payload)
                if _is_anthropic_payload(payload, path):
                    total_saved, messages_compressed, messages_total = \
                        _compress_anthropic_messages(payload, skipped, query)
                else:
                    messages = payload.get("messages")
                    if isinstance(messages, list):
                        messages_total = len(messages)
                        new_messages = []
                        n = len(messages)
                        for idx, msg in enumerate(messages):
                            compressed_msg, saved = _try_compress_message(msg, skipped, query, n - 1 - idx)
                            total_saved += saved
                            if saved > 0:
                                messages_compressed += 1
                            new_messages.append(compressed_msg)
                        if total_saved > 0:
                            payload["messages"] = new_messages

                if path == "v1/responses":
                    # Responses API: input (string or item list) + instructions
                    inp = payload.get("input")
                    if isinstance(inp, list):
                        new_items, saved = _compress_responses_items(inp, skipped, query, messages_total)
                        messages_total = len(inp)
                        total_saved += saved
                        messages_compressed = sum(1 for a, b in zip(inp, new_items) if json.dumps(a) != json.dumps(b))
                        if saved > 0:
                            payload["input"] = new_items
                    elif isinstance(inp, str):
                        compressed, saved = _compress_recoverable_text(inp, "user", query)
                        if saved > 0:
                            payload["input"] = compressed
                            total_saved += saved
                            messages_compressed = 1
                            messages_total = 1
                    instr = payload.get("instructions")
                    if isinstance(instr, str):
                        compressed, saved = _compress_recoverable_text(instr, "system", query)
                        if saved > 0:
                            payload["instructions"] = compressed
                            total_saved += saved

                # Compress the tools/functions arrays (reshipped on every request)
                for key in ("tools", "functions"):
                    items = payload.get(key)
                    if isinstance(items, list) and items:
                        new_items, t_saved = _compress_tools(items)
                        if t_saved > 0:
                            payload[key] = new_items
                            tools_compressed = len(items)
                            tools_saved_chars += t_saved

                if total_saved > 0 or tools_saved_chars > 0:
                    total_saved += tools_saved_chars
                    body = json.dumps(payload).encode("utf-8")
                    compressed_size = len(body)
                    log.info("Compressed %d chars across messages + tools", total_saved)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass  # not JSON — forward as-is

    log.debug("%s %s → %s (%d bytes)", request.method, path, url, len(body))

    # Fix #1: Use async with for safe context manager handling;
    # wrap stream to guarantee cleanup even on client disconnect.
    try:
        upstream_ctx = _http.stream(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        upstream = await upstream_ctx.__aenter__()
    except httpx.ConnectError as e:
        # Fix #6: Don't leak internals to client; log full error server-side
        log.error("Upstream connect error: %s", e)
        return Response(
            json.dumps({"error": "upstream_unreachable"}),
            status_code=502,
            media_type="application/json",
        )
    except Exception as e:
        # Fix #6: Don't leak internals to client; log full error server-side
        log.error("Proxy error: %s %s", type(e).__name__, e)
        return Response(
            json.dumps({"error": "proxy_error"}),
            status_code=502,
            media_type="application/json",
        )

    # Fix #1: Guarantee __aexit__ runs in ALL paths, even if relay() is
    # never consumed (client disconnect). The try/finally around the entire
    # block after __aenter__ ensures the context manager always closes.
    try:
        content_type = upstream.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type

        # For SSE, StreamingResponse runs the generator lazily (after return).
        # We must NOT close the upstream connection in the handler's finally block.
        # Instead, the relay() generator handles cleanup via __aexit__ when done.
        # For non-SSE, clean up in the handler's finally.
        _cleanup_ctx = not is_sse
        is_streaming = is_sse

        # ── Record stats (chat/completions, responses, anthropic messages) ──
        if path in ("v1/chat/completions", "v1/responses", "v1/messages"):
            ratio = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
            REQUEST_STATS.append({
                "timestamp": time.time(),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "path": path,
                "method": request.method,
                "original_size": original_size,
                "compressed_size": compressed_size,
                "chars_saved": original_size - compressed_size,
                "compression_ratio": round(ratio, 1),
                "messages_compressed": messages_compressed,
                "messages_total": messages_total,
                "skipped": skipped,
                "tools_compressed": tools_compressed,
                "tools_saved_chars": tools_saved_chars,
                "streaming": is_streaming,
                "req_id": req_id,
            })
            if total_saved > 0:
                asyncio.create_task(_backfill_token_stats(req_id, original_body, body))

        if is_sse:
            async def relay():
                captured_usage: dict = {}
                captured_cost = None
                pending = b""
                try:
                    async for chunk in upstream.aiter_bytes():
                        if chunk:
                            pending += chunk
                            parts = pending.split(b"\n\n")
                            pending = parts.pop()
                            for part in parts:
                                if b"usage" not in part and b"cost" not in part:
                                    continue
                                for line in part.splitlines():
                                    if not line.startswith(b"data:"):
                                        continue
                                    try:
                                        evt = json.loads(line[5:].strip())
                                    except (json.JSONDecodeError, ValueError):
                                        continue
                                    if not isinstance(evt, dict):
                                        continue
                                    if isinstance(evt.get("usage"), dict):
                                        captured_usage.update(evt["usage"])
                                    for container_key in ("response", "message", "delta"):
                                        container = evt.get(container_key)
                                        if isinstance(container, dict) and isinstance(container.get("usage"), dict):
                                            captured_usage.update(container["usage"])
                                    if "cost" in evt and captured_cost is None:
                                        captured_cost = evt.get("cost")
                        yield chunk
                finally:
                    await upstream.aclose()
                    await upstream_ctx.__aexit__(None, None, None)
                    if path in ("v1/chat/completions", "v1/responses", "v1/messages"):
                        entry = _find_entry(req_id)
                        if entry is not None:
                            if captured_usage:
                                entry["upstream_usage"] = captured_usage
                                entry["upstream_prompt_tokens"] = captured_usage.get(
                                    "prompt_tokens") or captured_usage.get("input_tokens")
                            if captured_cost is not None:
                                entry["upstream_cost"] = captured_cost

            resp_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding")
            }
            resp_headers["cache-control"] = "no-cache"
            resp_headers["x-accel-buffering"] = "no"
            if total_saved > 0:
                resp_headers["x-compressed"] = f"saved {total_saved} chars"
            return StreamingResponse(relay(), status_code=upstream.status_code,
                                     headers=resp_headers, media_type="text/event-stream")
        else:
            try:
                resp_body = await upstream.aread()
                if path in ("v1/chat/completions", "v1/responses", "v1/messages"):
                    entry = _find_entry(req_id)
                    if entry is not None:
                        try:
                            resp_json = json.loads(resp_body)
                        except (json.JSONDecodeError, ValueError):
                            resp_json = None
                        if isinstance(resp_json, dict):
                            usage = resp_json.get("usage")
                            if not isinstance(usage, dict) and isinstance(resp_json.get("response"), dict):
                                usage = resp_json["response"].get("usage")
                            if isinstance(usage, dict):
                                entry["upstream_usage"] = usage
                                entry["upstream_prompt_tokens"] = usage.get(
                                    "prompt_tokens") or usage.get("input_tokens")
                            if "cost" in resp_json:
                                entry["upstream_cost"] = resp_json.get("cost")
            finally:
                await upstream.aclose()

            log.debug("← %d (%d bytes)", upstream.status_code, len(resp_body))

            resp_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in ("transfer-encoding", "content-length", "connection", "content-encoding")
            }
            if total_saved > 0:
                resp_headers["x-compressed"] = f"saved {total_saved} chars"
            return Response(content=resp_body, status_code=upstream.status_code,
                            headers=resp_headers, media_type=content_type or "application/json")
    finally:
        if _cleanup_ctx:
            await upstream_ctx.__aexit__(None, None, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT)
