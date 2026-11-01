"""pixel-office plugin — a pixel-art virtual office for Hermes agents.

Every Hermes agent (main sessions AND delegate_task subagents) shows up as
an animated pixel character sitting at a desk in a tiny office, rendered in
your browser at http://127.0.0.1:8113 (port configurable).

Design:

* Hooks are pure observers — they never block, veto, or transform anything.
  Each hook appends one JSON line to ``~/.hermes/pixel-office/events.jsonl``.
  Appends are O(1) and wrapped in try/except, so the agent loop never pays
  more than a few microseconds.

* A daemon HTTP server thread is started lazily on the first event. It
  serves the office page and ``/state``, which folds the event log into a
  current-agents snapshot. Because state is derived from the shared event
  file (not process memory), agents from OTHER Hermes processes (gateway +
  CLI at once, cron sessions) appear in the same office. If the port is
  already bound, another Hermes process is serving — we just keep appending
  events and skip serving.

* The event log is trimmed when it exceeds ~512 KB (keeps the newest half),
  so it never grows unbounded.

Configuration (all optional, config.yaml):

    plugins:
      entries:
        pixel-office:
          port: 8113        # HTTP port for the office page
          enabled: true

Nothing here touches the conversation, the prompt cache, or tool results.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8113
_MAX_LOG_BYTES = 512 * 1024
# An agent with no events for this long is swept from the office.
_STALE_SECONDS = 30 * 60
# Real-time tool-call feed length (newest N tool_start events surfaced in /state).
_RECENT_FEED = 12
# How many recent transcript rows form one agent's live "what is it doing/
# thinking" tail, and how long that tail is cached (state.db is 700MB+; the
# index makes a per-agent query ~25ms, the TTL keeps several open tabs from
# re-querying the same session on every poll).
_LIVE_ROWS = 8
_LIVE_TTL = 1.0

_lock = threading.Lock()
_server_started = False
_port: int = DEFAULT_PORT
# Approval hooks don't carry session_id (only a gateway session_key), so we
# attribute them to the most recent session that fired a tool in this
# process — approvals always happen inside a tool dispatch.
_last_session_id: str = ""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _office_dir() -> Path:
    d = get_hermes_home() / "pixel-office"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _events_path() -> Path:
    return _office_dir() / "events.jsonl"


# ---------------------------------------------------------------------------
# Event publishing (hook side — must be cheap and never raise)
# ---------------------------------------------------------------------------

def _publish(event: Dict[str, Any]) -> None:
    try:
        event.setdefault("ts", time.time())
        event.setdefault("pid", os.getpid())
        line = json.dumps(event, ensure_ascii=False, default=str)
        path = _events_path()
        with _lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            _maybe_trim(path)
        _ensure_server()
    except Exception as exc:  # observers must never break the loop — but say so
        logger.warning("pixel-office: failed to record event (%s: %s)",
                       type(exc).__name__, exc)


def _maybe_trim(path: Path) -> None:
    try:
        if path.stat().st_size <= _MAX_LOG_BYTES:
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = lines[len(lines) // 2:]
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("pixel-office trim failed", exc_info=True)


# ---------------------------------------------------------------------------
# State folding (server side)
# ---------------------------------------------------------------------------

def _read_events() -> List[Dict[str, Any]]:
    path = _events_path()
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        logger.debug("pixel-office read failed", exc_info=True)
    return out


def _agent_key(ev: Dict[str, Any]) -> Optional[str]:
    sid = ev.get("session_id") or ev.get("child_session_id")
    if sid:
        return str(sid)
    # Fall back to pid so events without a session id still get a character.
    pid = ev.get("pid")
    return f"pid-{pid}" if pid else None


# Model label for a session, read from state.db (best-effort, cached).
# sessions.model holds the *routed* alias (e.g. "auto/best-reasoning") which
# may not match the model actually answering; session_model_usage holds the
# real per-call models. We show both when they differ.
_model_cache: Dict[str, Any] = {}
_MODEL_TTL = 10.0


def _model_for(sid: str) -> str:
    """Best-effort short model label, TTL-cached so a mid-session model
    switch lands within a few polls without re-querying every fold."""
    now = time.time()
    hit = _model_cache.get(sid)
    if hit and now - hit[0] < _MODEL_TTL:
        return hit[1]
    label = ""
    try:
        con = sqlite3.connect(
            f"file:{get_hermes_home()}/state.db?mode=ro", uri=True, timeout=2
        )
        try:
            row = con.execute(
                "SELECT model FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            routed = (row[0] if row else "") or ""
            if routed:
                label = routed
            # If the session is routed through an alias, the models actually
            # serving the calls are in session_model_usage; show the one that
            # did the real work. api_call_count is misleading here (one huge
            # call counts the same as one tiny one), so rank by tokens.
            if "auto/" in routed:
                used = [
                    r[0]
                    for r in con.execute(
                        "SELECT model FROM session_model_usage "
                        "WHERE session_id = ? AND model NOT LIKE 'auto/%' "
                        "ORDER BY (input_tokens + output_tokens) DESC LIMIT 1",
                        (sid,),
                    )
                ]
                if used:
                    label = f"{routed}→{used[0]}"
        finally:
            con.close()
    except Exception:
        label = ""
    if label:
        _model_cache[sid] = (now, label)
    return label


def _short(text: Any, n: int = 60) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s[: n - 1] + "…" if len(s) > n else s


# Human-meaningful identifier for a session, read from the canonical
# state.db (best-effort, cached in-process). The office folds state every
# ~1.5 s, but a session's title/platform rarely change, so cache keyed by id.
_meta_cache: Dict[str, Any] = {}


def _session_meta(sid: str) -> Optional[Dict[str, Any]]:
    """Best-effort {source, title, title_source} for a session, cached. Misses
    re-query next poll so a title deriving right after first-seen is caught."""
    if sid in _meta_cache:
        return _meta_cache[sid]
    meta: Optional[Dict[str, Any]] = None
    try:
        con = sqlite3.connect(
            f"file:{get_hermes_home()}/state.db?mode=ro", uri=True, timeout=2
        )
        try:
            row = con.execute(
                "SELECT source, title, title_source FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            if row:
                meta = {
                    "source": row[0] or "",
                    "title": row[1] or "",
                    "title_source": row[2] or "",
                }
        finally:
            con.close()
    except Exception:
        meta = None
    if meta:  # only cache real rows; misses re-query next poll
        _meta_cache[sid] = meta
    return meta


def _main_label(ev: Dict[str, Any], key: str) -> str:
    """Human label for a main agent: the session's own title, falling back to
    platform+id. Subagents keep their goal label (see _short(child_goal))."""
    meta = _session_meta(key) or {}
    title = (meta.get("title") or "").strip()
    if title:
        return title
    plat = ev.get("platform") or meta.get("source") or ""
    prefix = (plat or "hermes").lower()
    return f"{prefix} {key[-6:]}"


def _main_topic(key: str) -> str:
    """LLM-derived session topic ('what am I working on'), '' until derived."""
    meta = _session_meta(key) or {}
    if meta.get("title_source") == "llm" and meta.get("title"):
        return _short(meta["title"], 30)
    return ""


_live_cache: Dict[str, Any] = {}


def _live_for(sid: str) -> Dict[str, Any]:
    """Live transcript tail for one session: what it is thinking, saying, and
    the last few tool steps — read straight from state.db.

    state.db is written per message (reasoning included) as the turn runs, so
    this IS the real-time signal: no plugin stream hook, no reasoning-delta
    opt-in, and it works for sessions in every other Hermes process, subagents
    included. Best-effort: any failure yields an empty tail, never an error.
    """
    now = time.time()
    hit = _live_cache.get(sid)
    if hit and now - hit[0] < _LIVE_TTL:
        return hit[1]
    out: Dict[str, Any] = {"think": "", "say": "", "steps": [], "last_ts": 0.0}
    steps: List[Dict[str, Any]] = []
    try:
        con = sqlite3.connect(
            f"file:{get_hermes_home()}/state.db?mode=ro", uri=True, timeout=2
        )
        try:
            rows = con.execute(
                "SELECT role, tool_name, substr(content,1,700), "
                "substr(coalesce(nullif(reasoning_content,''), reasoning, ''),1,900), "
                "timestamp FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (sid, _LIVE_ROWS),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return out
    for role, tool, content, reasoning, ts in rows:  # newest first
        ts = float(ts or 0)
        out["last_ts"] = max(out["last_ts"], ts)
        think = _short(reasoning, 700)
        if think and not out["think"]:
            out["think"] = think
            steps.append({"t": "think", "p": _short(reasoning, 260), "ts": ts})
        body = str(content or "").strip()
        if role == "tool":
            steps.append({
                "t": "tool", "n": str(tool or "tool"),
                "p": _short(body.splitlines()[0] if body else "", 150), "ts": ts,
            })
        elif role == "assistant" and body:
            if not out["say"]:
                out["say"] = _short(body, 700)
            steps.append({"t": "say", "p": _short(body, 260), "ts": ts})
        elif role == "user" and body:
            steps.append({"t": "user", "p": _short(body, 150), "ts": ts})
    out["steps"] = steps[:6]
    _live_cache[sid] = (now, out)
    if len(_live_cache) > 200:  # ponytail: blunt cap; a long-lived server only
        _live_cache.clear()     # ever needs the handful of live sessions.
    return out


def build_state() -> Dict[str, Any]:
    """Fold the event log into {agents: [...], recent: [...]} for the frontend."""
    agents: Dict[str, Dict[str, Any]] = {}
    # Live tool-call feed: newest-first, newest appended as events are walked.
    recent: List[Dict[str, Any]] = []
    # key → the tick entry for that agent's in-flight call, so tool_end can
    # stamp the outcome (ok/error + duration) onto the same tick.
    inflight: Dict[str, Dict[str, Any]] = {}
    now = time.time()

    def ensure(key: str, ev: Dict[str, Any]) -> Dict[str, Any]:
        a = agents.get(key)
        if a is None:
            a = {
                "id": key,
                "label": f"agent {key[-6:]}",
                "kind": "main",
                "status": "idle",
                "tool": "",
                "activity": "",
                "detail": "",
                "platform": ev.get("platform") or "",
                "model": "",
                "topic": "",
                "live": {},
                "first_seen": ev.get("ts", now),
                "updated_at": ev.get("ts", now),
            }
            agents[key] = a
        a["updated_at"] = ev.get("ts", a["updated_at"])
        if not a.get("model"):
            a["model"] = _model_for(key)
        return a

    for ev in _read_events():
        kind = ev.get("event")
        key = _agent_key(ev)
        if not key:
            continue

        if kind == "session_start":
            a = ensure(key, ev)
            a["status"] = "idle"
            plat = ev.get("platform") or ""
            a["label"] = _main_label(ev, key)
            a["detail"] = "session started"
        elif kind == "tool_start":
            # Single tool_start branch: re-derive async labels AND set live
            # status/tool from the call, then feed the real-time tick.
            a = ensure(key, ev)
            if a.get("kind") == "main":
                a["label"] = _main_label(ev, key)
                a["topic"] = _main_topic(key)
            tool = str(ev.get("tool_name") or "")
            a["status"] = "working"
            a["tool"] = tool
            a["activity"] = str(ev.get("activity") or "working")
            a["detail"] = _short(ev.get("preview"), 180)
            entry = {
                "id": key,
                "label": a.get("label") or _main_label(ev, key),
                "tool": tool,
                "activity": a["activity"],
                "preview": a["detail"],
                "ts": ev.get("ts", now),
            }
            recent.append(entry)
            inflight[key] = entry
        elif kind == "session_end":
            if key in agents:
                agents[key]["status"] = "gone"
                agents[key]["updated_at"] = ev.get("ts", now)
        elif kind == "subagent_start":
            child = ev.get("child_session_id")
            if child:
                ck = str(child)
                ev2 = dict(ev)
                ev2["session_id"] = ck
                a = ensure(ck, ev2)
                a["kind"] = "subagent"
                a["label"] = _short(ev.get("child_goal"), 26) or f"sub {ck[-6:]}"
                a["status"] = "working"
                a["detail"] = _short(ev.get("child_goal"))
                a["parent"] = str(ev.get("parent_session_id") or "")
        elif kind == "subagent_stop":
            child = ev.get("child_session_id")
            if child and str(child) in agents:
                agents[str(child)]["status"] = "done"
                agents[str(child)]["updated_at"] = ev.get("ts", now)
        elif kind == "tool_end":
            a = ensure(key, ev)
            a["status"] = "thinking"
            a["tool"] = ""
            ok = ev.get("status") != "error"
            e = inflight.pop(key, None)
            if e is not None:  # stamp the outcome onto its tick
                e["status"] = "ok" if ok else "error"
                e["dur"] = ev.get("duration_ms")
            if not ok:
                a["detail"] = f"⚠ {_short(ev.get('error_message'), 40)}"
            else:
                a["detail"] = ""
        elif kind == "approval_request":
            a = ensure(key, ev)
            a["status"] = "waiting"
            a["tool"] = ""
            a["detail"] = _short(ev.get("command"), 40) or "needs approval"
        elif kind == "approval_response":
            a = ensure(key, ev)
            choice = str(ev.get("choice") or "")
            if choice in ("deny", "timeout"):
                a["status"] = "thinking"
                a["detail"] = f"approval: {choice}"
            else:
                a["status"] = "working"
                a["detail"] = ""

    # Sweep stale + long-gone agents.
    visible = []
    for a in agents.values():
        age = now - float(a.get("updated_at") or 0)
        if a["status"] == "gone" and age > 20:
            continue
        if a["status"] == "done" and age > 120:
            continue
        if age > _STALE_SECONDS:
            continue
        # Agents quiet for a bit are "idle", not eternally "thinking".
        if a["status"] in ("working", "thinking") and age > 300:
            a["status"] = "idle"
        # ponytail: topic re-derive on every fold; O(agents) DB lookups per
        # ~1.5s poll, fine at dashboard scale — per-agent cache absorbs it.
        if a.get("kind") == "main":
            a["model"] = _model_for(a["id"]) or a.get("model") or ""
            a["topic"] = _main_topic(a["id"])
        a["live"] = _live_for(a["id"])
        visible.append(a)

    visible.sort(key=lambda a: (a["kind"] != "main", a.get("first_seen", 0)))
    # Newest-first, capped; trim previews late so the tick doesn't carry
    # bytes from a brief live preview that was later truncated.
    if recent:
        recent.reverse()
        recent = recent[:_RECENT_FEED]
    return {"agents": visible, "recent": recent, "ts": now}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def _resolve_port() -> int:
    try:
        from hermes_cli.config import cfg_get, load_config

        p = cfg_get(load_config(), "plugins", "entries", "pixel-office", "port")
        if p:
            return int(p)
    except Exception:
        pass
    return DEFAULT_PORT


def _ensure_server() -> None:
    global _server_started
    if _server_started:
        return
    with _lock:
        if _server_started:
            return
        _server_started = True
    t = threading.Thread(target=_serve, name="pixel-office-http", daemon=True)
    t.start()


def _serve() -> None:
    global _port
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _port = _resolve_port()
    html_path = Path(__file__).resolve().parent / "web" / "index.html"
    web_dir = html_path.parent

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence stdout
            pass

        def do_GET(self) -> None:
            try:
                if self.path.split("?")[0] in ("/", "/index.html"):
                    body = html_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    # The page is JS-rendered, so an intermediate cache holding
                    # a stale copy looks like "the office never updates"
                    # after a plugin upgrade. Serve always-fresh.
                    self.send_header("Cache-Control", "no-store")
                elif self.path.split("?")[0] == "/manifest.json":
                    body = (web_dir / "manifest.json").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/manifest+json")
                    self.send_header("Cache-Control", "no-store")
                elif self.path.split("?")[0] == "/sw.js":
                    body = (web_dir / "sw.js").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/javascript")
                    self.send_header("Cache-Control", "no-store")
                elif self.path.split("?")[0] in ("/icon-512.png", "/icon-maskable-512.png"):
                    body = (web_dir / self.path.lstrip("/")).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "max-age=604800")
                elif self.path.split("?")[0] == "/state":
                    body = json.dumps(build_state()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                else:
                    self.send_response(404)
                    body = b"not found"
                    self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                logger.debug("pixel-office request failed", exc_info=True)

    try:
        srv = ThreadingHTTPServer(("127.0.0.1", _port), Handler)
    except OSError as exc:
        # Port already bound. Probe it: a healthy pixel-office answers /state
        # with JSON containing "agents". Anything else is a foreign squatter
        # (another app, or a ghost VS Code port-forward) — say so LOUDLY,
        # because to the user it looks like "office unreachable".
        verdict = _probe_port(_port)
        if verdict == "office":
            logger.info(
                "pixel-office: port %s already serving a healthy office "
                "(another Hermes process) — this process will feed events only",
                _port,
            )
        else:
            logger.warning(
                "pixel-office: could NOT bind 127.0.0.1:%s (%s) and the "
                "current listener does not answer like a pixel-office "
                "(probe: %s). Another app or a stale VS Code port-forward is "
                "squatting the port. Fix: free the port, or set "
                "plugins.entries.pixel-office.port in config.yaml and update "
                "the extension's hermesPixelOffice.stateUrl to match.",
                _port, exc, verdict,
            )
        return
    logger.info("pixel-office serving at http://127.0.0.1:%s", _port)
    try:
        srv.serve_forever()
    except Exception:
        logger.debug("pixel-office server exited", exc_info=True)


def _probe_port(port: int) -> str:
    """Classify whatever is listening on *port*: 'office', 'foreign', or 'dead'."""
    try:
        import urllib.request

        req = urllib.request.Request(f"http://127.0.0.1:{port}/state")
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = resp.read(4096).decode("utf-8", errors="replace")
        return "office" if '"agents"' in body else "foreign"
    except Exception as exc:
        return f"dead/{type(exc).__name__}"


# ---------------------------------------------------------------------------
# Hook callbacks — all **kwargs so core payload changes never break us
# ---------------------------------------------------------------------------

# Tool name → activity shown in the office (drives the character animation).
_ACTIVITY = {
    "write_file": "typing", "patch": "typing", "skill_manage": "typing",
    "read_file": "reading", "search_files": "reading", "skill_view": "reading",
    "web_search": "browsing", "web_extract": "browsing",
    "browser_navigate": "browsing", "browser_click": "browsing",
    "browser_snapshot": "browsing", "browser_vision": "browsing",
    "terminal": "running", "execute_code": "running", "process": "running",
    "delegate_task": "delegating",
}


def _activity_for(tool: str) -> str:
    return _ACTIVITY.get(str(tool or ""), "working")


def _on_session_start(**kw: Any) -> None:
    _publish({
        "event": "session_start",
        "session_id": kw.get("session_id"),
        "platform": kw.get("platform"),
    })


def _on_session_end(**kw: Any) -> None:
    _publish({"event": "session_end", "session_id": kw.get("session_id")})


def _pre_tool_call(**kw: Any) -> None:
    global _last_session_id
    sid = kw.get("session_id") or ""
    if sid:
        _last_session_id = str(sid)
    args = kw.get("args") or {}
    preview = ""
    if isinstance(args, dict):
        for k in ("command", "path", "query", "url", "goal", "pattern", "prompt"):
            if args.get(k):
                preview = str(args[k])
                break
    tool = kw.get("tool_name")
    _publish({
        "event": "tool_start",
        "session_id": sid,
        "tool_name": tool,
        "activity": _activity_for(tool),
        # 200 chars: the office panel shows the real command/path, and the
        # canvas still wraps to a few short lines. Log trim keeps the file small.
        "preview": _short(preview, 200),
    })
    return None  # observer — never blocks


def _post_tool_call(**kw: Any) -> None:
    _publish({
        "event": "tool_end",
        "session_id": kw.get("session_id"),
        "tool_name": kw.get("tool_name"),
        "status": kw.get("status") or "ok",
        "error_message": kw.get("error_message"),
        "duration_ms": kw.get("duration_ms"),
    })


def _subagent_start(**kw: Any) -> None:
    _publish({
        "event": "subagent_start",
        "parent_session_id": kw.get("parent_session_id"),
        "child_session_id": kw.get("child_session_id"),
        "child_role": kw.get("child_role"),
        "child_goal": kw.get("child_goal"),
    })


def _subagent_stop(**kw: Any) -> None:
    _publish({
        "event": "subagent_stop",
        "child_session_id": kw.get("child_session_id"),
    })


def _pre_approval_request(**kw: Any) -> None:
    _publish({
        "event": "approval_request",
        "session_id": _last_session_id,
        "command": kw.get("command") or kw.get("description"),
        "surface": kw.get("surface"),
    })


def _post_approval_response(**kw: Any) -> None:
    _publish({
        "event": "approval_response",
        "session_id": _last_session_id,
        "choice": kw.get("choice"),
    })


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("post_tool_call", _post_tool_call)
    ctx.register_hook("subagent_start", _subagent_start)
    ctx.register_hook("subagent_stop", _subagent_stop)
    ctx.register_hook("pre_approval_request", _pre_approval_request)
    ctx.register_hook("post_approval_response", _post_approval_response)
    logger.info(
        "pixel-office registered — office at http://127.0.0.1:%s once events flow",
        _resolve_port(),
    )
