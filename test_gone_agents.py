"""Runnable check for the pixel-office gone-agent behavior.

Run with the Hermes venv python and HERMES_HOME unset (matches the serving
process env), so `hermes_constants` imports but we NEVER touch the real
events.jsonl — _read_events is patched with synthetic events and _session_title
is stubbed.

    /usr/local/lib/hermes-agent/venv/bin/python \
        /home/hermes/.hermes/plugins/pixel-office/test_gone_agents.py

Exits 0 and prints OK if every assertion holds; otherwise raises.
"""
import os
import sys
import time

# The plugin lives under plugin dirs not on sys.path by default.
PLUGIN = os.path.expanduser("~/.hermes/plugins/pixel-office")
sys.path.insert(0, PLUGIN)
os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))

import __init__ as po  # noqa: E402

# --- stubs: never touch the live office ----------------------------------
def fake_events(events):
    po._read_events = lambda: events
    po._session_title = lambda sid: ""

def start(i, ts):
    return {"event": "session_start", "session_id": f"s{i}", "platform": "desktop", "ts": ts}

def end(i, ts):
    return {"event": "session_end", "session_id": f"s{i}", "ts": ts}

def ids_of():
    return [a["id"] for a in po.build_state()["agents"]]

now = time.time()

# --- scenario 1: a fresh gone agent persists past the old 20s rule ---------
fake_events([start(0, now - 60), start(1, now - 50), start(2, now - 300), end(2, now - 45)])
agents = po.build_state()["agents"]
stat = {a["id"]: a["status"] for a in agents}
assert stat.get("s2") == "gone", f"expected gone status, got {stat.get('s2')}"
assert "s2" in ids_of()
print("OK 1/4  gone agent persists past the old 20s sweep  (s2:", stat["s2"], ")")

# --- scenario 2: gone agents sort AFTER live agents ------------------------
assert ids_of() == ["s0", "s1", "s2"], f"ordering wrong: {ids_of()}"
print("OK 2/4  gone agent sorts after live agents:", ids_of())

# --- scenario 3: gone agent is not culled by the 30m stale rule -------------
fake_events([start(0, now - 60), start(1, now - 3000), end(1, now - 2400)])
assert "s1" in ids_of(), "gone agent wrongly culled by the stale rule"
print("OK 3/4  gone agent survives the stale (30m) sweep")

# --- scenario 4: overflow drops gone (tail) agents, keeps live -------------
po._MAX_AGENTS = 3
fake_events([start(0, now - 60), start(1, now - 50), start(2, now - 45), end(2, now - 45),
             start(3, now - 40), end(3, now - 40)])
agents = po.build_state()["agents"]
ids = [a["id"] for a in agents]
# 4 agents (s0,s1 live; s2,s3 gone), budget 3 → newest ghost (s3) leaves.
assert len(ids) == 3, f"expected 3 after overflow, got {ids}"
assert "s0" in ids and "s1" in ids, f"live agent dropped on overflow: {ids}"
assert "s2" in ids and "s3" not in ids
print("OK 4/4  overflow drops a gone agent, never a live one:", ids)

print("\nALL PASS — gone agents persist, sort after live, drop on overflow only.")