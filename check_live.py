"""Self-check for the office's live-transcript fields (`agent.live`).

    /usr/local/lib/hermes-agent/venv/bin/python check_live.py

Asserts build_state() attaches a live tail to every agent and that at least
one real session yields transcript steps — the thing the detail panel renders.
"""

import __init__ as po

state = po.build_state()
assert "agents" in state and "recent" in state, "state shape broke"
for a in state["agents"]:
    live = a.get("live")
    assert isinstance(live, dict), f"{a['id']}: missing live tail"
    assert {"think", "say", "steps", "last_ts", "result", "think_ts"} <= set(live), f"{a['id']}: {sorted(live)}"

with_steps = [a for a in state["agents"] if a["live"]["steps"]]
assert with_steps, "no agent produced live transcript steps (state.db path/index?)"
for a in with_steps:
    for s in a["live"]["steps"]:
        assert s["t"] in ("think", "say", "tool"), s
        assert isinstance(s.get("ts"), float), s

# The spotlight's headline fields must be populated for at least one agent.
with_live = [a for a in state["agents"] if a["live"]["think"] or a["live"]["result"]]
assert with_live, "no agent exposed live think/result"
for a in with_live:
    assert isinstance(a.get("last_tool"), dict), a["id"]

print("ok:", len(state["agents"]), "agents,", len(with_steps), "with live steps,",
      len(with_live), "with think/result")
