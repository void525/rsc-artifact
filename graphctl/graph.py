"""Graph definition loading (graph/graph.yaml) and readiness logic."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List

import yaml

from .state import GRAPH_DIR, ROOT

GRAPH_PATH = GRAPH_DIR / "graph.yaml"

FINISHED = ("done", "skipped")
NOT_DISPATCHABLE = ("done", "skipped", "blocked")


def load_graph() -> Dict[str, Any]:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        g = yaml.safe_load(f)
    if not isinstance(g, dict) or not isinstance(g.get("nodes"), list) or not g["nodes"]:
        raise ValueError("graph.yaml must define a non-empty 'nodes' list")
    ids = [n["id"] for n in g["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate node ids in graph.yaml")
    known = set(ids)
    for n in g["nodes"]:
        if "spec" not in n:
            raise ValueError(f"node {n['id']} has no 'spec'")
        for d in n.get("depends_on", []):
            if d not in known:
                raise ValueError(f"{n['id']} depends on unknown node {d}")
    topo_order(g)  # raises on a cycle, at load time rather than mid-dispatch
    return g


def node_by_id(g: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    for n in g["nodes"]:
        if n["id"] == node_id:
            return n
    raise KeyError(f"unknown node {node_id}")


def topo_order(g: Dict[str, Any]) -> List[str]:
    order: List[str] = []
    seen = set()
    by_id = {n["id"]: n for n in g["nodes"]}

    def visit(nid: str, stack=()):
        if nid in seen:
            return
        if nid in stack:
            raise ValueError(f"cycle in graph: {' -> '.join(stack + (nid,))}")
        for d in by_id[nid].get("depends_on", []):
            visit(d, stack + (nid,))
        seen.add(nid)
        order.append(nid)

    for n in g["nodes"]:
        visit(n["id"])
    return order


def ready_nodes(g: Dict[str, Any], state: Dict[str, Any], gpu_busy: bool = False) -> List[Dict[str, Any]]:
    """Nodes whose dependencies are done/skipped and that are not finished,
    ordered by priority (lower number first) then topological order."""
    st = state.get("nodes", {})
    order = topo_order(g)
    rank = {nid: i for i, nid in enumerate(order)}
    out = []
    for nid in order:
        n = node_by_id(g, nid)
        s = st.get(nid, {}).get("status", "pending")
        if s in NOT_DISPATCHABLE:
            continue
        if not all(st.get(d, {}).get("status") in FINISHED for d in n.get("depends_on", [])):
            continue
        if gpu_busy and "gpu" in n.get("resources", []) and s != "in_progress":
            continue
        out.append(n)
    out.sort(key=lambda n: (n.get("priority", 100), rank[n["id"]]))
    return out


def writes_of(n: Dict[str, Any]) -> List[str]:
    return [str(w) for w in (n.get("writes") or [])]


def _prefix(glob: str) -> List[str]:
    """The fixed leading path components of a writes-glob: ``runs/replay/**`` -> ['runs', 'replay']."""
    head = str(glob).replace("\\", "/").split("*")[0]
    return [p for p in head.split("/") if p not in ("", ".")]


def conflicts(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True if two nodes declare overlapping write areas (so they must not run in parallel).

    Comparison is per path component, so ``runs/replay/**`` and ``runs/replay_N200/**`` are *not* a
    conflict while ``runs/replay/**`` and ``runs/**`` are.  A node without a ``writes`` list is assumed
    not to conflict (the graph author must then keep it apart by dependencies)."""
    pa = [_prefix(x) for x in writes_of(a)]
    pb = [_prefix(y) for y in writes_of(b)]
    for x in pa:
        for y in pb:
            if not x or not y:
                continue
            k = min(len(x), len(y))
            if x[:k] == y[:k]:
                return True
    return False


def days_until(date_str: str) -> float:
    """Days from now until an ISO timestamp.  A timestamp without a timezone is read as Beijing time
    (UTC+8), the timezone of the machine this project runs on; 'now' is taken in the same frame, so
    the result is correct whatever the OS locale is set to."""
    d = _dt.datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
    cn = _dt.timezone(_dt.timedelta(hours=8))
    if d.tzinfo is None:
        d = d.replace(tzinfo=cn)
    return (d - _dt.datetime.now(cn)).total_seconds() / 86400.0


def read_spec(n: Dict[str, Any]) -> str:
    p = ROOT / n["spec"]
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"(spec file {n['spec']} missing)"
    except OSError as e:
        return f"(spec file {n['spec']} unreadable: {e})"
