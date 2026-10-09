"""Atomic chunks: the tail after the stop is in-flight, and set completion waits for S."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("matched20", ROOT / "scripts" / "20_matched_service.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_inflight_is_the_tail_of_the_atomic_chunk():
    order = ["a", "b", "c", "d"]
    want = {"b"}
    ctrl = mod.ExternalChunks(order, batch_first=4, batch_rest=4, stop_fn=lambda obs: want.issubset(obs))
    batch = ctrl.next_batch()
    assert batch == order
    ctrl.observe(batch, [0.0, 1.0, 2.0, 3.0])
    assert ctrl.done
    assert ctrl.inflight == 2  # c and d were already in the chunk when b arrived
    assert set(ctrl.observed) == set(order)


def test_set_completion_does_not_stop_before_s_and_rsc_can():
    # External order a,b,c. S = {c}. Set completion must take the whole prefix through c.
    order = ["a", "b", "c"]
    committed = {"c"}
    sett = mod.ExternalChunks(order, 1, 1, lambda obs: committed.issubset(obs))
    scored = []
    while not sett.done:
        batch = sett.next_batch()
        assert batch
        sett.observe(batch, [float(len(scored))] * len(batch))
        scored.extend(batch)
    assert scored == order
    assert sett.inflight == 0  # one document per chunk, stop lands on c itself

    # Certificate that holds as soon as two documents are scored. Same chunks of 2.
    rsc = mod.ExternalChunks(order, 2, 2, lambda obs: len(obs) >= 2)
    batch = rsc.next_batch()
    rsc.observe(batch, [1.0, 2.0])
    assert rsc.done and rsc.inflight == 0
    assert "c" not in rsc.observed
