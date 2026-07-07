"""C-integrate — offline shadow of the post-call escalation controller (§5.5).

Replays the historical corpus turn-by-turn and answers the Phase-2 gate question
(§10) *without touching live traffic*: **if these turns had been routed
local-with-cascade, what fraction would the trip-wires have escalated?**

The three just-built units are wired together here exactly as the live path (§5.4
post-call flow) would wire them:

  * ``router.tripwires.TripwireState`` (C1) — driven per turn with the real
    ``(assistant response, tool_results)`` stream lifted from the corpus JSONL.
    A fresh state per turn (trip-wires are per-turn, §5.5); ``.fired()`` polled
    after each action; the first wire to cross threshold latches.
  * ``router.state.DictSessionStore`` (C3) — the per-session memory. Privacy pin
    (one-way), sticky route, per-turn strike reset, ``escalated`` marking, and
    turn counting all flow through the store, so the shadow exercises the exact
    ABC the Redis swap will sit behind.
  * ``router.escalate.rebuild_for_escalation`` (C2) — for every turn that would
    have escalated we rebuild that turn's transcript for the higher rung and
    assert the result is well-formed (thinking stripped, tool IDs paired, no
    empty messages). This proves the handover artifact the controller produces
    is valid on real transcripts, not just synthetic ones.

Denominator = **cascaded turns** (§11: "escalation rate *of cascaded turns*").
Cascade eligibility is decided by the real policy engine (``router.policy``) fed
by the real feature extractor (``router.features``), so the rate is measured over
exactly the turns the router would actually arm trip-wires on.

Privacy pin (§5.3/§5.5): a pinned session's escalation target is the **USER**, not
a cloud rung. Those "would-have-escalated" events are counted **separately** as
pin-blocked — they never imply data leaving the machine.

Honest scope of this shadow (stated so the number is not over-read):
  * The corpus is 100 % Claude Code *frontier* traffic. We are asking what the
    trip-wires would have caught had these exact trajectories been produced by
    the local rung. The frontier model rarely fails mechanically, so the
    mechanical wires (edit-apply / loop / no-progress / parse) fire rarely — this
    rate is a **lower bound** on the local escalation rate, not a prediction of
    it. A local model would fail *more*.
  * ``parse_schema`` is effectively untestable here: S5 (malformed local tool
    calls) is FALSIFIED for this corpus (``miner.scenarios``) — no local-model
    traffic exists to trip it. Reported as 0, by construction.
  * ``turn_budget`` is a *cost* guard sized to "2x median tokens for this intent
    class" of the **local** model. Applying it to replayed **frontier** token
    counts measures frontier verbosity, not local runaway, so it is EXCLUDED
    from the headline (budget = None -> wire inert) and reported only as a labeled
    sensitivity.

Usage: PYTHONPATH=src .venv/bin/python -m router.shadow_postcall
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from miner.parser import (
    ParseStats,
    SourceFile,
    content_blocks,
    iter_records,
    iter_source_files,
    text_of,
    tool_result_blocks,
)
from miner.turns import EDIT_FAIL_MARKER, INTERRUPT_PREFIX

from .escalate import THINKING_TYPES, rebuild_for_escalation
from .features import extract
from .policy import CONTEXT_HEADROOM, REGISTRY, route_turn
from .state import DictSessionStore
from .tripwires import TripwireHit, TripwireState, TurnBudget, routes_to_registry

# §5.5 budget multiplier ("2x median tokens for this intent class"). Only used for
# the labeled cost-wire *sensitivity*, never the headline (see module docstring).
BUDGET_TOKEN_MULTIPLIER = 2


# ── turn assembly ────────────────────────────────────────────────────────────
@dataclass
class ReplayTurn:
    """One assembled turn: routing metadata + the ordered action stream that
    drives the trip-wires. ``events`` is a list of tagged tuples in issue order:
      ("assistant", content_blocks, output_tokens)  -> observe_response
      ("user",      content_blocks)                  -> observe_tool_results
    The initiating instruction is NOT an event (it only sets features); interrupt
    attribution is done by look-ahead in the replay, not by feeding it here.
    """

    session_id: str
    source_kind: str
    source_path: str
    ts: str
    instruction_text: str = ""
    interrupted: bool = False
    events: list[tuple] = field(default_factory=list)
    # context-estimate substrate (deduped per API message.id, matching miner)
    _ctx_tokens: int = 0
    _n_asst_msgs: int = 0
    _seen_msg_ids: set = field(default_factory=set)
    output_tokens_total: int = 0
    n_error_results: int = 0
    n_edit_failures: int = 0

    @property
    def context_estimate(self) -> int:
        return int(self._ctx_tokens / max(self._n_asst_msgs, 1))

    @property
    def n_actions(self) -> int:
        return len(self.events)


def _usage_int(usage: Any, key: str) -> int:
    if isinstance(usage, dict) and isinstance(usage.get(key), (int, float)):
        return int(usage[key])
    return 0


def _walk_file(source: SourceFile, stats: ParseStats) -> Iterator[ReplayTurn]:
    """Assemble one transcript file into turns (file-order grouping — the same
    fallback ``miner.turns`` uses when parentUuid chains break). A new turn opens
    on an initiating user record (a user message with no tool_result blocks);
    assistant records and tool_result-bearing user records attach to it."""
    current: Optional[ReplayTurn] = None

    for rec in iter_records(source, stats):
        rtype = rec.get("type")

        if rtype == "user":
            message = rec.get("message")
            results = tool_result_blocks(message)
            blocks = content_blocks(message)

            if results and current is not None:
                # continuation: tool_results feed the no-progress / edit / parse
                # / interrupt wires.
                current.events.append(("user", blocks))
                for blk in results:
                    blob = str(blk.get("content", ""))
                    if blk.get("is_error"):
                        current.n_error_results += 1
                    if blk.get("is_error") and EDIT_FAIL_MARKER in blob:
                        current.n_edit_failures += 1
                continue

            # new initiating turn
            if current is not None:
                yield current
            text = text_of(message)
            current = ReplayTurn(
                session_id=rec.get("sessionId") or source.session_id,
                source_kind=source.kind,
                source_path=str(source.path.name),
                ts=rec.get("timestamp") or "",
                instruction_text=text[:2000],
                interrupted=text.startswith(INTERRUPT_PREFIX),
            )

        elif rtype == "assistant":
            if current is None:
                continue
            message = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            blocks = content_blocks(message)
            msg_id = message.get("id")
            key = msg_id if isinstance(msg_id, str) else rec.get("uuid", "")
            out_tokens = 0
            if key not in current._seen_msg_ids:
                current._seen_msg_ids.add(key)
                current._n_asst_msgs += 1
                usage = message.get("usage")
                out_tokens = _usage_int(usage, "output_tokens")
                current._ctx_tokens += _usage_int(usage, "input_tokens") + _usage_int(
                    usage, "cache_read_input_tokens"
                )
                current.output_tokens_total += out_tokens
            current.events.append(("assistant", blocks, out_tokens))

    if current is not None:
        yield current


def walk_turns(corpus_root: Path) -> list[ReplayTurn]:
    stats = ParseStats()
    turns: list[ReplayTurn] = []
    for source in iter_source_files(corpus_root):
        turns.extend(_walk_file(source, stats))
    return turns


# ── privacy pins ─────────────────────────────────────────────────────────────
def load_pins(secrets_path: Path) -> dict[str, int]:
    """session_id -> 1-based first_hit_turn_index (from the A8 secret scan)."""
    pins: dict[str, int] = {}
    if not secrets_path.exists():
        return pins
    data = json.loads(secrets_path.read_text())
    for item in data.get("would_have_pinned", []):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        sid, meta = item
        if not isinstance(sid, str):
            continue
        idx = meta.get("first_hit_turn_index", 1) if isinstance(meta, dict) else 1
        pins[sid] = int(idx) if isinstance(idx, (int, float)) else 1
    return pins


# ── escalation transcript rebuild (C2 wiring) ────────────────────────────────
def build_messages(turn: ReplayTurn) -> list[dict]:
    """Reconstruct an Anthropic messages list from the turn's action stream,
    prepending the initiating instruction as the opening user message so the
    transcript is a real conversation for ``rebuild_for_escalation``."""
    messages: list[dict] = []
    if turn.instruction_text:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": turn.instruction_text}]}
        )
    for ev in turn.events:
        if ev[0] == "assistant":
            messages.append({"role": "assistant", "content": ev[1]})
        else:
            messages.append({"role": "user", "content": ev[1]})
    return messages


def failure_note(hit: TripwireHit) -> list[str]:
    """<=3-line note the higher rung reads (§5.5 step 3)."""
    target = "capability registry" if routes_to_registry(hit.type) else "difficulty model"
    lines = [
        f"A previous local attempt tripped the '{hit.name}' trip-wire ({hit.type.value}).",
        f"Signal routes to the {target}; do not repeat the local model's dead end.",
    ]
    detail = hit.detail
    if hit.name == "edit_apply":
        lines.append(f"The edit failed to apply {detail.get('strikes', '?')}x on exact-string match.")
    elif hit.name == "loop":
        lines.append(f"The same call repeated {detail.get('repeats', '?')}x with no new result.")
    elif hit.name == "no_progress":
        lines.append(f"{detail.get('actions', '?')} actions advanced nothing (no new read/diff/output).")
    return lines


def validate_rebuilt(messages: list[dict]) -> tuple[bool, str]:
    """Well-formedness of a rebuilt escalation transcript (mirrors the C2 tests):
    non-empty message list; every message has a role and a NON-empty content list;
    no thinking/redacted_thinking blocks survive; every tool_result.tool_use_id is
    paired with a tool_use.id in the same request (B5 internal-pairing rule)."""
    if not isinstance(messages, list) or not messages:
        return False, "empty message list"
    use_ids: set[str] = set()
    result_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            return False, "bad role"
        content = msg.get("content")
        if not isinstance(content, list) or not content:
            return False, "empty/invalid content list"
        for blk in content:
            if not isinstance(blk, dict):
                return False, "non-dict block"
            btype = blk.get("type")
            if btype in THINKING_TYPES:
                return False, f"residual thinking block ({btype})"
            if btype == "tool_use" and isinstance(blk.get("id"), str):
                use_ids.add(blk["id"])
            if btype == "tool_result" and isinstance(blk.get("tool_use_id"), str):
                result_ids.add(blk["tool_use_id"])
    if not result_ids <= use_ids:
        return False, f"orphaned tool_result ids: {sorted(result_ids - use_ids)[:3]}"
    return True, "ok"


# ── the shadow run ───────────────────────────────────────────────────────────
@dataclass
class ShadowResult:
    total_turns: int = 0
    cascaded_turns: int = 0
    fired_cascaded: int = 0
    pin_blocked: int = 0            # would-escalate in a pinned session (target = USER)
    cloud_escalations: int = 0      # would-escalate to a higher rung
    wire_names: Counter = field(default_factory=Counter)
    wire_types: Counter = field(default_factory=Counter)
    train_registry: int = 0         # dialect escalations (§5.7)
    train_router: int = 0           # difficulty/cost/quality escalations
    rebuild_ok: int = 0
    rebuild_fail: list[str] = field(default_factory=list)
    cascade_layers: Counter = field(default_factory=Counter)
    noncascade_layers: Counter = field(default_factory=Counter)
    pinned_sessions_seen: set = field(default_factory=set)
    pinned_sessions_expected: set = field(default_factory=set)
    latencies_us: list[float] = field(default_factory=list)
    # budget sensitivity (labeled, non-headline)
    budget_median: float = 0.0
    budget_added: int = 0
    strike_shape_ok: bool = True


def run_shadow(corpus_root: Path, pins: dict[str, int]) -> ShadowResult:
    res = ShadowResult()
    res.pinned_sessions_expected = set(pins)

    turns = walk_turns(corpus_root)
    res.total_turns = len(turns)

    # Group by TRAJECTORY (session_id, source_path), not merged session_id: the
    # parser stamps every subagent transcript with its PARENT session dir name,
    # so keying on session_id alone merges independent subagent trajectories and
    # accumulates trip-wire strikes across unrelated work (review finding). The
    # store still keys per-trajectory so strikes/state don't bleed; the privacy
    # pin is looked up by the session_id component (pins are session-scoped).
    by_traj: dict[tuple[str, str], list[ReplayTurn]] = defaultdict(list)
    for t in turns:
        by_traj[(t.session_id, t.source_path)].append(t)

    store = DictSessionStore()
    cascaded_output_tokens: list[tuple[int, bool]] = []  # (tokens, fired_mechanical)

    for (sid, spath), sturns in by_traj.items():
        traj_key = f"{sid}::{spath}"     # per-trajectory state key
        sturns.sort(key=lambda t: t.ts)
        for idx, turn in enumerate(sturns):
            # (a) privacy pin — one-way, applied at/after the scan's first-hit turn.
            #     Pins are session-scoped, so look up by sid but pin this trajectory.
            if sid in pins and (idx + 1) >= pins[sid]:
                store.pin_privacy(traj_key)
                res.pinned_sessions_seen.add(sid)

            store.touch(traj_key)
            state = store.get_session(traj_key)
            # Hysteresis is a live-stickiness optimisation; for a well-defined
            # per-cascaded-turn rate we evaluate each turn on its own merits and
            # do NOT let one escalation drop following turns from the denominator.
            state.escalated_this_episode = False

            f = extract(
                turn.instruction_text,
                context_tokens=turn.context_estimate,
                turn_index=idx,
                recent_errors=turn.n_error_results,
                recent_edit_failures=turn.n_edit_failures,
                prev_turn_interrupted=sturns[idx - 1].interrupted if idx > 0 else False,
                privacy_pinned=state.privacy_pinned,
            )

            t0 = time.perf_counter_ns()
            route = route_turn(f, state)
            store.set_route(traj_key, route.rung)
            store.incr_turn(traj_key)

            if not route.cascade:
                res.noncascade_layers[route.layer] += 1
                res.latencies_us.append((time.perf_counter_ns() - t0) / 1000)
                continue

            # ── cascaded: arm the trip-wires and drive the real action stream ──
            res.cascaded_turns += 1
            res.cascade_layers[route.layer] += 1
            store.reset_strikes(traj_key)     # per-turn strike reset (§5.5)
            tw = TripwireState(budget=None)   # headline: no invented cost budget

            hit: Optional[TripwireHit] = None
            for ev in turn.events:
                if ev[0] == "assistant":
                    hit = tw.observe_response(ev[1], output_tokens=ev[2])
                else:
                    hit = tw.observe_tool_results(ev[1])
                if hit is not None:
                    break

            # user_interrupt attribution: if the *next* turn is an interrupt-retry,
            # the user interrupted THIS turn (§5.5: 1 event, escalate the retry).
            if hit is None and idx + 1 < len(sturns) and sturns[idx + 1].interrupted:
                hit = tw.note_interrupt("user interrupted this turn; a retry followed")

            res.latencies_us.append((time.perf_counter_ns() - t0) / 1000)

            # mirror the turn's strike snapshot into the store (shape check, §5.6).
            # incr_strike (like Redis HINCRBY) only materialises a key once it is
            # incremented, so an un-tripped kind is an implicit zero — assert every
            # kind the trip-wire tracked is carried faithfully under that key.
            for kind, cnt in tw.strikes.items():
                for _ in range(cnt):
                    store.incr_strike(traj_key, kind)
            persisted = store.get_session(traj_key).strikes
            if any(persisted.get(kind, 0) != cnt for kind, cnt in tw.strikes.items()):
                res.strike_shape_ok = False

            fired_mechanical = hit is not None
            cascaded_output_tokens.append((turn.output_tokens_total, fired_mechanical))

            if hit is None:
                continue

            # ── would have escalated ──
            res.fired_cascaded += 1
            res.wire_names[hit.name] += 1
            res.wire_types[hit.type.value] += 1
            if routes_to_registry(hit.type):
                res.train_registry += 1
            else:
                res.train_router += 1

            if state.privacy_pinned:
                res.pin_blocked += 1          # target = USER, never cloud
            else:
                res.cloud_escalations += 1
                store.mark_escalated(sid)     # exercised; neutralised next turn

            # C2: rebuild this turn's transcript for the higher rung and validate.
            rebuilt = rebuild_for_escalation(build_messages(turn), failure_note(hit))
            ok, why = validate_rebuilt(rebuilt)
            if ok:
                res.rebuild_ok += 1
            else:
                res.rebuild_fail.append(f"{sid[:8]}/{turn.source_path}: {why}")

    # ── budget sensitivity (labeled, excluded from headline) ──
    if cascaded_output_tokens:
        toks = sorted(t for t, _ in cascaded_output_tokens)
        res.budget_median = statistics.median(toks)
        limit = BUDGET_TOKEN_MULTIPLIER * res.budget_median
        res.budget_added = sum(
            1 for t, fired in cascaded_output_tokens if not fired and t > limit
        )
    return res


# ── report ───────────────────────────────────────────────────────────────────
def render_report(res: ShadowResult, gate: float = 0.30) -> tuple[str, bool]:
    rate = res.fired_cascaded / res.cascaded_turns if res.cascaded_turns else 0.0
    gate_pass = rate < gate
    rebuild_pass = not res.rebuild_fail
    pin_cov = res.pinned_sessions_expected <= res.pinned_sessions_seen
    overall = gate_pass and rebuild_pass and pin_cov and res.strike_shape_ok

    lat = sorted(res.latencies_us)
    p50 = statistics.median(lat) if lat else 0.0
    p99 = lat[int(len(lat) * 0.99)] if lat else 0.0

    L: list[str] = []
    L += ["# P1 post-call shadow — escalation controller (offline, no live routing)", ""]
    L += [
        "Wires C1 `TripwireState` + C2 `rebuild_for_escalation` + C3 `SessionStore` "
        "over the historical corpus (§5.5 post-call flow). Cascade eligibility is the "
        "real `router.policy` decision; the escalation rate is measured **of cascaded "
        "turns** (§11). No live traffic is touched.",
        "",
    ]
    L += ["## Headline", ""]
    L += [
        f"- Turns replayed: **{res.total_turns}** "
        f"({res.cascaded_turns} cascaded / trip-wires armed) — assembled from the raw "
        f"corpus JSONL (the action-level `(response, tool_results)` stream the wires "
        f"need, which `turns.parquet` aggregates away; count differs from the parquet's "
        f"parentUuid segmentation, but the rate is a ratio and robust to that).",
        f"- Would-have-escalated (cascaded): **{res.fired_cascaded}**",
        f"- **Predicted escalation rate: {100 * rate:.2f}%** of cascaded turns "
        f"(Phase-2 gate §10: < {int(gate * 100)}%)",
        f"- Pin-blocked escalations (target = USER, never cloud): **{res.pin_blocked}**",
        f"- Cloud-bound escalations (target = higher rung): **{res.cloud_escalations}**",
        "",
        f"### Gate: {'PASS' if gate_pass else 'FAIL'} "
        f"({100 * rate:.2f}% {'<' if gate_pass else '>='} {int(gate * 100)}%)",
        "",
    ]

    L += ["## Trip-wire TYPE distribution (fired, cascaded)", "",
          "Type drives the flywheel split (§5.7/B5): DIALECT trains the capability "
          "registry, everything else the difficulty/router model.", "",
          "| Type | Trains | n |", "|---|---|---|"]
    type_train = {"dialect": "registry", "difficulty": "router", "cost": "router", "quality": "router"}
    for typ, n in res.wire_types.most_common():
        L.append(f"| {typ} | {type_train.get(typ, '?')} | {n} |")
    if not res.wire_types:
        L.append("| _(none)_ | — | 0 |")
    L += ["",
          f"Dialect escalations (train registry): **{res.train_registry}** · "
          f"difficulty/cost/quality (train router): **{res.train_router}**.", ""]

    L += ["## By wire", "", "| Wire | n |", "|---|---|"]
    for name, n in res.wire_names.most_common():
        L.append(f"| {name} | {n} |")
    if not res.wire_names:
        L.append("| _(none fired)_ | 0 |")
    L += [""]

    L += ["## Cascade band mix (denominator)", "", "| Policy layer | n |", "|---|---|"]
    for layer, n in res.cascade_layers.most_common():
        L.append(f"| {layer} | {n} |")
    L += ["",
          "Turns the router would NOT cascade (trip-wires never armed — excluded "
          "from the rate):", "", "| Policy layer | n |", "|---|---|"]
    for layer, n in res.noncascade_layers.most_common():
        L.append(f"| {layer} | {n} |")
    L += ["",
          "`gate:pin_context_conflict` (§5.8): a pinned session outgrew local "
          "context — no legal rung, surfaced to the user rather than cascaded.", ""]

    L += ["## Escalation transcript rebuild (C2)", "",
          f"- Rebuilt + validated on every fired turn: **{res.rebuild_ok} OK**, "
          f"{len(res.rebuild_fail)} malformed.",
          "- Validation: thinking stripped, tool_use/tool_result IDs paired "
          "(B5 internal-pairing), no empty messages."]
    if res.rebuild_fail:
        L.append("- **Failures:**")
        for fail in res.rebuild_fail[:10]:
            L.append(f"  - {fail}")
    L += [""]

    L += ["## Privacy pin", "",
          f"- Pinned sessions expected (A8 scan): **{len(res.pinned_sessions_expected)}**; "
          f"seen in replay: **{len(res.pinned_sessions_seen)}**"
          f"{' — PASS' if pin_cov else ' — FAIL (missing)'}.",
          "- Every escalation inside a pinned session targets the USER, never a "
          f"cloud rung ({res.pin_blocked} such events). A pin is one-way (§5.6).", ""]

    L += ["## Decision latency", "",
          f"- Per-turn route + full trip-wire drive: p50 **{p50:.1f}us**, p99 **{p99:.1f}us** "
          f"(§11 budget: <5ms sticky, <30ms decision path).",
          f"- SessionState.strikes shape matches TripwireState.strikes: "
          f"{'PASS' if res.strike_shape_ok else 'FAIL'}.", ""]

    L += ["## Cost-wire sensitivity (labeled, EXCLUDED from headline)", "",
          f"The `turn_budget` wire is a *cost* guard sized to {BUDGET_TOKEN_MULTIPLIER}x "
          f"median tokens **of the local model**. Replayed frontier token counts measure "
          f"frontier verbosity, not local runaway, so it is inert in the headline "
          f"(budget = None).",
          f"- If a {BUDGET_TOKEN_MULTIPLIER}x-median-token budget "
          f"(>{int(BUDGET_TOKEN_MULTIPLIER * res.budget_median)} output tok/turn) were "
          f"applied to these frontier turns, **{res.budget_added}** additional cascaded "
          f"turns would trip the cost wire — a frontier-verbosity artifact, not a local "
          f"failure signal.", ""]

    L += ["## Caveats (why this is a lower bound)", "",
          "- Corpus is 100% Claude Code **frontier** traffic. We ask what the wires "
          "would catch had these trajectories come from the **local** rung; the frontier "
          "model rarely fails mechanically, so mechanical wires fire rarely. A local "
          "model would fail *more* — this rate is a **lower bound**.",
          "- `parse_schema` reads 0 by construction: S5 (malformed local tool calls) is "
          "FALSIFIED for this corpus (`miner.scenarios`) — no local traffic exists to "
          "trip it. It needs workstream C (Ollama traffic via the capture proxy).",
          "- `edit_apply` requires `is_error` truthy (not just the marker string), so "
          "doc-mentions of the marker do not inflate the count (C1 design note).", ""]

    L += [f"**Verdict: {'PASS' if overall else 'FAIL'}** — "
          f"escalation gate {'PASS' if gate_pass else 'FAIL'}, "
          f"rebuild {'PASS' if rebuild_pass else 'FAIL'}, "
          f"pin coverage {'PASS' if pin_cov else 'FAIL'}."]
    return "\n".join(L) + "\n", overall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--secrets", type=Path, default=Path("data/secrets-scan.json"))
    ap.add_argument("--report", type=Path, default=Path("reports/p1-postcall-shadow.md"))
    ap.add_argument("--gate", type=float, default=0.30)
    args = ap.parse_args()

    pins = load_pins(args.secrets)
    res = run_shadow(args.corpus, pins)
    report, overall = render_report(res, gate=args.gate)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(report)
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
