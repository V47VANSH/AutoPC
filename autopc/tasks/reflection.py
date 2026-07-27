"""
Reflection and episodic recall -- learning from every attempt, not just wins.

Skills (skills/) capture HOW to do a task once it has been done right. That
leaves two gaps that matter most on long, hard tasks:

  * a FAILED attempt teaches the most, and a skill cannot be distilled from it;
  * a fresh attempt in a fresh context would otherwise repeat the same mistake.

After every attempt a reviewer (a separate model call) reads the ledger and the
action trace and returns:

    summary        what happened, in a sentence or two
    lessons        specific, actionable guidance for the next attempt at this or
                   a similar task ("Edge's Downloads panel opens with Ctrl+J;
                   the toolbar icon kept missing")
    facts          durable knowledge true beyond this task (an app's layout, a
                   user preference) -> saved to long-term core memory
    next_approach  on failure: the concrete, DIFFERENT approach to try next

This is the Reflexion pattern. Lessons feed the next attempt's context; facts
go to core memory; and the whole attempt is stored as an EPISODE in archival
memory, so a similar task next week starts with "last time, this is what
happened" -- recalled automatically, not only when the agent thinks to search.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

JsonFn = Callable[..., Awaitable[Dict[str, Any]]]

REFLECT_SYSTEM = """\
You review one attempt by a desktop automation agent (it operates a real Windows
PC with mouse and keyboard) and extract what should be carried forward.

Be concrete and specific to what actually happened in the trace. Generic advice
("be careful", "verify your work") is worthless -- name the app, the control,
the shortcut, the dialog, the exact mistake.

summary        1-2 sentences: what happened and why it ended as it did.
lessons        0-5 actionable lessons for the next attempt at this or a similar
               task. Prefer what to DO differently over what went wrong.
facts          0-5 durable facts that stay true beyond this task (how an app is
               laid out, where a setting lives, a user preference). key is a
               short stable snake_case label. Only facts the trace supports.
next_approach  if the attempt did not succeed: the concrete, DIFFERENT approach
               the next attempt should take. Empty string on success.
"""

REFLECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "lessons": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"]}},
        "next_approach": {"type": "string"},
    },
    "required": ["summary", "lessons", "facts", "next_approach"],
}


@dataclass
class Reflection:
    summary: str = ""
    lessons: List[str] = field(default_factory=list)
    facts: List[Dict[str, str]] = field(default_factory=list)
    next_approach: str = ""
    error: str = ""


def _fmt_action(a: Dict[str, Any]) -> str:
    a = dict(a or {})
    t = a.pop("type", "?")
    a.pop("label", None)
    if "text" in a and isinstance(a["text"], str) and len(a["text"]) > 60:
        a["text"] = a["text"][:57] + "..."
    return f"{t} {json.dumps(a, ensure_ascii=False)[:140]}"


def _build_prompt(ledger, outcome: str, steps: List[Dict[str, Any]]) -> str:
    out = [f"TASK: {ledger.goal}", f"ATTEMPT OUTCOME: {outcome}", "", "DEFINITION OF DONE:"]
    out += [f"  {c.id} [{c.status}] {c.text}" + (f" -- {c.evidence}" if c.evidence else "")
            for c in ledger.criteria]
    if ledger.plan:
        out += ["", "PLAN:"] + [f"  [{p.status}] {p.text}" for p in ledger.plan]
    if ledger.verdicts:
        out += ["", "INDEPENDENT VERIFICATIONS (latest last):"]
        out += [f"  {v.get('overall')} accepted={v.get('accepted')} missing={v.get('missing', '')}"
                for v in ledger.verdicts[-3:]]
    if ledger.blockers:
        out += ["", "BLOCKERS:"] + [f"  {b['text']}" for b in ledger.blockers[-4:]]
    if ledger.progress:
        out += ["", "PROGRESS JOURNAL (latest last):"] + [f"  {p['note']}" for p in ledger.progress[-20:]]
    if steps:
        out += ["", f"ACTION TRACE (last {min(40, len(steps))} of {len(steps)}):"]
        for i, s in enumerate(steps[-40:], 1):
            thought = (s.get("thought") or "").replace("\n", " ")[:120]
            line = f"  {i}. {_fmt_action(s.get('action'))} -> {str(s.get('outcome', ''))[:120]}"
            out.append(line + (f"   [why: {thought}]" if thought else ""))
    return "\n".join(out)


async def reflect(ledger, outcome: str, steps: Optional[List[Dict[str, Any]]] = None,
                  generate_json: Optional[JsonFn] = None,
                  model: Optional[str] = None) -> Reflection:
    """Review one attempt. Never raises; returns a minimal Reflection on error."""
    if generate_json is None:
        from autopc.agent_brain.llm import generate_json as _gj
        generate_json = _gj
    model = model or os.environ.get("AUTOPC_REFLECT_MODEL") or None
    try:
        data = await generate_json(_build_prompt(ledger, outcome, steps or []),
                                   REFLECT_SCHEMA, system=REFLECT_SYSTEM, model=model)
    except Exception as exc:
        log.warning("Reflection failed: %s", exc)
        return Reflection(summary=f"Attempt ended: {outcome}.", error=str(exc))
    facts = []
    for f in data.get("facts") or []:
        k, v = str(f.get("key", "")).strip(), str(f.get("value", "")).strip()
        if k and v:
            facts.append({"key": k[:60], "value": v[:400]})
    return Reflection(
        summary=str(data.get("summary", ""))[:600],
        lessons=[str(l).strip()[:300] for l in (data.get("lessons") or []) if str(l).strip()][:5],
        facts=facts[:5],
        next_approach=str(data.get("next_approach", ""))[:600],
    )


def episode_text(ledger, outcome: str, r: Reflection) -> str:
    when = datetime.now().strftime("%Y-%m-%d")
    parts = [f"EPISODE {when} | goal: {ledger.goal} | outcome: {outcome}"]
    if r.summary:
        parts.append(f"summary: {r.summary}")
    if r.lessons:
        parts.append("lessons: " + " ; ".join(r.lessons))
    return " | ".join(parts)


def apply_reflection(ledger, outcome: str, r: Reflection, memory=None) -> None:
    """Carry a reflection forward: into the ledger (for the next attempt), core
    memory (durable facts), and archival memory (the episode)."""
    known = {l.lower() for l in ledger.lessons}
    for l in r.lessons:
        if l.lower() not in known:
            ledger.lessons.append(l)
            known.add(l.lower())
    a = ledger.attempts[-1] if ledger.attempts else None
    if a is not None:
        a.reflection = (r.summary + (f" Next: {r.next_approach}" if r.next_approach else ""))[:2000]
    ledger.log(f"Reflection: {r.summary}" + (f" Next approach: {r.next_approach}"
                                             if r.next_approach else ""))
    if memory is None:
        return
    for f in r.facts:
        try:
            memory.memory_save(f["key"], f["value"])
        except Exception as exc:
            log.debug("could not save fact %s: %s", f, exc)
    try:
        memory.vectors.add(episode_text(ledger, outcome, r),
                           {"type": "episode", "task_id": ledger.id,
                            "goal": ledger.goal, "outcome": outcome})
    except Exception as exc:
        log.debug("could not store episode: %s", exc)


def similar_episodes(memory, goal: str, k: int = 3) -> str:
    """Past episodes resembling `goal`, formatted for the agent's context."""
    if memory is None:
        return ""
    try:
        hits = memory.vectors.search(f"EPISODE goal: {goal}", top_k=k * 4)
    except Exception:
        return ""
    floor = 0.55 if getattr(memory.vectors, "semantic", False) else 0.12
    eps = [h for h in hits if (h.get("meta") or {}).get("type") == "episode"
           and h.get("score", 0) >= floor][:k]
    if not eps:
        return ""
    return "\n".join(f"  - {h['text']}" for h in eps)
