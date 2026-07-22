"""
Task contract -- turn a goal into a definition of done before acting.

"Do X" is ambiguous about when X is finished, and an agent left to decide that
for itself tends to stop at the first screen that looks plausible. So before the
first action, the goal is converted into a contract:

    criteria   1-6 end-states that are observable on screen (or checkable
               facts), covering exactly what was asked -- nothing more
    plan       an ordered first draft of steps (the agent may revise it)
    assumptions any ambiguity resolved by a reasonable reading of the request

The criteria are what the independent verifier later checks a completion claim
against (tasks/verifier.py). They are written once per task and never relaxed
by the acting agent -- that is the core of task reinforcement: the target does
not move to meet the work.

If the contract call fails, a conservative fallback is used: one criterion that
restates the goal as a visible end-state. The task can always proceed.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

JsonFn = Callable[..., Awaitable[Dict[str, Any]]]

CONTRACT_SYSTEM = """\
You write the definition of done for a desktop automation agent that operates a
real Windows PC by looking at the screen and using the mouse and keyboard.

Given the user's request, produce:

criteria -- 1 to 6 conditions that, together, mean the request is fully done.
  * Each is an END-STATE you could confirm by looking at the screen (or a
    checkable fact), NOT an action. Write "Notepad shows the text 'hello'", not
    "type hello".
  * Cover EVERYTHING the user asked for, and NOTHING they did not. No extra
    polish, no side tasks, no "also make sure..." the user never requested.
  * Use the exact names, values, and text from the request.
  * If the request implies the result must persist or be delivered (save, send,
    submit, book, install), include a criterion that confirms it actually
    happened (e.g. "the title bar shows the saved filename with no unsaved
    marker", "a 'message sent' confirmation is visible").
plan -- 2 to 10 short, concrete, ordered steps to get there.
assumptions -- any ambiguity you resolved, stated briefly (empty if none).
"""

CONTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "plan": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criteria", "plan"],
}

MAX_CRITERIA = 6
MAX_PLAN = 10


@dataclass
class Contract:
    criteria: List[str]
    plan: List[str]
    assumptions: List[str] = field(default_factory=list)
    fallback: bool = False


def fallback_contract(goal: str) -> Contract:
    return Contract(
        criteria=[f"The requested outcome is visibly complete on screen: {goal.strip()}"],
        plan=[goal.strip()],
        fallback=True,
    )


def _clean(items: Any, limit: int) -> List[str]:
    out, seen = [], set()
    for it in items or []:
        s = " ".join(str(it).split())
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s[:400])
        if len(out) >= limit:
            break
    return out


async def derive_contract(goal: str, context: str = "",
                          generate_json: Optional[JsonFn] = None,
                          model: Optional[str] = None) -> Contract:
    """Build the contract for `goal`. `context` may carry relevant memory or a
    matched skill so the plan starts from experience. Never raises."""
    if generate_json is None:
        from autopc.agent_brain.llm import generate_json as _gj
        generate_json = _gj
    model = model or os.environ.get("AUTOPC_PLANNER_MODEL") or None
    prompt = f"User request:\n{goal.strip()}"
    if context.strip():
        prompt += f"\n\nWhat the agent already knows (use it to make the plan concrete):\n{context.strip()}"
    try:
        data = await generate_json(prompt, CONTRACT_SCHEMA, system=CONTRACT_SYSTEM, model=model)
    except Exception as exc:
        log.warning("Contract generation failed (%s); using fallback.", exc)
        return fallback_contract(goal)
    criteria = _clean(data.get("criteria"), MAX_CRITERIA)
    plan = _clean(data.get("plan"), MAX_PLAN)
    if not criteria:
        return fallback_contract(goal)
    return Contract(criteria=criteria, plan=plan or [goal.strip()],
                    assumptions=_clean(data.get("assumptions"), 6))


def apply_contract(ledger, contract: Contract) -> None:
    """Write a contract into a task ledger (criteria + initial plan)."""
    ledger.set_criteria(contract.criteria)
    ledger.set_plan([{"text": s, "status": "in_progress" if i == 0 else "pending"}
                     for i, s in enumerate(contract.plan)])
    note = f"Definition of done set ({len(contract.criteria)} criteria"
    note += ", fallback)" if contract.fallback else ")"
    if contract.assumptions:
        note += " Assumptions: " + "; ".join(contract.assumptions)
    ledger.log(note)
