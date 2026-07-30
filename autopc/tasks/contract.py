"""
Task contract -- the definition of done, derived before acting.

A deliberate NON-goal of this module is to plan the task. The agent's strength
is on-the-spot adaptability: it holds the goal in mind and decides every step
from what is actually on the screen, reacting to what it finds rather than
following a route guessed in advance. A pre-laid plan would work against that --
and where a reliable HOW already exists, a learned skill (skills/) supplies it
from real experience, not a guess. So we never prescribe steps.

What we DO derive up front is the definition of done: 1-6 conditions that are
observable on screen (or checkable facts) and that, together, mean the request
is fully satisfied. This is the WHAT, not the HOW. It does not constrain how the
agent works; it is the finish line the independent verifier (tasks/verifier.py)
checks a completion claim against -- which is what stops the agent declaring
victory early. The criteria are written once and never relaxed by the actor.

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

You do NOT plan the task or list steps -- the agent decides how to do it in the
moment, adapting to what it sees. Your only job is to state WHAT "done" means.

Given the user's request, produce:

criteria -- 1 to 6 conditions that, together, mean the request is fully done.
  * Each is an END-STATE you could confirm by looking at the screen (or a
    checkable fact), NOT an action or a step. Write "Notepad shows the text
    'hello'", never "type hello" or "open Notepad".
  * Cover EVERYTHING the user asked for, and NOTHING they did not. No extra
    polish, no side tasks, no "also make sure..." the user never requested.
  * Use the exact names, values, and text from the request.
  * If the request implies the result must persist or be delivered (save, send,
    submit, book, install), include a criterion that confirms it actually
    happened (e.g. "the title bar shows the saved filename with no unsaved
    marker", "a 'message sent' confirmation is visible").
  * Keep each criterion to what is genuinely necessary. Fewer, sharper criteria
    are better than many fussy ones.
assumptions -- any ambiguity you resolved by a reasonable reading of the
  request, stated briefly (empty list if none).
"""

CONTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criteria"],
}

MAX_CRITERIA = 6


@dataclass
class Contract:
    criteria: List[str]
    assumptions: List[str] = field(default_factory=list)
    fallback: bool = False


def fallback_contract(goal: str) -> Contract:
    return Contract(
        criteria=[f"The requested outcome is visibly complete on screen: {goal.strip()}"],
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
    """Derive the definition of done for `goal`. `context` may carry relevant
    memory or a matched skill. Never raises; never produces a plan."""
    if generate_json is None:
        from autopc.agent_brain.llm import generate_json as _gj
        generate_json = _gj
    model = model or os.environ.get("AUTOPC_PLANNER_MODEL") or None
    prompt = f"User request:\n{goal.strip()}"
    if context.strip():
        prompt += ("\n\nWhat the agent already knows (background only -- do NOT turn "
                   f"this into steps):\n{context.strip()}")
    try:
        data = await generate_json(prompt, CONTRACT_SCHEMA, system=CONTRACT_SYSTEM, model=model)
    except Exception as exc:
        log.warning("Contract generation failed (%s); using fallback.", exc)
        return fallback_contract(goal)
    criteria = _clean(data.get("criteria"), MAX_CRITERIA)
    if not criteria:
        return fallback_contract(goal)
    return Contract(criteria=criteria, assumptions=_clean(data.get("assumptions"), 6))


def apply_contract(ledger, contract: Contract) -> None:
    """Write a contract's definition of done into a task ledger. No plan is set:
    the agent improvises the steps and may keep its own plan if it chooses."""
    ledger.set_criteria(contract.criteria)
    note = f"Definition of done set ({len(contract.criteria)} criteria"
    note += ", fallback)" if contract.fallback else ")"
    if contract.assumptions:
        note += " Assumptions: " + "; ".join(contract.assumptions)
    ledger.log(note)
