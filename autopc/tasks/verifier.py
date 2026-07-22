"""
Independent completion verifier.

The most common way an autonomous agent fails a long task is not a wrong click
-- it is declaring victory early. A screen that changed, a dialog that closed, a
step that "probably worked": the acting agent is the worst-placed judge of its
own success, because it shares every assumption that led it there.

So a completion claim is checked by a separate judge:

    * a FRESH screenshot and the on-screen text, taken at the moment of the claim
    * the task's definition of done (tasks/contract.py)
    * a separate model call with its own skeptical system prompt and no access
      to the agent's conversation -- the agent's summary is shown as a claim to
      test, never as evidence

The judge returns a per-criterion verdict. `decide()` turns that into accept /
reject, with a message telling the agent exactly what is still missing. A
rejected claim does not end the task -- the agent keeps working.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

JsonFn = Callable[..., Awaitable[Dict[str, Any]]]

VERIFIER_SYSTEM = """\
You are a strict, independent verifier for a desktop automation agent. The agent
claims it has completed a task. You decide whether that is TRUE, judging ONLY
from the current screen (the screenshot and the on-screen text provided).

Rules:
* The agent's claim is NOT evidence. Only what you can see counts.
* A criterion is met only if the screen positively shows it. Plausible is not
  enough. A dialog that is still open, an unsaved-changes marker, a draft not
  sent, a value that differs from the request -- these mean NOT met.
* If a criterion simply cannot be confirmed from this screen (e.g. a file was
  supposedly saved but nothing on screen shows it), mark it met=false and say in
  `reason` what would need to be visible to confirm it.
* Judge only what the criteria ask. Do not fail the task for things the user
  never requested.

overall:
  "pass"      every criterion is met on screen
  "fail"      at least one criterion is visibly NOT met
  "uncertain" nothing is visibly wrong, but some criterion cannot be confirmed
              from this screen
missing: one or two sentences -- what is still left to do, or to show.
"""

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "met": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "met", "reason"],
            },
        },
        "overall": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "missing": {"type": "string"},
    },
    "required": ["criteria", "overall", "missing"],
}

#: How many 'uncertain' rejections in one attempt before an uncertain verdict is
#: accepted on the agent's own evidence. Asking the agent to put proof on screen
#: is valuable (open the folder, show the sent item) -- but not forever.
MAX_UNCERTAIN_REJECTIONS = 2


@dataclass
class Verdict:
    overall: str = "uncertain"
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    missing: str = ""
    error: str = ""

    def unmet(self) -> List[Dict[str, Any]]:
        return [c for c in self.criteria if not c.get("met")]


def _build_prompt(ledger, screen_text: str, window: str, claim: str) -> str:
    crit = "\n".join(f"  {c.id}: {c.text}" for c in ledger.criteria)
    agent_marks = "\n".join(
        f"  {c.id}: agent says {c.status}" + (f" -- {c.evidence}" if c.evidence else "")
        for c in ledger.criteria)
    return (
        f"TASK: {ledger.goal}\n\n"
        f"DEFINITION OF DONE:\n{crit}\n\n"
        f"AGENT'S CLAIM (to test, not to trust): {claim or '(none given)'}\n"
        f"Agent's own per-criterion marks:\n{agent_marks}\n\n"
        f"ACTIVE WINDOW: {window or '(unknown)'}\n\n"
        f"ON-SCREEN TEXT (OCR, top to bottom; may be incomplete):\n{screen_text or '(none)'}\n\n"
        "The screenshot is attached. Return a verdict for EVERY criterion id."
    )


def _normalize(data: Dict[str, Any], ledger) -> Verdict:
    by_id = {}
    for c in data.get("criteria") or []:
        cid = str(c.get("id", "")).strip()
        if cid:
            by_id[cid] = {"id": cid, "met": bool(c.get("met")),
                          "reason": str(c.get("reason", ""))[:400]}
    rows = []
    for c in ledger.criteria:
        # A criterion the judge skipped is unconfirmed -- never silently 'met'.
        rows.append(by_id.get(c.id, {"id": c.id, "met": False,
                                     "reason": "verifier gave no verdict for this criterion"}))
    overall = str(data.get("overall", "uncertain")).lower()
    if overall not in ("pass", "fail", "uncertain"):
        overall = "uncertain"
    # 'pass' is only credible if every criterion is actually met.
    if overall == "pass" and any(not r["met"] for r in rows):
        overall = "uncertain"
    return Verdict(overall=overall, criteria=rows, missing=str(data.get("missing", ""))[:600])


async def verify(ledger, screenshot=None, screen_text: str = "", window: str = "",
                 claim: str = "", generate_json: Optional[JsonFn] = None,
                 model: Optional[str] = None) -> Verdict:
    """Judge a completion claim against the current screen. Never raises: an
    unavailable verifier yields a Verdict with `error` set."""
    if not ledger.criteria:
        return Verdict(overall="uncertain", error="task has no definition of done")
    if generate_json is None:
        from autopc.agent_brain.llm import generate_json as _gj
        generate_json = _gj
    try:
        data = await generate_json(
            _build_prompt(ledger, screen_text, window, claim), VERIFIER_SCHEMA,
            system=VERIFIER_SYSTEM, model=model,
            images=[screenshot] if screenshot is not None else None)
        return _normalize(data, ledger)
    except Exception as exc:
        log.warning("Verifier unavailable: %s", exc)
        return Verdict(overall="uncertain", error=f"{type(exc).__name__}: {exc}")


def decide(verdict: Verdict, ledger, uncertain_rejections: int = 0) -> Tuple[bool, str]:
    """Accept or reject a completion claim. Returns (accepted, message-for-agent)."""
    if verdict.error:
        if ledger.all_marked_met() and all(c.evidence for c in ledger.criteria):
            return True, ("Accepted on your own evidence (the independent verifier was "
                          f"unavailable: {verdict.error}).")
        return False, ("The verifier is unavailable, so completion needs your own evidence: "
                       "call mark_criterion for EVERY criterion with what on screen proves it, "
                       "then call complete_task again.")

    if verdict.overall == "pass":
        return True, "Verified: every criterion is met on screen."

    unmet = "\n".join(f"  - {c['id']}: {c['reason']}" for c in verdict.unmet())
    if verdict.overall == "fail":
        return False, ("NOT VERIFIED -- the task is not done yet. An independent check of the "
                       f"current screen found:\n{unmet}\nStill missing: {verdict.missing}\n"
                       "Keep working on exactly these points, then call complete_task again.")

    # uncertain: nothing visibly wrong, but something cannot be confirmed here.
    if uncertain_rejections >= MAX_UNCERTAIN_REJECTIONS and ledger.all_marked_met():
        return True, ("Accepted: nothing on screen contradicts completion and you have marked "
                      "every criterion with evidence.")
    return False, ("NOT YET VERIFIED -- the current screen does not show proof of:\n"
                   f"{unmet}\nWhat would show it: {verdict.missing}\n"
                   "Bring that proof onto the screen (for example open the folder that holds "
                   "the saved file, or the sent-items view), then call complete_task again. "
                   "If it genuinely cannot be shown on screen, call mark_criterion for it with "
                   "your evidence, then claim again.")


def record(ledger, verdict: Verdict, accepted: bool) -> None:
    """Store the verdict and sync per-criterion status with the judge's view."""
    ledger.verdicts.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "overall": verdict.overall, "accepted": accepted,
        "missing": verdict.missing, "error": verdict.error,
        "criteria": verdict.criteria,
    })
    if accepted:
        for c in ledger.criteria:
            c.status = "met"
    elif not verdict.error:
        for row in verdict.criteria:
            for c in ledger.criteria:
                if c.id == row["id"]:
                    c.status = "met" if row["met"] else "unmet"
                    if row.get("reason"):
                        c.evidence = f"verifier: {row['reason']}"[:600]
    ledger.log(f"Completion claim {'ACCEPTED' if accepted else 'rejected'} "
               f"(verifier: {verdict.overall}{', ' + verdict.error if verdict.error else ''}).")
