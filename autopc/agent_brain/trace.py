"""
Action-trace recorder.

The skill crystallizer (skills/crystallizer.py) learns reusable playbooks from
successful runs -- but it needs a record of what the agent actually *did*. The
Claude Agent SDK does not hand that back as a tidy list, so we assemble one from
the two places the information naturally appears:

    thoughts   the assistant's text blocks, streamed in run.py's consumer and
               pushed here via `note_thought`
    actions    every physical action the executor runs, recorded from the one
               choke point they all pass through (tools._dispatch) via `record`
    outcome    the executor's own result string for that action

The result is a list of {thought, action, outcome} steps -- exactly the shape
the crystallizer and the working-memory tier expect. Observation-only calls
(observe, recall, read_screen) are deliberately not recorded: they are not steps
of the task, and crystallization wants the steps.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class TraceRecorder:
    """Accumulates the (thought, action, outcome) steps of one goal."""

    #: Action types that are not task steps and should never be crystallized.
    NON_STEP_TYPES = {"wait", "observe", "screenshot", "focus", "hover",
                      "hover_point"}

    def __init__(self) -> None:
        self.goal: str = ""
        self.steps: List[Dict[str, Any]] = []
        self.success: Optional[bool] = None
        self.result_summary: str = ""
        self._pending_thought: str = ""

    def start(self, goal: str) -> None:
        """Begin recording a fresh goal."""
        self.goal = goal
        self.steps = []
        self.success = None
        self.result_summary = ""
        self._pending_thought = ""

    def note_thought(self, text: str) -> None:
        """Remember the assistant's latest reasoning, to attach to its next action."""
        text = (text or "").strip()
        if text:
            self._pending_thought = text[:500]

    def record(self, action: Dict[str, Any], outcome: str, ok: bool = True) -> Dict[str, Any]:
        """Record one executed action and the outcome the executor reported."""
        step = {
            "thought": self._pending_thought,
            "action": action,
            "outcome": outcome,
            "ok": ok,
        }
        self.steps.append(step)
        self._pending_thought = ""
        return step

    def complete(self, success: bool, summary: str) -> None:
        """Mark the goal finished (called by the complete_task tool)."""
        self.success = bool(success)
        self.result_summary = (summary or "").strip()

    def meaningful_steps(self) -> List[Dict[str, Any]]:
        """Steps that count toward crystallization (real actions that succeeded)."""
        return [
            s for s in self.steps
            if s.get("ok", True)
            and s.get("action", {}).get("type") not in self.NON_STEP_TYPES
        ]

    def __len__(self) -> int:
        return len(self.steps)
