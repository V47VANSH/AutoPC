"""
Goal-reinforcement hooks -- the task is the agent's only objective, enforced.

Prompting an agent to "stay on task" works until it doesn't: the context fills,
gets compacted, and the goal blurs; a repeated action fails silently; the model
ends its turn with a polite summary of a task it has not finished, or asks a
question nobody is there to answer. The guard enforces the objective with the
Claude Agent SDK's hooks rather than with more prompt text:

    Stop                the completion gate. The agent may only end its turn
                        once the attempt has an accepted outcome: the task was
                        independently VERIFIED (tasks/verifier.py), a blocker
                        only the user can clear was recorded, or it gave up
                        after being pushed back. Otherwise the stop is blocked
                        and the agent is handed the goal, what is still
                        unmet, and its next step. Bounded, so it cannot loop
                        forever -- after the cap the attempt ends and the runner
                        reflects and retries in a fresh context.
    PostToolUse         re-anchors the goal every N calls (and right after a
                        context compaction), detects loops (the same action on
                        the same target, repeatedly), stalls (actions that
                        change nothing on screen), failure streaks, and
                        observe-without-acting; periodically nudges the agent to
                        checkpoint progress and save durable knowledge.
    PostToolUseFailure  counts tool errors toward the failure streak.
    PreCompact          checkpoints the ledger and schedules a full re-anchor,
                        because compaction is exactly when a goal gets lost.

A user's stop request (core.control) always wins over the gate.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from autopc.tasks.state import AttemptState

log = logging.getLogger(__name__)

TOOL_PREFIX = "mcp__desktop__"

#: Tools that physically change the desktop. Loop/stall detection applies here.
ACTING_TOOLS = {"click_point", "hover_point", "drag", "type_text", "type_code",
                "press_key", "scroll", "navigate"}
#: Tools that only look.
LOOKING_TOOLS = {"observe", "read_screen"}


def _short(tool_name: str) -> str:
    return tool_name[len(TOOL_PREFIX):] if tool_name.startswith(TOOL_PREFIX) else tool_name


def _response_text(resp: Any) -> str:
    """Flatten an MCP tool response (dict/list/str) to its text."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        if "content" in resp:
            return _response_text(resp["content"])
        if resp.get("type") == "text":
            return str(resp.get("text", ""))
        return json.dumps(resp)[:2000]
    if isinstance(resp, list):
        return "\n".join(_response_text(x) for x in resp)
    return str(resp)


def _is_error(resp: Any, text: str) -> bool:
    if isinstance(resp, dict) and resp.get("isError"):
        return True
    return text.lstrip().startswith("FAILED")


def _signature(short: str, tool_input: Dict[str, Any]) -> str:
    """A target-level fingerprint of an action. Coordinates are bucketed so two
    clicks a few pixels apart on the same control count as the same action."""
    parts = []
    for k in sorted(tool_input or {}):
        v = tool_input[k]
        if isinstance(v, (int, float)) and k in ("x", "y", "start_x", "start_y", "end_x", "end_y"):
            v = int(v) // 15
        elif isinstance(v, str):
            v = v.strip().lower()[:80]
        parts.append(f"{k}={v}")
    return short + "(" + ",".join(parts) + ")"


def _user_stopped() -> bool:
    try:
        from autopc.core.control import CONTROL
        a = CONTROL.aborted
        return bool(a() if callable(a) else a)
    except Exception:
        return False


class TaskGuard:
    def __init__(self, state: AttemptState, *, reanchor_every: int = 10,
                 nudge_every: int = 25, max_stop_blocks: int = 6,
                 loop_repeat: int = 3, stall_limit: int = 3,
                 failure_limit: int = 3, look_limit: int = 6):
        self.state = state
        self.reanchor_every = reanchor_every
        self.nudge_every = nudge_every
        self.max_stop_blocks = max_stop_blocks
        self.loop_repeat = loop_repeat
        self.stall_limit = stall_limit
        self.failure_limit = failure_limit
        self.look_limit = look_limit
        self._looks_in_a_row = 0

    # ------------------------------------------------------------- wiring
    def hooks(self) -> Dict[str, List[Any]]:
        """Hook configuration for ClaudeAgentOptions(hooks=...)."""
        from claude_agent_sdk import HookMatcher
        return {
            "Stop": [HookMatcher(hooks=[self.on_stop])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[self.on_post_tool])],
            "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[self.on_tool_failure])],
            "PreCompact": [HookMatcher(hooks=[self.on_pre_compact])],
        }

    # -------------------------------------------------------- stop gate
    def continuation(self, last_message: str = "") -> str:
        L = self.state.ledger
        lines = ["You are trying to stop, but the task is NOT finished.", "", L.anchor()]
        if L.verdicts and not L.verdicts[-1].get("accepted"):
            lines.append(f"Last independent check said still missing: "
                         f"{L.verdicts[-1].get('missing', '')}")
        if last_message.rstrip().endswith("?"):
            lines += ["", "Nobody is available to answer questions -- you are running "
                      "autonomously. Make the reasonable choice that best serves the goal "
                      "and keep going."]
        lines += ["",
                  "Continue working now. Observe the screen and take the next step.",
                  "When -- and only when -- every criterion is visibly met, call "
                  "complete_task(success=true, summary=...). It is checked independently.",
                  "If something only the user can provide is blocking you, call "
                  "report_blocker(description, needs_user=true), then "
                  "complete_task(success=false, ...)."]
        return "\n".join(lines)

    async def on_stop(self, input_data: Dict[str, Any], tool_use_id: Optional[str],
                      context: Any) -> Dict[str, Any]:
        A, L = self.state, self.state.ledger
        if A.may_end():
            return {}
        if _user_stopped():
            A.interrupted = True
            L.log("Stopped by the user.")
            return {}
        if A.stop_blocks >= self.max_stop_blocks:
            L.log(f"Stop gate released after {A.stop_blocks} continuations; "
                  "ending the attempt for review.")
            return {}
        A.stop_blocks += 1
        if L.current_attempt:
            L.current_attempt.stop_blocks = A.stop_blocks
        L.log(f"Premature stop blocked ({A.stop_blocks}/{self.max_stop_blocks}).")
        return {"decision": "block",
                "reason": self.continuation(str(input_data.get("last_assistant_message", "")))}

    # ------------------------------------------------- per-tool supervision
    def _anchor_block(self) -> str:
        A = self.state
        return (f"--- TASK ANCHOR (tool call {A.tool_calls}) ---\n{A.ledger.anchor()}\n"
                "Work only toward this. Do nothing the task did not ask for.")

    def advise(self, tool_name: str, tool_input: Dict[str, Any], response: Any,
               failed: bool = False) -> str:
        """Update counters for one tool call and return guidance (or '')."""
        A, L = self.state, self.state.ledger
        short = _short(tool_name)
        text = _response_text(response)
        A.tool_calls += 1
        A.calls_since_anchor += 1
        if L.current_attempt:
            L.current_attempt.tool_calls = A.tool_calls
        notes: List[str] = []

        if short in ACTING_TOOLS:
            self._looks_in_a_row = 0
            failed = failed or _is_error(response, text)
            A.failure_streak = A.failure_streak + 1 if failed else 0
            if "UNCHANGED" in text:
                A.unchanged_streak += 1
            elif "changed" in text:
                A.unchanged_streak = 0
            sig = _signature(short, tool_input)
            A.recent.append((sig, "fail" if failed else "ok"))
            repeats = sum(1 for s, _ in A.recent if s == sig)
            if repeats >= self.loop_repeat:
                notes.append(f"LOOP DETECTED: you have done {short} on the same target "
                             f"{repeats} times recently. It is not working -- do not repeat "
                             "it again. Change approach: a keyboard shortcut, a different "
                             "control or menu, scroll to reveal the real target, close "
                             "whatever is in the way, or observe again to re-check where you are.")
                A.recent.clear()
            if A.unchanged_streak >= self.stall_limit:
                notes.append(f"STALLED: your last {A.unchanged_streak} actions changed nothing "
                             "on screen. They are landing on nothing, on a disabled control, "
                             "or the window is not focused. Observe, then try a different route.")
                A.unchanged_streak = 0
            if A.failure_streak >= self.failure_limit:
                notes.append(f"FAILING: your last {A.failure_streak} actions failed. The screen "
                             "is probably not what you think it is -- observe before acting again.")
                A.failure_streak = 0
        elif short in LOOKING_TOOLS:
            self._looks_in_a_row += 1
            if self._looks_in_a_row >= self.look_limit:
                notes.append(f"You have looked {self._looks_in_a_row} times without acting. "
                             "Decide on the next step and take it.")
                self._looks_in_a_row = 0

        if short != "complete_task" and not A.may_end():
            if A.reanchor_pending or A.calls_since_anchor >= self.reanchor_every:
                notes.append(self._anchor_block())
                A.calls_since_anchor = 0
                A.reanchor_pending = False
            if self.nudge_every and A.tool_calls % self.nudge_every == 0:
                notes.append("CHECKPOINT: call log_progress with where things stand, mark any "
                             "criteria you have now met, and `remember` any durable fact you "
                             "learned about this app or the user.")

        if A.tool_calls % 5 == 0:
            L.save()
        return "\n\n".join(notes)

    async def on_post_tool(self, input_data: Dict[str, Any], tool_use_id: Optional[str],
                           context: Any) -> Dict[str, Any]:
        msg = self.advise(input_data.get("tool_name", ""), input_data.get("tool_input") or {},
                          input_data.get("tool_response"))
        if not msg:
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                       "additionalContext": msg}}

    async def on_tool_failure(self, input_data: Dict[str, Any], tool_use_id: Optional[str],
                              context: Any) -> Dict[str, Any]:
        if input_data.get("is_interrupt"):
            return {}
        msg = self.advise(input_data.get("tool_name", ""), input_data.get("tool_input") or {},
                          {"isError": True, "content": str(input_data.get("error", ""))},
                          failed=True)
        if not msg:
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PostToolUseFailure",
                                       "additionalContext": msg}}

    async def on_pre_compact(self, input_data: Dict[str, Any], tool_use_id: Optional[str],
                             context: Any) -> Dict[str, Any]:
        A, L = self.state, self.state.ledger
        L.log(f"Context compaction ({input_data.get('trigger', '?')}) at tool call "
              f"{A.tool_calls}; goal will be re-anchored.")
        A.reanchor_pending = True
        return {}
