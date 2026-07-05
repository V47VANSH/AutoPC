"""
A second, reasoning-only Claude session.

The agent that drives the desktop spends most of its turns on cheap work:
screenshot, scroll, click, type. None of that needs a frontier model. The
occasional hard part -- a tricky piece of logic, a self-contained coding
sub-problem -- does, and it is better worked out away from the perception loop.
So this module keeps a separate reasoning session (Claude Opus by default)
whose only job is to think.
It never sees the screen and never touches the mouse; the driver hands it a
fully described problem and reads back the answer.

Keeping one session open for the whole run buys two things. Spawning the CLI is
what makes the first call slow, so paying that once at boot keeps every later
call quick (`warm()` does it, and primes the cache). And since each `ask()` is
another turn in the same conversation, a follow-up can lean on what the model
already worked out instead of starting from scratch.

Auth is whatever the Claude Agent SDK is configured with, same as the driver.
Override the model with AUTOPC_SOLVER_MODEL and the effort level with
AUTOPC_SOLVER_EFFORT (low | medium | high | xhigh | max).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
    TextBlock,
)

from autopc import config

log = logging.getLogger("agent_brain.solver")

#: Reasoning depth. Defaults to "high"; raise it with AUTOPC_SOLVER_EFFORT when a
#: problem is worth the extra thinking time.
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def _effort() -> str:
    effort = (os.environ.get("AUTOPC_SOLVER_EFFORT") or "high").strip().lower()
    if effort not in _EFFORT_LEVELS:
        log.warning("AUTOPC_SOLVER_EFFORT=%r is not one of %s; using 'high'.",
                    effort, ", ".join(_EFFORT_LEVELS))
        return "high"
    return effort

SOLVER_SYSTEM_PROMPT = """\
You are the reasoning helper behind a computer-use agent. The agent works the
screen and the keyboard; you do the thinking it hands off. You never see the
screen -- you are given a self-contained problem as text and you return a worked
answer.

Think it through as carefully as the problem needs. Being right matters more
than being quick. If the text looks like it came off the screen through OCR,
read past the small noise (l/1, O/0, a dropped bracket) to the obvious intent.

Give the answer plainly:
- For a question, state the answer and, in a line or two, why.
- For code, return complete, correct, readable code that fits what you were
  given (existing signatures, language, style). If you were asked for code
  alone, reply with just the code and nothing around it.
"""


def build_solve_prompt(problem: str, context: str = "") -> str:
    """Frame a sub-problem for the solver: the problem itself, plus any context
    (constraints, data, starter code) the driver gathered off the screen."""
    parts = [problem.strip()]
    if context.strip():
        parts += ["", "Context:", context.strip()]
    return "\n".join(parts)


class Solver:
    """A persistent reasoning session the driver consults for hard sub-problems."""

    def __init__(self, model: Optional[str] = None, effort: Optional[str] = None):
        self.model = model or config.solver_model()
        self.effort = effort or _effort()
        self._client: Optional[ClaudeSDKClient] = None
        self._options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=SOLVER_SYSTEM_PROMPT,
            # Let it think as long as the sub-problem warrants. This session is
            # the opposite of the driver, which wants cheap, fast turns.
            thinking={"type": "adaptive"},
            effort=self.effort,
            # A pure reasoner: no desktop tools, no built-in tools, no repo
            # settings. It only ever emits text.
            mcp_servers={},
            tools=[],
            allowed_tools=[],
            permission_mode="bypassPermissions",
            setting_sources=[],
            # No tools, so one assistant turn per ask. Kept small as a backstop.
            max_turns=1,
            max_buffer_size=4 * 1024 * 1024,
        )

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def warm(self) -> None:
        """Connect the session and prime it, before any real question arrives."""
        if self._client is not None:
            return
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()
        try:
            await self.ask("Reply with the single word READY.")
        except Exception as exc:  # priming is best-effort
            log.warning("solver prime failed (non-fatal): %s", exc)

    async def ask(self, prompt: str) -> str:
        """Send one turn and return its text reply. Stays in the same
        conversation, so a follow-up can build on the previous answer."""
        if self._client is None:
            await self.warm()
        await self._client.query(prompt)
        chunks: list[str] = []
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                break
        return "".join(chunks).strip()

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None
