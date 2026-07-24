"""
Attempt state -- what the tools and the guard hooks share during one attempt.

The ledger (tasks/ledger.py) is the durable record. This is the live, in-memory
bookkeeping for a single attempt in a single agent context: how many completion
claims were rejected, whether the agent has (legitimately) given up, and the
counters the guard uses to detect loops and decide when to re-anchor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

from autopc.tasks.ledger import TaskLedger


@dataclass
class AttemptState:
    ledger: TaskLedger
    #: Completion claims the verifier found 'uncertain' (proof not on screen).
    uncertain_rejections: int = 0
    #: Completion claims rejected outright, of any kind.
    rejected_claims: int = 0
    #: complete_task(success=false) calls this attempt.
    failure_claims: int = 0
    #: The agent gave up after the 'not yet' pushback -- the attempt may end.
    gave_up: bool = False
    #: Interrupted by the user (pause/stop control or Ctrl+C).
    interrupted: bool = False

    # --- counters the guard maintains --------------------------------------
    tool_calls: int = 0
    stop_blocks: int = 0
    calls_since_anchor: int = 0
    reanchor_pending: bool = False
    unchanged_streak: int = 0
    failure_streak: int = 0
    recent: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=8))

    def may_end(self) -> bool:
        """True once this attempt has an accepted outcome and may stop."""
        return (self.ledger.status in ("verified", "needs_user")
                or self.gave_up or self.interrupted)
