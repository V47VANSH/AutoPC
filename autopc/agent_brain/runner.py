"""
Task runner -- orchestrates a goal to a verified conclusion.

This is where the long-horizon pieces come together:

  1. CONTRACT. A new goal is turned into a definition of done (tasks/contract.py)
     before any action, informed by relevant memory, skills, and past episodes.
  2. ATTEMPTS. The agent works the task in a FRESH context per attempt -- the
     Reflexion insight: a new try should not inherit the previous one's dead
     ends, only its distilled lessons. Each attempt is one SDK `query()` with
     the goal-reinforcement guard hooks bound (tasks/guard.py), so the agent
     cannot drift off-task or stop early.
  3. VERIFY. Completion is gated by the independent verifier inside
     complete_task; the runner just reads the resulting ledger status.
  4. REFLECT. After every attempt, a reviewer extracts lessons, durable facts
     and an episode (tasks/reflection.py). Lessons feed the next attempt; facts
     and the episode go to long-term memory. On a verified success a skill is
     crystallized.
  5. RETRY until verified, a user is needed, the user interrupts, or the attempt
     budget is spent (then the task is marked failed, with everything learned
     preserved for a future `resume`).

Nothing lives only in memory: the ledger is on disk throughout, so a crash
mid-attempt loses at most the current attempt, and `resume` continues. A Ctrl+C
arrives as task cancellation: the attempt is recorded as interrupted, saved, and
the cancellation is re-raised to the caller.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

from claude_agent_sdk import query

from autopc import config
from autopc.agent_brain import tools as desktop
from autopc.tasks.contract import apply_contract, derive_contract
from autopc.tasks.guard import TaskGuard
from autopc.tasks.ledger import TaskLedger
from autopc.tasks.reflection import apply_reflection, reflect, similar_episodes
from autopc.tasks.state import AttemptState
from autopc.tasks.store import TaskStore

log = logging.getLogger("autopc.runner")


def _default_max_attempts() -> int:
    return max(1, config.env_int("AUTOPC_MAX_ATTEMPTS", 3))


class TaskRunner:
    def __init__(self, store: TaskStore, *, consume, build_options,
                 model: str, max_turns: int = 80, budget_usd: float = 4.0,
                 max_attempts: Optional[int] = None):
        self.store = store
        self._consume = consume              # agent_brain.run._consume
        self._build_options = build_options  # agent_brain.run._build_options
        self.model = model
        self.max_turns = max_turns
        self.budget_usd = budget_usd
        self.max_attempts = max_attempts or _default_max_attempts()

    # ------------------------------------------------------------ public API
    async def run(self, goal: str) -> TaskLedger:
        """Create a task for `goal` and drive it to a terminal status."""
        ledger = self.store.create(goal, status="active")
        return await self._drive(ledger)

    async def resume(self, ledger: TaskLedger) -> TaskLedger:
        """Continue an unfinished task from its on-disk ledger."""
        ledger.log("Resumed.")
        return await self._drive(ledger)

    # --------------------------------------------------------------- context
    def _episodes(self, goal: str) -> str:
        return similar_episodes(desktop.get_memory(), goal, k=3)

    async def _ensure_contract(self, ledger: TaskLedger) -> None:
        if ledger.criteria:
            return  # already has a definition of done (e.g. on resume)
        ctx_parts = [desktop.memory_preamble(), desktop.skills_preamble(ledger.goal),
                     self._episodes(ledger.goal)]
        context = "\n\n".join(p for p in ctx_parts if p and p.strip())
        contract = await derive_contract(ledger.goal, context=context,
                                         model=os.environ.get("AUTOPC_PLANNER_MODEL") or None)
        apply_contract(ledger, contract)

    def _attempt_prompt(self, ledger: TaskLedger, attempt_n: int) -> str:
        blocks = [
            "You are an autonomous desktop agent. Completing the task below is your "
            "ONLY objective for this session. Work until every criterion in the "
            "definition of done is visibly met on screen, then call "
            "complete_task(success=true). You run unattended: never wait for or ask "
            "the user -- make the reasonable choice and proceed.",
            ledger.report(),
        ]
        mem = desktop.memory_preamble()
        if mem.strip():
            blocks.append("## What you already know\n" + mem)
        skl = desktop.skills_preamble(ledger.goal)
        if skl.strip():
            blocks.append(skl)
        eps = self._episodes(ledger.goal)
        if eps.strip():
            blocks.append("## Related past attempts (yours)\n" + eps)
        if attempt_n > 1:
            blocks.append(
                f"## This is attempt {attempt_n}\nEarlier attempts did not finish the "
                "task. Read the lessons above and take a DIFFERENT approach -- repeating "
                "what already failed will fail again. Start by calling task_status and "
                "observe to re-ground yourself.")
        blocks.append("Begin now: call observe to see the screen, then act.")
        return "\n\n".join(blocks)

    # --------------------------------------------------------------- driving
    async def _drive(self, ledger: TaskLedger) -> TaskLedger:
        await self._ensure_contract(ledger)
        memory = desktop.get_memory()

        while not ledger.finished:
            done_attempts = sum(1 for a in ledger.attempts if a.ended)
            if done_attempts >= self.max_attempts:
                ledger.finish("failed",
                              "Attempt budget exhausted without an independently verified "
                              "completion. Progress and lessons are saved; resume to continue.")
                break

            attempt = ledger.begin_attempt()
            state = AttemptState(ledger=ledger)
            guard = TaskGuard(state)
            desktop.bind_attempt(state)
            desktop.start_trace(ledger.goal)
            options = self._build_options(self.model, self.max_turns, self.budget_usd,
                                          hooks=guard.hooks())
            print("=" * 64)
            print(f"TASK {ledger.id}  |  attempt {attempt.n}/{self.max_attempts}")
            print(f"GOAL: {ledger.goal}")
            print("=" * 64)
            started = time.time()
            try:
                await self._consume(query(prompt=self._attempt_prompt(ledger, attempt.n),
                                          options=options), started, 0)
            except asyncio.CancelledError:
                # Ctrl+C. Record the attempt so `resume` picks up cleanly, then
                # let the cancellation reach the caller.
                state.interrupted = True
                ledger.log("Interrupted by the user mid-attempt.")
                ledger.end_attempt("interrupted", "interrupted by the user")
                ledger.save()
                print(f"\nattempt {attempt.n} -> interrupted (resume with: "
                      f"autopc resume {ledger.id})")
                raise
            except Exception as exc:
                log.exception("Attempt crashed")
                ledger.log(f"Attempt {attempt.n} crashed: {type(exc).__name__}: {exc}")
            finally:
                # Grade surfaced skills and crystallize on a verified success.
                try:
                    note = await desktop.finish_goal()
                    if note and note != "no skill changes":
                        print(f"skills      : {note}")
                except Exception:
                    pass
                desktop.bind_attempt(None)

            outcome = self._outcome(ledger, state)
            # Don't spend an LLM call reflecting on an attempt the user cut short.
            if state.interrupted:
                ledger.end_attempt(outcome, "interrupted by the user")
                print(f"attempt {attempt.n} -> interrupted")
                break
            steps = desktop.trace().steps
            r = await reflect(ledger, outcome, steps,
                              model=os.environ.get("AUTOPC_REFLECT_MODEL") or None)
            apply_reflection(ledger, outcome, r, memory=memory)
            ledger.end_attempt(outcome, r.summary)
            print(f"attempt {attempt.n} -> {outcome}: {r.summary[:160]}")

            if outcome in ("verified", "needs_user"):
                break

        ledger.save()
        self._print_result(ledger)
        return ledger

    @staticmethod
    def _outcome(ledger: TaskLedger, state: AttemptState) -> str:
        if ledger.status == "verified":
            return "verified"
        if ledger.status == "needs_user":
            return "needs_user"
        if state.interrupted:
            return "interrupted"
        if state.gave_up:
            return "gave_up"
        return "exhausted"   # ran out of turns/budget, or the stop gate released

    def _print_result(self, ledger: TaskLedger) -> None:
        print("\n" + "=" * 64)
        print(f"TASK {ledger.id}  FINAL STATUS: {ledger.status.upper()}")
        if ledger.final_summary:
            print(ledger.final_summary)
        if ledger.status not in ("verified",):
            unmet = [c.text for c in ledger.unmet()]
            if unmet:
                print("Not satisfied: " + "; ".join(unmet))
            print(f"Resume with:  autopc resume {ledger.id}")
        print("=" * 64)
