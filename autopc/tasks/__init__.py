"""
AutoPC task layer -- what makes the agent reliable over long horizons.

    ledger.py   the task's single source of truth (goal, definition of done,
                plan, progress, attempts), persisted atomically
    store.py    every task on disk; queue order; resumable work

The goal-reinforcement machinery (contract, verifier, guard hooks, runner) is
built on top of these.
"""
from __future__ import annotations

import os

from autopc import config
from autopc.tasks.ledger import Attempt, Criterion, PlanItem, TaskLedger
from autopc.tasks.store import TaskStore

__all__ = ["Attempt", "Criterion", "PlanItem", "TaskLedger", "TaskStore",
           "build_task_store"]


def build_task_store(data_dir: str = None) -> TaskStore:
    data_dir = data_dir or config.data_dir()
    return TaskStore(os.path.join(data_dir, "tasks"))
