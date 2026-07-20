"""
Task store -- every task the agent has been given, on disk.

Each task lives in its own directory under `<data>/tasks/`:

    <data>/tasks/20261018-193045-open-notepad-and-type/
        ledger.json     the structured state (tasks/ledger.py)
        progress.md     a human-readable journal of the same progress

Directory names sort chronologically, so "oldest queued" and "most recent
unfinished" are cheap to answer. Because nothing lives only in memory, a crash
or restart loses nothing: `autopc resume` reloads the ledger and continues.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from autopc.tasks.ledger import UNFINISHED, TaskLedger, _now, slugify


class TaskStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ----------------------------------------------------------------- create
    def create(self, goal: str, status: str = "active") -> TaskLedger:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("A task needs a goal.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{slugify(goal)}"
        task_id, n = base, 1
        while os.path.exists(os.path.join(self.root, task_id)):
            n += 1
            task_id = f"{base}-{n}"
        ledger = TaskLedger(id=task_id, goal=goal, created=_now(), status=status,
                            path=os.path.join(self.root, task_id))
        ledger.save()
        ledger.log(f"Task created ({status}).")
        return ledger

    # ------------------------------------------------------------------- read
    def _dirs(self) -> List[str]:
        try:
            names = sorted(os.listdir(self.root))
        except FileNotFoundError:
            return []
        return [n for n in names
                if os.path.isfile(os.path.join(self.root, n, "ledger.json"))]

    def load(self, task_id: str) -> Optional[TaskLedger]:
        """Load by full id or unique prefix."""
        task_id = (task_id or "").strip()
        if not task_id:
            return None
        names = self._dirs()
        if task_id in names:
            return TaskLedger.load(os.path.join(self.root, task_id))
        matches = [n for n in names if n.startswith(task_id)]
        if len(matches) == 1:
            return TaskLedger.load(os.path.join(self.root, matches[0]))
        return None

    def all(self) -> List[TaskLedger]:
        """Every task, oldest first. Unreadable entries are skipped."""
        out = []
        for n in self._dirs():
            try:
                out.append(TaskLedger.load(os.path.join(self.root, n)))
            except Exception:
                continue
        return out

    def recent(self, limit: int = 20) -> List[TaskLedger]:
        return list(reversed(self.all()))[:limit]

    def unfinished(self) -> List[TaskLedger]:
        return [t for t in self.all() if t.status in UNFINISHED]

    def next_queued(self) -> Optional[TaskLedger]:
        """Oldest queued task -- the queue is first-in, first-out."""
        for t in self.all():
            if t.status == "queued":
                return t
        return None

    def latest_resumable(self) -> Optional[TaskLedger]:
        """Most recent task that was started but not finished (crashed,
        interrupted, or paused). Queued-but-never-started tasks are not
        'resumable' -- `autopc work` picks those up."""
        for t in reversed(self.all()):
            if t.status in ("active", "paused"):
                return t
        return None
