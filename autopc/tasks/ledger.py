"""
Task ledger -- the single source of truth for what the agent is trying to do.

Long-horizon agents fail in three characteristic ways: they *drift* (the goal
erodes as the context fills and gets compacted), they *declare victory early*
(a changed screen is mistaken for a finished task), and they *forget where they
were* (a crash or restart loses all progress). The ledger addresses all three by
keeping the task's state OUTSIDE the conversation, on disk:

    goal        the user's words, verbatim and immutable -- never paraphrased,
                never re-scoped
    criteria    the definition of done: concrete, screen-observable conditions.
                The task is finished when these are met -- not when the agent
                feels finished.
    plan        ordered steps, at most one in progress at a time
    progress    a timestamped journal of checkpoints, so a fresh session (or a
                restarted process) can pick up exactly where work stands
    attempts    each run at the task, how it ended, and what was learned
    verdicts    every independent verification of a completion claim

The conversation is disposable; the ledger is not. Every write is atomic
(temp file + replace), so a crash mid-save can never leave a corrupt ledger.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

#: Task lifecycle. `active`/`paused`/`queued` are unfinished and resumable;
#: `verified`/`needs_user`/`failed` are terminal.
UNFINISHED = ("active", "paused", "queued")
TERMINAL = ("verified", "needs_user", "failed")

PLAN_STATUSES = ("pending", "in_progress", "done", "blocked")
CRITERION_STATUSES = ("pending", "met", "unmet")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Criterion:
    """One condition of the definition of done.

    `status` is the agent's own assessment. It is advisory: a task only reaches
    `verified` when the independent verifier agrees (see tasks/verifier.py).
    """
    id: str
    text: str
    status: str = "pending"
    evidence: str = ""


@dataclass
class PlanItem:
    text: str
    status: str = "pending"


@dataclass
class Attempt:
    """One run at the task, in one fresh agent context."""
    n: int
    started: str
    ended: str = ""
    #: verified | gave_up | needs_user | exhausted | interrupted | error
    outcome: str = ""
    tool_calls: int = 0
    stop_blocks: int = 0
    reflection: str = ""


@dataclass
class TaskLedger:
    id: str
    goal: str
    created: str
    updated: str = ""
    status: str = "active"
    criteria: List[Criterion] = field(default_factory=list)
    plan: List[PlanItem] = field(default_factory=list)
    attempts: List[Attempt] = field(default_factory=list)
    progress: List[Dict[str, Any]] = field(default_factory=list)
    blockers: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    final_summary: str = ""
    skill_used: str = ""
    #: Directory holding ledger.json + progress.md. Not serialized.
    path: str = field(default="", repr=False, compare=False)

    # ------------------------------------------------------------ persistence
    @property
    def ledger_file(self) -> str:
        return os.path.join(self.path, "ledger.json")

    @property
    def journal_file(self) -> str:
        return os.path.join(self.path, "progress.md")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("path", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "") -> "TaskLedger":
        return cls(
            id=data["id"],
            goal=data["goal"],
            created=data.get("created", _now()),
            updated=data.get("updated", ""),
            status=data.get("status", "active"),
            criteria=[Criterion(**c) for c in data.get("criteria", [])],
            plan=[PlanItem(**p) for p in data.get("plan", [])],
            attempts=[Attempt(**a) for a in data.get("attempts", [])],
            progress=list(data.get("progress", [])),
            blockers=list(data.get("blockers", [])),
            verdicts=list(data.get("verdicts", [])),
            lessons=list(data.get("lessons", [])),
            final_summary=data.get("final_summary", ""),
            skill_used=data.get("skill_used", ""),
            path=path,
        )

    @classmethod
    def load(cls, path: str) -> "TaskLedger":
        with open(os.path.join(path, "ledger.json"), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), path=path)

    def save(self) -> None:
        """Atomically persist the ledger (temp file + os.replace)."""
        if not self.path:
            return
        os.makedirs(self.path, exist_ok=True)
        self.updated = _now()
        fd, tmp = tempfile.mkstemp(dir=self.path, prefix=".ledger-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.ledger_file)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _journal(self, line: str) -> None:
        if not self.path:
            return
        try:
            new = not os.path.exists(self.journal_file)
            with open(self.journal_file, "a", encoding="utf-8") as f:
                if new:
                    f.write(f"# Task {self.id}\n\nGoal: {self.goal}\n\n")
                f.write(line.rstrip() + "\n")
        except OSError:
            pass

    # --------------------------------------------------------------- progress
    def log(self, note: str, save: bool = True) -> None:
        """Append a checkpoint to the progress journal."""
        note = (note or "").strip()
        if not note:
            return
        entry = {"ts": _now(), "note": note[:1000]}
        self.progress.append(entry)
        self._journal(f"- [{entry['ts']}] {entry['note']}")
        if save:
            self.save()

    def add_blocker(self, text: str, needs_user: bool = False) -> None:
        self.blockers.append({"ts": _now(), "text": (text or "").strip()[:1000],
                              "needs_user": bool(needs_user),
                              "attempt": self.current_attempt.n if self.current_attempt else 0})
        self.log(f"BLOCKER{' (needs user)' if needs_user else ''}: {text}")

    def has_user_blocker(self) -> bool:
        """A blocker only the user can resolve, recorded in the current attempt."""
        n = self.current_attempt.n if self.current_attempt else 0
        return any(b.get("needs_user") and b.get("attempt", 0) == n for b in self.blockers)

    # ------------------------------------------------------------------- plan
    def set_plan(self, items: List[Dict[str, Any]]) -> None:
        """Replace the plan. Enforces at most one step in progress -- doing one
        thing at a time is what keeps a long task from fragmenting."""
        plan: List[PlanItem] = []
        seen_in_progress = False
        for it in items or []:
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            status = str(it.get("status", "pending")).strip().lower()
            if status not in PLAN_STATUSES:
                status = "pending"
            if status == "in_progress":
                if seen_in_progress:
                    status = "pending"
                seen_in_progress = True
            plan.append(PlanItem(text=text[:300], status=status))
        self.plan = plan
        self.save()

    def current_step(self) -> str:
        for p in self.plan:
            if p.status == "in_progress":
                return p.text
        for p in self.plan:
            if p.status == "pending":
                return p.text
        return ""

    # --------------------------------------------------------------- criteria
    def set_criteria(self, texts: List[str]) -> None:
        self.criteria = [Criterion(id=f"c{i}", text=t.strip()[:400])
                         for i, t in enumerate(texts, 1) if t and t.strip()]
        self.save()

    def mark_criterion(self, cid: str, met: bool, evidence: str = "") -> bool:
        for c in self.criteria:
            if c.id == cid:
                c.status = "met" if met else "unmet"
                c.evidence = (evidence or "").strip()[:600]
                self.save()
                return True
        return False

    def unmet(self) -> List[Criterion]:
        return [c for c in self.criteria if c.status != "met"]

    def all_marked_met(self) -> bool:
        return bool(self.criteria) and not self.unmet()

    # --------------------------------------------------------------- attempts
    @property
    def current_attempt(self) -> Optional[Attempt]:
        if self.attempts and not self.attempts[-1].ended:
            return self.attempts[-1]
        return None

    def begin_attempt(self) -> Attempt:
        a = Attempt(n=len(self.attempts) + 1, started=_now())
        self.attempts.append(a)
        self.status = "active"
        # The agent's per-criterion marks are a fresh judgement each attempt --
        # carrying a stale "met" forward would let an earlier, wrong claim leak in.
        for c in self.criteria:
            if c.status == "met":
                c.status = "pending"
        self.log(f"Attempt {a.n} started.")
        return a

    def end_attempt(self, outcome: str, reflection: str = "") -> None:
        a = self.current_attempt
        if a is None:
            return
        a.ended = _now()
        a.outcome = outcome
        a.reflection = (reflection or "")[:2000]
        self.log(f"Attempt {a.n} ended: {outcome}.")

    # ----------------------------------------------------------------- status
    @property
    def finished(self) -> bool:
        return self.status in TERMINAL

    def finish(self, status: str, summary: str = "") -> None:
        self.status = status
        if summary:
            self.final_summary = summary.strip()[:2000]
        self.log(f"Task {status}." + (f" {summary}" if summary else ""))

    # -------------------------------------------------------------- rendering
    def _criteria_lines(self) -> List[str]:
        marks = {"met": "[x]", "unmet": "[!]", "pending": "[ ]"}
        return [f"  {marks.get(c.status, '[ ]')} {c.id}: {c.text}" for c in self.criteria]

    def anchor(self) -> str:
        """Compact goal anchor, re-injected periodically during a long run."""
        lines = [f"TASK (your only objective): {self.goal}"]
        unmet = self.unmet()
        if self.criteria:
            if unmet:
                lines.append("Still to satisfy: " + "; ".join(f"{c.id} {c.text}" for c in unmet))
            else:
                lines.append("All criteria marked met -- verify on screen, then call "
                             "complete_task(success=true).")
        step = self.current_step()
        if step:
            lines.append(f"Current step: {step}")
        return "\n".join(lines)

    def report(self, recent: int = 8) -> str:
        """Full status, for the task_status tool and for resuming."""
        out = [f"TASK {self.id}  [{self.status}]", f"Goal: {self.goal}"]
        if self.criteria:
            out += ["", "Definition of done:"] + self._criteria_lines()
        if self.plan:
            marks = {"done": "[x]", "in_progress": "[>]", "blocked": "[!]", "pending": "[ ]"}
            out += ["", "Plan:"] + [f"  {marks.get(p.status, '[ ]')} {p.text}" for p in self.plan]
        if self.progress:
            out += ["", f"Recent progress (last {min(recent, len(self.progress))}):"]
            out += [f"  - {p['note']}" for p in self.progress[-recent:]]
        if self.blockers:
            out += ["", "Blockers:"] + [f"  - {b['text']}" + (" (needs user)" if b.get("needs_user") else "")
                                        for b in self.blockers[-4:]]
        if self.lessons:
            out += ["", "Lessons from earlier attempts:"] + [f"  - {l}" for l in self.lessons[-8:]]
        if self.verdicts:
            v = self.verdicts[-1]
            out += ["", f"Last verification: {v.get('overall')} -- {v.get('missing', '')}"]
        return "\n".join(out)


_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    s = _SLUG.sub("-", (text or "").lower()).strip("-")
    return (s[:max_len].rstrip("-")) or "task"
