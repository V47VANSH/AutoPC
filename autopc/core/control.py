"""
Run control -- pause/resume/stop for an in-flight agent session.

The agent's physical actions (mouse travel, typing, scrolling) run as small
step loops on worker threads. This module gives those loops a single shared
gate they check between steps, so an operator can:

  * **pause** -- freeze the hands mid-motion (part-way through a bezier move or
    mid-word) and later **resume from exactly that step**, and
  * **stop** -- abort the current action promptly so the session can be torn
    down and go idle.

The design is deliberately tiny and thread-safe: two `threading.Event`s and a
`gate()` the hands call. When running, `gate()` returns in microseconds (the
resume event is set, so `wait()` returns immediately), so it adds no measurable
latency to the hundreds of calls a single move makes. When paused, `gate()`
blocks on that event until resumed. When stopped, `gate()` raises `Aborted`,
which unwinds the physical action.

`CONTROL` is the process-wide instance. `core/hands.py` calls `CONTROL.gate()`;
a controlling caller invokes `pause()/resume()/toggle()/abort()/reset()` from
its own thread.
"""
from __future__ import annotations

import threading


class Aborted(Exception):
    """Raised inside a gated physical action when the run is stopped."""


class RunControl:
    """A pause/resume/stop gate shared between the controller and the hands."""

    def __init__(self) -> None:
        self._resume = threading.Event()
        self._resume.set()          # set == running; cleared == paused
        self._abort = threading.Event()

    # -- state ---------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # -- operator side (called from the controller thread) -------------------

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def toggle(self) -> bool:
        """Flip paused/running. Returns True if now paused."""
        if self.paused:
            self.resume()
            return False
        self.pause()
        return True

    def abort(self) -> None:
        """Stop the current run. Also releases the resume event so any thread
        blocked in `gate()` wakes up and raises `Aborted` instead of hanging."""
        self._abort.set()
        self._resume.set()

    def reset(self) -> None:
        """Clear stop/pause so the next session starts clean."""
        self._abort.clear()
        self._resume.set()

    # -- worker side (called from the hands' step loops) ---------------------

    def gate(self) -> None:
        """
        Checkpoint between the small steps of a physical action.

        Returns immediately while running. Blocks while paused (checking for a
        stop every 100 ms). Raises `Aborted` if the run was stopped -- both when
        already paused and when running -- so a stop bails out of a move or a
        keystroke stream promptly.
        """
        if self._abort.is_set():
            raise Aborted()
        while not self._resume.wait(0.1):
            if self._abort.is_set():
                raise Aborted()
        if self._abort.is_set():
            raise Aborted()


#: Process-wide control shared by the hands and the controlling caller.
CONTROL = RunControl()
