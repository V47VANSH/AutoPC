"""
Executor -- turns a tool call into a checked physical input.

Sits between the agent's tools and the Interception driver, and owns the three
things that decide whether a click actually works:

  1. **Targeting.** The model points at a pixel it read off the screenshot. If
     that point is inside a small detected control (an icon, a checkbox), the
     click is snapped to the control's true centre; anywhere else it lands
     exactly where the model pointed.

  2. **Pre-flight occlusion check.** Before the pointer moves, we ask Win32
     which window owns the target point. If another window is covering it, we
     refuse and say so. This is the single cheapest guard against the failure
     that looks like a mis-click but is really "a dialog was on top" -- the
     agent clicks, something unrelated happens, and the trace shows a
     successful click on the wrong thing.

  3. **Post-flight observation.** We report what changed, so the caller can
     tell a click that did something from a click that did nothing.

Actuation stays with the Interception kernel driver moving a real cursor along
a bezier curve. Nothing here synthesises input through the accessibility layer
or SendInput -- physical input is the input model this project is built around.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from autopc.core.perception import Observation, PerceptionEngine

log = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """The action could not be carried out. Message is fed back to the model."""


@dataclass
class ExecutionResult:
    """What happened, in enough detail for the agent to judge it."""

    ok: bool
    action: dict
    detail: str = ""
    target: Optional[Tuple[int, int]] = None
    target_label: str = ""
    occluded_by: str = ""
    screen_changed: Optional[bool] = None
    change_ratio: float = 0.0
    window_before: str = ""
    window_after: str = ""
    duration: float = 0.0

    def summary(self) -> str:
        bits = [f"{self.action.get('type')}"]
        if self.target:
            bits.append(f"at {self.target}")
        if self.target_label:
            bits.append(f"({self.target_label})")
        bits.append("OK" if self.ok else "FAILED")
        if self.detail:
            bits.append(f"-- {self.detail}")
        if self.screen_changed is not None:
            bits.append(
                f"[screen {'changed' if self.screen_changed else 'UNCHANGED'} "
                f"{self.change_ratio:.1%}]"
            )
        if self.window_before != self.window_after:
            bits.append(f"[window: '{self.window_before[:28]}' -> '{self.window_after[:28]}']")
        return " ".join(bits)


class Executor:
    """Resolves targets, checks them, and drives the physical input layer."""

    # Below this fraction of changed pixels we treat the screen as static.
    # Tuned above cursor-blink and clock-tick noise, below any real UI response.
    CHANGE_EPSILON = 0.0015

    def __init__(
        self,
        perception: PerceptionEngine,
        hands,
        settle_time: float = 0.35,
        verify: bool = True,
    ):
        self.perception = perception
        self.hands = hands
        self.settle_time = settle_time
        self.verify = verify

    # -- main entry ---------------------------------------------------------

    def execute(self, action: dict, observation: Observation) -> ExecutionResult:
        """
        Run one action against the observation the model was shown.

        `observation` must be the frame the model looked at: snapping uses its
        detected controls, and its screenshot is the "before" for change
        detection.
        """
        started = time.perf_counter()

        before_img = observation.screenshot if self.verify else None
        fg = self.perception.foreground_window()
        window_before = fg.title if fg else ""

        try:
            target, label = self._resolve_target(action, observation)
            if target is not None:
                ok, reason = self.perception.check_clickable(*target)
                if not ok:
                    return ExecutionResult(
                        ok=False, action=action, detail=reason,
                        target=target, target_label=label,
                        occluded_by=reason, window_before=window_before,
                        window_after=window_before,
                        duration=time.perf_counter() - started,
                    )

            detail = self._dispatch(action, observation)

        except ExecutionError as exc:
            return ExecutionResult(
                ok=False, action=action, detail=str(exc),
                window_before=window_before, window_after=window_before,
                duration=time.perf_counter() - started,
            )
        except Exception as exc:
            log.exception("Unexpected executor failure")
            return ExecutionResult(
                ok=False, action=action,
                detail=f"{type(exc).__name__}: {exc}",
                window_before=window_before, window_after=window_before,
                duration=time.perf_counter() - started,
            )

        time.sleep(self.settle_time)

        result = ExecutionResult(
            ok=True, action=action, detail=detail,
            target=target, target_label=label,
            window_before=window_before,
            duration=time.perf_counter() - started,
        )

        if self.verify and before_img is not None:
            after_img = self.perception.screenshot_fn()
            result.change_ratio = self._change_ratio(before_img, after_img)
            result.screen_changed = result.change_ratio > self.CHANGE_EPSILON
        fg_after = self.perception.foreground_window()
        result.window_after = fg_after.title if fg_after else ""

        return result

    # -- target resolution --------------------------------------------------

    def _resolve_target(
        self, action: dict, obs: Observation
    ) -> Tuple[Optional[Tuple[int, int]], str]:
        """The screen point a pointer action will act on, or None for keyboard
        actions. The point is what the occlusion check runs against."""
        atype = action.get("type", "")

        # The model names a point it read off the screenshot; we snap it to the
        # centre of a small detected control it lands inside, so a slightly off
        # estimate on an icon still lands dead-on.
        if atype == "click_xy":
            x, y = int(action["x"]), int(action["y"])
            label = action.get("label", "")
            if action.get("snap", True):
                (x, y), snap_note = self._snap_to_element(x, y, obs)
                label = f"{label} {snap_note}".strip()
            return (x, y), label

        # A drag is occlusion-checked at its START point -- a drag that begins
        # under another window grabs the wrong thing. The end point is not pre-checked: a drag often
        # releases somewhere that is not a discrete clickable (empty canvas, the
        # far end of a text selection, a slider track position).
        if atype == "drag_xy":
            sx, sy = int(action["start_x"]), int(action["start_y"])
            return (sx, sy), action.get("label", "")

        return None, ""

    #: A control this size or smaller benefits from snapping (a 24px icon the
    #: model can only point at approximately). Anything larger -- a text line, a
    #: button with a label, a panel -- is a big enough target that the model's
    #: own point is already inside it, and snapping to a *centre* would move a
    #: deliberate click (cursor between two code lines, end of a line, a spot
    #: inside a word) to the wrong place. So snap is small-control-only.
    SNAP_MAX_EDGE = 80

    def _snap_to_element(
        self, x: int, y: int, obs: Observation
    ) -> Tuple[Tuple[int, int], str]:
        """
        Pull a click onto the true centre of a *small* control it lands inside.

        The model points where it saw a target; on a small icon its estimate can
        be a few px off and detection geometry is exact, so we snap to the icon's
        centre. Crucially we snap ONLY to compact controls the point is inside --
        never a wide text line, a container, or empty space -- so clicking
        whitespace, between code lines, or a precise spot in text lands exactly
        where the model meant.
        """
        def _is_snappable_icon(c) -> bool:
            w, h = c.rect[2] - c.rect[0], c.rect[3] - c.rect[1]
            if w > self.SNAP_MAX_EDGE or h > self.SNAP_MAX_EDGE:
                return False            # text line, button, container -> leave raw
            ar = w / max(1, h)
            if not (0.6 <= ar <= 1.6):
                return False            # wide/thin -> not an icon, leave raw
            return c.rect[0] <= x <= c.rect[2] and c.rect[1] <= y <= c.rect[3]

        containing = [c for c in obs.candidates if _is_snappable_icon(c)]
        if containing:
            best = min(containing, key=lambda c: c.area)
            lbl = f'"{best.text}"' if best.text else best.hint
            return best.center, f"(snapped to {lbl})"
        return (x, y), ""

    # -- dispatch -----------------------------------------------------------

    def _dispatch(self, action: dict, obs: Observation) -> str:
        atype = action["type"]

        if atype == "click_xy":
            return self._do_pointer(action, obs)

        if atype == "drag_xy":
            return self._do_drag_xy(action)

        if atype == "scroll":
            return self._do_scroll(action)

        if atype == "type":
            self.hands.execute_action({"type": "type", "text": action["text"]})
            return f"typed {len(action['text'])} chars"

        if atype == "type_code":
            from autopc.core.code_typer import indent_is_significant, type_code as _type_code
            language = action.get("language") or None
            # Report which indent strategy was used -- it is the difference
            # between riding the editor's auto-indent and clearing it per line.
            strategy = ("exact" if indent_is_significant(action["text"], language)
                        else "editor auto-indent")
            n = _type_code(
                self.hands, action["text"],
                replace=bool(action.get("replace", True)),
                language=language,
            )
            return (f"typed {n} chars of code (quirk-aware, indent: {strategy}"
                    f"{', replaced editor contents' if action.get('replace', True) else ''})")

        if atype == "hotkey":
            self.hands.execute_action({"type": "hotkey", "keys": action["keys"]})
            return f"pressed {'+'.join(action['keys'])}"

        if atype == "navigate":
            return self._do_navigate(action)

        if atype == "press":
            count = max(1, int(action.get("count", 1)))
            for _ in range(count):
                self.hands.execute_action({"type": "hotkey", "keys": [action["key"]]})
                time.sleep(0.04)
            return f"pressed {action['key']} x{count}"

        raise ExecutionError(f"Executor cannot dispatch action type {atype!r}.")

    def _do_pointer(self, action: dict, obs: Observation) -> str:
        (x, y), label = self._resolve_target(action, obs)  # type: ignore[misc]

        if action.get("hover"):
            self.hands.move_to(x, y)
            return f"hovered {label} at ({x},{y})".strip()
        button = str(action.get("button", "left")).lower()
        if action.get("double"):
            self.hands.execute_action({"type": "double_click_coord", "x": x, "y": y})
            return f"double-clicked {label} at ({x},{y})".strip()
        if button == "right":
            self.hands.execute_action({"type": "right_click_coord", "x": x, "y": y})
        else:
            self.hands.execute_action({"type": "click_coord", "x": x, "y": y, "button": "left"})
        return f"{button}-clicked {label} at ({x},{y})".strip()

    def _do_drag_xy(self, action: dict) -> str:
        """Press at (start), move with the button held, release at (end).

        Used for sliders, scrollbars, reordering, range selection and drawing.
        """
        sx, sy = int(action["start_x"]), int(action["start_y"])
        ex, ey = int(action["end_x"]), int(action["end_y"])
        self.hands.execute_action({
            "type": "drag_coord",
            "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
        })
        return f"dragged ({sx},{sy})->({ex},{ey})"

    def _do_scroll(self, action: dict) -> str:
        """
        Position the pointer over the pane, then scroll.

        Windows routes wheel events to whatever sits under the cursor, so
        scrolling without positioning first frequently scrolls the wrong pane
        -- a failure that looks like "scroll did nothing".
        """
        moved = ""
        if action.get("x") is not None and action.get("y") is not None:
            x, y = int(action["x"]), int(action["y"])
            self.hands.move_to(x, y)
            time.sleep(0.05)
            moved = f" at ({x},{y})"

        self.hands.execute_action({
            "type": "scroll",
            "direction": action.get("direction", "down"),
            "amount": int(action.get("amount", 3)),
        })
        return f"scrolled {action.get('direction', 'down')} x{action.get('amount', 3)}{moved}"

    # Pauses inside the navigate sequence. Small, but not zero: the address bar
    # needs a moment to take focus and to accept the selection before the URL
    # is typed over it, and Windows drops keystrokes sent into a control that
    # is still animating in.
    NAV_FOCUS_SETTLE = 0.35
    NAV_SELECT_SETTLE = 0.12
    NAV_SUBMIT_SETTLE = 0.15
    NAV_LOAD_SETTLE = 1.5

    def _do_navigate(self, action: dict) -> str:
        """
        Address bar -> select all -> type -> enter, with nothing able to interleave.

        The point of doing this here rather than as four tool calls is that a
        tool call costs seconds, and a page that steals focus during that
        gap silently redirects the URL into its own search box. Inside one
        executor call there is no gap to lose.

        `ctrl+a` after `ctrl+L` looks redundant -- focusing the bar already
        selects its contents -- but it is what makes a *stale* address bar
        safe. Without it, a bar that was focused but not selected appends the
        new URL to the old one, which is how a run ends up requesting
        `example.com/page/one-itemhttps://example.com/catalog/`.
        """
        url = action["url"].strip()

        self.hands.execute_action({"type": "hotkey", "keys": ["ctrl", "l"]})
        time.sleep(self.NAV_FOCUS_SETTLE)

        self.hands.execute_action({"type": "hotkey", "keys": ["ctrl", "a"]})
        time.sleep(self.NAV_SELECT_SETTLE)

        self.hands.execute_action({"type": "type", "text": url})
        time.sleep(self.NAV_SUBMIT_SETTLE)

        self.hands.execute_action({"type": "hotkey", "keys": ["enter"]})
        time.sleep(self.NAV_LOAD_SETTLE)

        return f"navigated to {url}"

    # -- change detection ---------------------------------------------------

    @staticmethod
    def _change_ratio(before, after) -> float:
        """
        Fraction of pixels that changed between two frames.

        Deliberately crude and downscaled: we only need to answer "did anything
        happen", and comparing at full resolution would be both slow and
        oversensitive to antialiasing and cursor artefacts.
        """
        try:
            import numpy as np
            small = (240, 135)
            a = np.asarray(before.convert("L").resize(small), dtype=np.int16)
            b = np.asarray(after.convert("L").resize(small), dtype=np.int16)
            return float((np.abs(a - b) > 18).mean())
        except Exception as exc:
            log.debug("Change detection failed: %s", exc)
            return 0.0
