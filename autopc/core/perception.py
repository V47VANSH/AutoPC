"""
Perception -- what the agent sees, plus the window context around it.

`observe()` captures one frame and returns an `Observation`: the screenshot the
model is shown, the small controls detected in it (used to snap clicks onto
icons, see `core/vision_grounding.py`), and plain Win32 window information.

## Vision only, by design

This engine does not read UI Automation or any accessibility tree. The agent
should behave identically in every application, including elevated processes
that withhold an accessibility tree from an unelevated client, plus games,
canvas apps and remote sessions that never had one. A perception path that
changes shape depending on how cooperative the app is, is a perception path
that fails in exactly the environments that matter most.

Window queries here (`WindowFromPoint`, `GetForegroundWindow`) are plain
Win32, not accessibility APIs, so the occlusion and focus checks work against
every window.
"""
from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from autopc.core.display import ensure_dpi_aware, get_geometry
from autopc.core.vision_grounding import Box, VisionCandidate, VisionGrounder
from autopc.core import ocr as _ocr

ensure_dpi_aware()

log = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32


@dataclass
class WindowInfo:
    """Plain Win32 view of a top-level window. No accessibility APIs involved."""

    hwnd: int
    title: str
    class_name: str
    process_name: str
    rect: Box
    is_foreground: bool = False

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    def describe(self) -> str:
        tag = " [FOREGROUND]" if self.is_foreground else ""
        return f'"{self.title[:60]}" ({self.process_name}) at {self.rect}{tag}'


@dataclass
class Observation:
    """One look at the screen: the frame the model is shown and acts against."""

    screenshot: object                       # PIL.Image, full screen
    candidates: List[VisionCandidate] = field(default_factory=list)
    foreground: Optional[WindowInfo] = None
    duration: float = 0.0

    def candidate(self, cid: int) -> Optional[VisionCandidate]:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None


class PerceptionEngine:
    """
    Vision-only perception with Win32 window context.

    Holds no COM state and makes no accessibility calls, so it is safe to use
    from any thread and behaves the same against every application.
    """

    def __init__(
        self,
        screenshot_fn,
        max_candidates: int = 90,
        ocr: bool = True,
    ):
        self.screenshot_fn = screenshot_fn
        self.geometry = get_geometry()
        self._grounder = VisionGrounder(max_candidates=max_candidates)

        # OCR gives the model every line of on-screen text with the exact
        # coordinate that clicks it, and labels detected controls by name. It
        # is disabled automatically if winsdk (Windows-only) is absent.
        self._ocr = _ocr.get_default_engine() if ocr else None
        if ocr and not (self._ocr and self._ocr.available):
            log.warning("OCR unavailable (winsdk missing or no OCR language "
                        "pack) -- the agent will read text from the image only.")
            self._ocr = None
        # One-slot cache: observe() labels controls with OCR, and the observe
        # tool then lists the same frame's text -- one OCR pass serves both.
        self._ocr_cache_key: Optional[Tuple[int, Tuple[int, int]]] = None
        self._ocr_cache_lines: List = []

    def _ocr_lines_for(self, shot) -> List:
        if self._ocr is None:
            return []
        key = (id(shot), shot.size)
        if key == self._ocr_cache_key:
            return self._ocr_cache_lines
        lines = self._ocr.read(shot)
        self._ocr_cache_key = key
        self._ocr_cache_lines = lines
        return lines

    # -- observation --------------------------------------------------------

    def observe(self, screenshot=None) -> Observation:
        """Capture the screen (or use `screenshot`) and detect its controls."""
        import time
        t0 = time.perf_counter()
        shot = screenshot if screenshot is not None else self.screenshot_fn()

        candidates = self._grounder.propose(shot)
        if self._ocr is not None:
            _ocr.label_candidates(candidates, lines=self._ocr_lines_for(shot))

        obs = Observation(
            screenshot=shot,
            candidates=candidates,
            foreground=self.foreground_window(),
            duration=time.perf_counter() - t0,
        )
        return obs

    # -- Win32 window context (no accessibility APIs) -----------------------

    @staticmethod
    def _title(hwnd: int) -> str:
        n = _user32.GetWindowTextLengthW(hwnd)
        if not n:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        _user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    @staticmethod
    def _class(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _rect(hwnd: int) -> Box:
        r = wintypes.RECT()
        _user32.GetWindowRect(hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right, r.bottom)

    @staticmethod
    def _process(hwnd: int) -> str:
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = _kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(512)
            size = ctypes.c_ulong(512)
            if _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1]
        finally:
            _kernel32.CloseHandle(handle)
        return ""

    def _window_info(self, hwnd: int, foreground_hwnd: int = 0) -> WindowInfo:
        return WindowInfo(
            hwnd=hwnd,
            title=self._title(hwnd),
            class_name=self._class(hwnd),
            process_name=self._process(hwnd),
            rect=self._rect(hwnd),
            is_foreground=(hwnd == foreground_hwnd),
        )

    def foreground_window(self) -> Optional[WindowInfo]:
        hwnd = _user32.GetForegroundWindow()
        if not hwnd:
            return None
        return self._window_info(hwnd, hwnd)

    # -- occlusion / safety checks -----------------------------------------

    def window_at(self, x: int, y: int) -> Optional[WindowInfo]:
        """
        Which top-level window would receive a click at this point?

        Pure Win32 hit-testing. Works against elevated and hardened processes
        that would refuse an accessibility query, which is why occlusion
        detection survives the vision-only constraint.
        """
        try:
            pt = wintypes.POINT(int(x), int(y))
            hwnd = _user32.WindowFromPoint(pt)
            if not hwnd:
                return None
            # Walk up to the top-level owner so we name the window, not a child control.
            GA_ROOT = 2
            root = _user32.GetAncestor(hwnd, GA_ROOT) or hwnd
            return self._window_info(root, _user32.GetForegroundWindow())
        except Exception as exc:
            log.debug("WindowFromPoint(%s,%s) failed: %s", x, y, exc)
            return None

    # Windows Shell renders "one" UI surface -- the Start menu, its
    # integrated search, the taskbar -- across several cooperating top-level
    # windows in separate processes, and GetForegroundWindow() does not track
    # them as a unit. Observed live (2026-07-20): pressing Win opens
    # StartMenuExperienceHost.exe; the instant the user types, keyboard focus
    # (and GetForegroundWindow) moves to SearchHost.exe ("Search") while the
    # pinned-apps grid stays on screen, still rendered by
    # StartMenuExperienceHost.exe underneath. A strict hit-must-equal-
    # foreground check rejected every click on that grid as "occluded" --
    # 4 for 4 false positives in one run, with the agent's mouse route fully
    # blocked as a result. Both windows are trusted OS shell components, so
    # treating them as one interactive surface when either currently owns the
    # foreground is safe and directly fixes the observed failure.
    _SHELL_PROCESSES = frozenset({
        "StartMenuExperienceHost.exe", "SearchHost.exe", "ShellExperienceHost.exe",
        "explorer.exe", "TextInputHost.exe",
    })

    def check_clickable(self, x: int, y: int) -> Tuple[bool, str]:
        """
        Would a click at (x, y) reach the active window?

        Called before every click. Catches the common and otherwise invisible
        failure where a dialog, tooltip, notification toast or another
        application is sitting on top of the intended target -- the agent
        clicks, something unrelated happens, and the trace shows a "successful"
        click on the wrong thing.

        Returns (ok, reason).
        """
        if not self.geometry.contains(int(x), int(y)):
            return False, f"({x},{y}) is outside the {self.geometry.primary_size} screen."

        fg = _user32.GetForegroundWindow()
        hit = self.window_at(x, y)
        if hit is None:
            return True, "no window resolved; proceeding"

        if hit.hwnd == fg:
            return True, "point belongs to the active window"

        # Popups, menus and combo dropdowns legitimately sit above their owner.
        transient = {"#32768", "tooltips_class32", "ComboLBox", "DV2ControlHost"}
        if hit.class_name in transient:
            return True, f"point is on a transient popup ({hit.class_name})"

        if hit.process_name in self._SHELL_PROCESSES:
            fg_process = self._process(fg) if fg else ""
            if fg_process in self._SHELL_PROCESSES:
                return True, (
                    f"shell surface ({hit.process_name}) cooperating with "
                    f"foreground shell process ({fg_process})"
                )

        return False, (
            f"occluded: ({x},{y}) belongs to \"{hit.title[:40]}\" "
            f"({hit.process_name}), not the active window "
            f"\"{(self._title(fg) or '')[:40]}\""
        )
