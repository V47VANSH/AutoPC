"""
Display / Coordinate-Space Authority.

This module owns ONE invariant, and the whole agent depends on it:

    the pixel coordinates the VLM reads off the screenshot
    ==
    the pixel coordinates the input driver clicks at

If that invariant breaks, every click silently lands in the wrong place and
the agent looks "dumb" when it is actually being lied to about the screen.

## Why this module exists

Windows reports a *virtualized* screen size to processes that have not declared
DPI awareness. On a 1920x1080 display at 125% scaling:

    DPI-unaware process  ->  GetSystemMetrics(SM_CXSCREEN) == 1536
    the actual desktop   ->  1920 physical pixels
    dxcam (DXGI)         ->  captures 1920 physical pixels, ALWAYS

`interception.move_to()` converts to the driver's 16-bit absolute space using
`0xFFFF / GetSystemMetrics(...)` (see interception/_utils.py). With a
DPI-unaware process that divisor is 1536 instead of 1920, so every click lands
at 1.25x the intended coordinate:

    click(800, 500)   ->  lands at (1000, 625)     [200px, 125px off]
    click(1500, 1040) ->  clamped to screen edge   [unreachable]

Anything past x=1536 / y=864 is unreachable entirely -- which is exactly where
the taskbar, window close buttons and overflow (three-dot) menus live.

Declaring per-monitor DPI awareness makes GetSystemMetrics return true physical
pixels, so the capture space and the input space agree and the bug disappears.

## Usage

Call `ensure_dpi_aware()` as the very first thing in the process, before any
window is created and before dxcam is initialised. `main.py` does this at
import time. Then call `assert_coordinate_spaces_agree(frame)` once at startup
so a future regression fails loudly instead of silently misclicking.
"""
from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# DPI_AWARENESS_CONTEXT values (windef.h). Passed as HANDLEs, hence the casts.
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
_PROCESS_PER_MONITOR_DPI_AWARE = 2

# GetSystemMetrics indices (winuser.h)
_SM_CXSCREEN, _SM_CYSCREEN = 0, 1
_SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
_SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79

_awareness_mode: Optional[str] = None


@dataclass(frozen=True)
class DisplayGeometry:
    """The authoritative screen geometry, in true physical pixels."""

    primary_width: int
    primary_height: int
    virtual_left: int
    virtual_top: int
    virtual_width: int
    virtual_height: int

    @property
    def primary_size(self) -> Tuple[int, int]:
        return self.primary_width, self.primary_height

    @property
    def is_multi_monitor(self) -> bool:
        return (
            self.virtual_width != self.primary_width
            or self.virtual_height != self.primary_height
        )

    def contains(self, x: int, y: int) -> bool:
        """True if (x, y) falls on the primary monitor's coordinate space."""
        return 0 <= x < self.primary_width and 0 <= y < self.primary_height

    def clamp(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp a point into the primary monitor, staying 1px inside the edge."""
        return (
            max(0, min(int(x), self.primary_width - 1)),
            max(0, min(int(y), self.primary_height - 1)),
        )


class CoordinateSpaceMismatch(RuntimeError):
    """Capture resolution and input resolution disagree -- clicks would miss."""


def ensure_dpi_aware() -> str:
    """
    Declare this process DPI-aware so Windows reports true physical pixels.

    Idempotent and safe to call repeatedly: once awareness is set, Windows
    returns E_ACCESSDENIED for subsequent calls, which we treat as success.

    Tries the three API generations newest-first, because the newer contexts
    handle per-monitor scaling changes that the legacy call does not:
      1. SetProcessDpiAwarenessContext (Win10 1703+)  -- per-monitor v2
      2. SetProcessDpiAwareness        (Win8.1+)      -- per-monitor v1
      3. SetProcessDPIAware            (Vista+)       -- system-wide

    Returns the mode that took effect, for logging.
    """
    global _awareness_mode
    if _awareness_mode is not None:
        return _awareness_mode

    # 1. Per-monitor v2 -- the modern, correct option.
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        if user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        ):
            _awareness_mode = "per-monitor-v2"
            return _awareness_mode
    except (AttributeError, OSError):
        pass  # Older than Win10 1703, or the symbol is missing.

    # 2. Per-monitor v1.
    try:
        # S_OK == 0; E_ACCESSDENIED (0x80070005) means it was already set,
        # which is exactly the state we want, so accept both.
        hr = ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)
        if hr in (0, 0x80070005):
            _awareness_mode = "per-monitor-v1"
            return _awareness_mode
    except (AttributeError, OSError):
        pass  # Older than Win8.1.

    # 3. Legacy system-wide. Good enough for a single-monitor setup.
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            _awareness_mode = "system"
            return _awareness_mode
    except (AttributeError, OSError):
        pass

    _awareness_mode = "unaware"
    log.error(
        "Could not declare DPI awareness. If this display uses scaling other "
        "than 100%%, every click will land at the wrong coordinates."
    )
    return _awareness_mode


def get_geometry() -> DisplayGeometry:
    """
    Read the current display geometry in physical pixels.

    Implies `ensure_dpi_aware()`: reading metrics before declaring awareness
    would cache virtualized values, so we never allow that ordering.
    """
    ensure_dpi_aware()
    m = ctypes.windll.user32.GetSystemMetrics
    return DisplayGeometry(
        primary_width=m(_SM_CXSCREEN),
        primary_height=m(_SM_CYSCREEN),
        virtual_left=m(_SM_XVIRTUALSCREEN),
        virtual_top=m(_SM_YVIRTUALSCREEN),
        virtual_width=m(_SM_CXVIRTUALSCREEN),
        virtual_height=m(_SM_CYVIRTUALSCREEN),
    )


def assert_coordinate_spaces_agree(frame, *, strict: bool = True) -> DisplayGeometry:
    """
    Verify that a captured frame matches the input driver's coordinate space.

    This is the guard rail for the entire class of "the agent clicks in the
    wrong place" bugs. Call it once at startup with a real captured frame.

    A mismatch here means the VLM is looking at an image whose pixel grid does
    not correspond to where the mouse will go. There is no safe way to continue,
    so by default we refuse to start rather than misclick thousands of times.

    Args:
        frame: a PIL Image or numpy array from the capture backend.
        strict: raise on mismatch (default). Set False to log and continue.

    Raises:
        CoordinateSpaceMismatch: if capture size != input size and strict.
    """
    geom = get_geometry()

    size = getattr(frame, "size", None)
    if size is not None:  # PIL Image -> (width, height)
        cap_w, cap_h = size
    else:  # numpy array -> (height, width, channels)
        cap_h, cap_w = frame.shape[0], frame.shape[1]

    if (cap_w, cap_h) == (geom.primary_width, geom.primary_height):
        log.info(
            "Coordinate spaces agree: %dx%d (DPI awareness: %s)",
            cap_w, cap_h, _awareness_mode,
        )
        return geom

    scale = cap_w / geom.primary_width if geom.primary_width else float("nan")
    message = (
        f"Coordinate space mismatch -- every click would land in the wrong place.\n"
        f"  screen capture : {cap_w}x{cap_h} (what the model sees)\n"
        f"  input driver   : {geom.primary_width}x{geom.primary_height} (where the mouse goes)\n"
        f"  clicks land at : {scale:.4g}x the intended coordinate\n"
        f"  DPI awareness  : {_awareness_mode}\n"
        f"Likely causes: ensure_dpi_aware() ran after a window/metrics query, "
        f"or the capture backend is grabbing a different monitor than the "
        f"primary one (check dxcam output_idx)."
    )
    if strict:
        raise CoordinateSpaceMismatch(message)
    log.error(message)
    return geom


# Declare awareness at import time. This module is imported before dxcam and
# before any window exists, so the ordering requirement is satisfied by
# construction rather than by remembering to call it.
ensure_dpi_aware()
