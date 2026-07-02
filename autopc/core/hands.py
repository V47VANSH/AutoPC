"""
The hands: mouse and keyboard input through the Interception driver.

Interception injects input at the driver level, below SendInput, so it reaches
windows that ignore or block synthetic input (elevated windows, remote sessions,
some games and canvas apps). Pointer moves follow a short Bezier curve from
`core/cursor_path.py` rather than jumping straight to the target.
"""
import ctypes
import logging
import math
import random
import time

# Imported before `interception` on purpose: interception derives its absolute
# coordinate scale from GetSystemMetrics, which reports virtualized (wrong)
# pixels until the process declares DPI awareness. See core/display.py.
from autopc.core.display import get_geometry
# Pause/resume/stop gate. gate() is a no-op (microseconds) while running, blocks
# while paused, and raises Aborted when stopped. See core/control.py.
from autopc.core.control import CONTROL
from autopc.core.cursor_path import generate_path
from autopc import config

try:
    import interception
except ImportError:
    interception = None
    logging.warning("interception-python not installed. Run 'pip install interception-python'")

log = logging.getLogger(__name__)


class InterceptionHands:
    """Turns the executor's physical action dicts into driver-level input."""

    def __init__(self, screen_size: tuple = None):
        # Default to the real, DPI-aware geometry rather than a hardcoded guess.
        self.geometry = get_geometry()
        if screen_size is None:
            screen_size = self.geometry.primary_size
        self.screen_w, self.screen_h = screen_size

        if (self.screen_w, self.screen_h) != self.geometry.primary_size:
            log.warning(
                "screen_size %s does not match the detected display %s. "
                "Mouse coordinates will be off by a factor of %.3f.",
                screen_size, self.geometry.primary_size,
                self.geometry.primary_width / max(self.screen_w, 1),
            )

        self._current_pos = (self.screen_w // 2, self.screen_h // 2)

        if interception is None:
            raise ImportError("interception-python is required for the hands module.")

        # Bind to the virtual devices
        try:
            interception.auto_capture_devices()
            log.info("Interception driver bound to virtual keyboard and mouse.")
        except Exception as e:
            log.error(f"Failed to capture Interception devices. Is the driver installed? {e}")
            raise

        # Interception injects RELATIVE HID movement (a low-level HID input).
        # Windows "Enhance pointer precision" then applies a non-linear
        # transform to those deltas, so longer bezier segments overshoot and land
        # tens of px off -- measured mean 10px / max 52px with it on, vs ~1px with
        # it off. That is the "clicking sometimes misses" failure. Disable it for
        # the session and restore on exit. It is a local pointer setting that
        # does not affect applications, and it is not persisted, so a crash
        # leaves it off only until the next sign-in. Opt out with
        # AUTOPC_DISABLE_MOUSE_ACCEL=0 to keep your normal mouse feel.
        self._saved_mouse_accel = None
        if config.env_flag("AUTOPC_DISABLE_MOUSE_ACCEL", default=True):
            self._disable_mouse_acceleration()

    # ------------------------------------------------------------------
    # Mouse acceleration (Enhance pointer precision) - off during a session
    # ------------------------------------------------------------------

    _SPI_GETMOUSE = 0x0003
    _SPI_SETMOUSE = 0x0004
    _SPIF_SENDCHANGE = 0x0002

    def _disable_mouse_acceleration(self) -> None:
        """Turn off Enhance pointer precision so relative moves land 1:1."""
        try:
            cur = (ctypes.c_int * 3)()
            if not ctypes.windll.user32.SystemParametersInfoA(
                self._SPI_GETMOUSE, 0, cur, 0
            ):
                return
            if cur[2] == 0:
                return  # already off - nothing to change or restore
            self._saved_mouse_accel = (cur[0], cur[1], cur[2])
            off = (ctypes.c_int * 3)(cur[0], cur[1], 0)
            ctypes.windll.user32.SystemParametersInfoA(
                self._SPI_SETMOUSE, 0, off, self._SPIF_SENDCHANGE
            )
            import atexit
            atexit.register(self._restore_mouse_acceleration)
            log.info("Disabled 'Enhance pointer precision' for click accuracy "
                     "(was %s); will restore on exit.", self._saved_mouse_accel)
        except Exception as exc:
            log.warning("Could not adjust mouse acceleration: %s", exc)

    def _restore_mouse_acceleration(self) -> None:
        """Put the user's Enhance-pointer-precision setting back."""
        if self._saved_mouse_accel is None:
            return
        try:
            saved = (ctypes.c_int * 3)(*self._saved_mouse_accel)
            ctypes.windll.user32.SystemParametersInfoA(
                self._SPI_SETMOUSE, 0, saved, self._SPIF_SENDCHANGE
            )
            log.info("Restored mouse acceleration to %s.", self._saved_mouse_accel)
        except Exception as exc:
            log.warning("Could not restore mouse acceleration: %s", exc)
        finally:
            self._saved_mouse_accel = None

    def execute_action(self, action: dict):
        """
        Carry out one physical action from the executor.

        Pointer: click_coord, double_click_coord, right_click_coord, drag_coord.
        Keyboard and wheel: type, hotkey, scroll.
        """
        action_type = action.get("type")

        # Don't begin a new physical action while paused; bail if stopped.
        CONTROL.gate()

        if action_type == "click_coord":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            button = action.get("button", "left")
            self.move_to(x, y)
            time.sleep(0.05)
            self.click(button)

        elif action_type == "double_click_coord":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            self.move_to(x, y)
            time.sleep(0.05)
            self.click("left")
            time.sleep(0.1)
            self.click("left")

        elif action_type == "right_click_coord":
            x, y = int(action.get("x", 0)), int(action.get("y", 0))
            self.move_to(x, y)
            time.sleep(0.05)
            self.click("right")

        elif action_type == "drag_coord":
            start_x = int(action.get("start_x", 0))
            start_y = int(action.get("start_y", 0))
            end_x   = int(action.get("end_x", 0))
            end_y   = int(action.get("end_y", 0))
            self.drag(start_x, start_y, end_x, end_y)

        elif action_type == "type":
            text = action.get("text", "")
            if not text:
                raise ValueError("Missing 'text' field for type action.")
            self.type_text(text, speed=action.get("speed", "normal"))

        elif action_type == "hotkey":
            keys = action.get("keys", [])
            if not keys:
                raise ValueError("Missing 'keys' field for hotkey action.")
            self.press_hotkey(keys)

        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 3)
            self.scroll(direction, amount)

        else:
            raise ValueError(f"Unknown physical action type: {action_type!r}")

    # ------------------------------------------------------------------
    # Physical Implementations
    # ------------------------------------------------------------------

    def move_to(self, x: int, y: int):
        """Move cursor to (x, y) using a Bezier curve."""
        # Off-screen coordinates are silently clamped by the driver, which turns
        # a model hallucination into a click at the screen edge that looks like
        # a legitimate action. Surface it instead.
        if not self.geometry.contains(int(x), int(y)):
            clamped = self.geometry.clamp(x, y)
            log.warning(
                "Target (%s, %s) is outside the %dx%d screen. Clamped to %s.",
                x, y, self.screen_w, self.screen_h, clamped,
            )
            x, y = clamped

        target = (int(x), int(y))
        path = generate_path(self._current_pos, target)

        # Base reaction time plus travel time proportional to distance, with a
        # little variance. Apps that open menus or tooltips on hover need the
        # pointer to arrive over a few frames, not in one jump.
        dist = math.hypot(x - self._current_pos[0], y - self._current_pos[1])
        screen_diag = math.hypot(self.screen_w, self.screen_h)
        duration = 0.2 + (dist / screen_diag) * 0.4 + random.uniform(-0.05, 0.1)
        duration = max(0.2, duration)

        # Windows time.sleep() resolves to ~15ms, so a long path would take
        # seconds. Sample it down to ~15ms per step, keeping the exact endpoint.
        max_steps = max(5, min(int(duration / 0.015), len(path)))
        if len(path) > max_steps:
            step_size = len(path) / max_steps
            sampled = [path[int(i * step_size)] for i in range(max_steps)]
            sampled[-1] = path[-1]
            path = sampled

        step_delay = duration / len(path)
        for point in path:
            CONTROL.gate()   # freeze mid-travel on pause; bail on stop
            px, py = int(point[0]), int(point[1])
            interception.move_to(px, py)
            self._current_pos = (px, py)
            time.sleep(step_delay)

    def click(self, button: str = "left"):
        """Perform a mouse click."""
        interception.click(button=button)
        
    #: Per-keystroke delay ranges, in seconds.
    #:
    #: "normal" is the touch-typist cadence used for everything a person composes
    #: character by character. "fast" is the burst you get from a held key or a
    #: Tab press -- reserved for runs of leading whitespace, which nobody types
    #: one space at a time. Using it there is both quicker and *more* faithful
    #: to how a person indents than spelling the run out at prose speed.
    KEY_DELAYS = {"normal": (0.02, 0.07), "fast": (0.006, 0.018)}

    def type_text(self, text: str, speed: str = "normal"):
        """
        Type a string of text with slight human variation.

        Line endings are normalized to bare "\\n" first, which the driver maps to
        VK_RETURN -- a real Enter, the same stroke `press_hotkey(["enter"])`
        sends. A raw "\\r" must never reach `interception.write`: it is listed as
        a keyboard key but never assigned a virtual-key code, so it keeps the
        library's -1 "unmapped" sentinel, and `divmod(-1, 0x100)` hands back a
        modifier byte with every bit set. The result is a phantom
        Ctrl+Alt+Shift chord wrapped around a scan-code-0 no-op -- twice per
        character, since both the key-down and the key-up re-send the modifiers.
        Nothing a person types produces that, and Alt+Shift alone is Windows'
        switch-input-language hotkey.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lo, hi = self.KEY_DELAYS.get(speed or "normal", self.KEY_DELAYS["normal"])
        for char in text:
            CONTROL.gate()   # freeze mid-word on pause; bail on stop
            interception.write(char)
            time.sleep(random.uniform(lo, hi))
            
    #: Key names the driver recognises but never assigned a virtual-key code to.
    #: They keep interception's -1 "unmapped" sentinel, which decodes into a
    #: phantom Ctrl+Alt+Shift chord instead of raising (see `type_text`). The
    #: Mac spellings matter in practice: a model that emits `command+s` would
    #: otherwise silently send that chord and then a bare `s`.
    _UNMAPPED_KEYS = {"\r", "fn", "yen", "command", "option",
                      "optionleft", "optionright"}

    #: What those Mac spellings should mean on Windows.
    _KEY_ALIASES = {"command": "ctrl", "cmd": "ctrl", "option": "alt",
                    "optionleft": "alt", "optionright": "alt"}

    def press_hotkey(self, keys: list):
        """
        Press a combination of keys (e.g., ['ctrl', 'c']).

        Unmapped names are rejected rather than injected: the driver would turn
        them into a Ctrl+Alt+Shift chord that no human keyboard can produce.
        Mac spellings are translated to their Windows equivalents first.
        """
        # Strip spaces/tabs only -- a literal "\r" is one of the names we must
        # reject, and .strip() would erase it to "" and slip it past the guard.
        keys = [str(k).lower().strip(" \t") for k in keys]
        keys = [self._KEY_ALIASES.get(k, k) for k in keys]
        bad = [k for k in keys if k in self._UNMAPPED_KEYS]
        if bad:
            raise ValueError(
                f"unsupported key(s) {bad}: the driver has no virtual-key code "
                f"for them and would emit a phantom Ctrl+Alt+Shift chord."
            )
        # Hold all keys down
        for key in keys:
            interception.key_down(key)
            time.sleep(0.02)
        # Release all keys in reverse order
        for key in reversed(keys):
            interception.key_up(key)
            time.sleep(0.02)
            
    def scroll(self, direction: str, amount: int):
        """
        Scroll the wheel `amount` notches. Vertical (`up`/`down`) and horizontal
        (`left`/`right`); horizontal drives the tilt wheel for wide content --
        code that runs off the right edge, big tables, timelines.

        NOTE: `interception.scroll()` takes the string `"up"`/`"down"`, not a
        signed magnitude. An earlier version of this code passed `120`/`-120`,
        which never equals `"up"`, so every scroll silently went DOWN -- that was
        the "can't scroll up" bug. Pass the direction string through instead.
        """
        direction = (direction or "down").lower()
        for _ in range(max(1, amount)):
            CONTROL.gate()   # pause/stop aware between notches
            if direction in ("up", "down"):
                interception.scroll(direction)
            elif direction in ("left", "right"):
                self._scroll_horizontal(direction)
            else:
                interception.scroll("down")
            time.sleep(0.1)

    def _scroll_horizontal(self, direction: str):
        """
        Send a horizontal (tilt) wheel event. `interception-python` exposes only
        vertical `scroll()`, so we build the same stroke it uses but with the
        MOUSE_HWHEEL flag: +120 tilts right, -120 tilts left. Uses the library's
        already-captured mouse context so it rides the same device as every other
        input. Degrades to a no-op (logged) if the library internals move.
        """
        try:
            import interception.inputs as _inp
            from interception import MouseStroke
            from interception.constants import (
                MouseButtonFlag, MouseFlag, MouseRolling,
            )
            data = (MouseRolling.MOUSE_WHEEL_UP if direction == "right"
                    else MouseRolling.MOUSE_WHEEL_DOWN)
            stroke = MouseStroke(
                MouseFlag.MOUSE_MOVE_RELATIVE, MouseButtonFlag.MOUSE_HWHEEL, data, 0, 0,
            )
            _inp._g_context.send(_inp._g_context.mouse, stroke)
            time.sleep(0.025)
        except Exception as exc:
            log.warning("horizontal scroll unavailable (%s); skipped.", exc)
            
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Press at start, move to end with the button held, release.

        The release is in a `finally`: a stop mid-drag raises `Aborted` from the
        move, and without it the left button would stay held down system-wide.
        """
        self.move_to(start_x, start_y)
        time.sleep(0.1)
        interception.mouse_down("left")
        try:
            time.sleep(0.1)
            self.move_to(end_x, end_y)
            time.sleep(0.1)
        finally:
            interception.mouse_up("left")
