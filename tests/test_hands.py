"""A stop in the middle of a drag must still release the mouse button. The
Interception driver is replaced with a recorder."""
import pytest

import autopc.core.hands as hands_mod
from autopc.core.control import CONTROL, Aborted


class _Geometry:
    def contains(self, x, y):
        return True


class _Driver:
    def __init__(self):
        self.calls = []

    def move_to(self, x, y):
        self.calls.append("move")

    def mouse_down(self, button):
        self.calls.append("down")
        CONTROL.abort()            # the user hits stop right after the press

    def mouse_up(self, button):
        self.calls.append("up")


def _hands():
    h = hands_mod.InterceptionHands.__new__(hands_mod.InterceptionHands)
    h.geometry = _Geometry()
    h.screen_w, h.screen_h = 1920, 1080
    h._current_pos = (0, 0)
    return h


def test_drag_releases_button_when_stopped(monkeypatch):
    driver = _Driver()
    monkeypatch.setattr(hands_mod, "interception", driver)
    CONTROL.reset()
    try:
        with pytest.raises(Aborted):
            _hands().drag(10, 10, 400, 300)
        assert driver.calls[-1] == "up"
        assert driver.calls.count("down") == driver.calls.count("up") == 1
    finally:
        CONTROL.reset()


def test_unknown_action_is_an_error(monkeypatch):
    monkeypatch.setattr(hands_mod, "interception", _Driver())
    CONTROL.reset()
    with pytest.raises(ValueError):
        _hands().execute_action({"type": "click"})
