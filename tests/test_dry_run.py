"""Dry run: no input tool may reach the executor or the reader, and live mode
still dispatches. The screen, executor and reader are stubbed."""
import pytest

from autopc.agent_brain import tools
from autopc.core.executor import ExecutionResult


class _Shot:
    width, height = 1920, 1080


class _Obs:
    screenshot = _Shot()
    candidates = []
    foreground = None


class _Perception:
    def _ocr_lines_for(self, shot):
        return []


class _Executor:
    def __init__(self):
        self.calls = []

    def execute(self, action, obs):
        self.calls.append(action)
        return ExecutionResult(ok=True, action=action, detail=f"did {action['type']}",
                               screen_changed=True, change_ratio=0.2)


class _Reader:
    def read_full(self, **kwargs):
        raise AssertionError("read_screen scrolled during a dry run")


INPUT_TOOLS = [
    (tools.click_point, {"x": 10, "y": 20}),
    (tools.click_point, {"x": 10, "y": 20, "button": "right", "double": True}),
    (tools.hover_point, {"x": 10, "y": 20}),
    (tools.drag, {"start_x": 1, "start_y": 2, "end_x": 30, "end_y": 40}),
    (tools.type_text, {"text": "hello"}),
    (tools.type_code, {"code": "print(1)\n", "language": "python"}),
    (tools.press_key, {"keys": "alt+f4"}),
    (tools.press_key, {"keys": "enter"}),
    (tools.navigate, {"url": "https://example.com"}),
    (tools.scroll, {"x": 5, "y": 5, "direction": "down", "amount": 3}),
]


@pytest.fixture
def desk(monkeypatch):
    ex = _Executor()
    monkeypatch.setattr(tools, "_PERCEPTION", _Perception())
    monkeypatch.setattr(tools, "_EXECUTOR", ex)
    monkeypatch.setattr(tools, "_READER", _Reader())
    monkeypatch.setattr(tools, "_LAST", _Obs())
    monkeypatch.setattr(tools, "_LAST_FRAME_SCALE", 1.0)
    return ex


def _txt(result):
    return result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", INPUT_TOOLS, ids=lambda v: getattr(v, "name", ""))
async def test_dry_run_touches_nothing(desk, monkeypatch, tool, args):
    monkeypatch.setattr(tools, "DRY_RUN", True)
    result = await tool.handler(args)
    assert _txt(result).startswith("DRY RUN")
    assert desk.calls == []


@pytest.mark.asyncio
async def test_dry_run_read_screen_does_not_scroll(desk, monkeypatch):
    monkeypatch.setattr(tools, "DRY_RUN", True)
    result = await tools.read_screen.handler({"x": 100, "y": 100})
    assert _txt(result).startswith("DRY RUN")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", INPUT_TOOLS, ids=lambda v: getattr(v, "name", ""))
async def test_live_mode_dispatches(desk, monkeypatch, tool, args):
    monkeypatch.setattr(tools, "DRY_RUN", False)
    result = await tool.handler(args)
    assert len(desk.calls) == 1
    assert "did " in _txt(result)
