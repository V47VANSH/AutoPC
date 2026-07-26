"""Goal-reinforcement hooks, driven with SDK-shaped hook inputs."""
import pytest

from autopc.tasks import TaskStore
from autopc.tasks.guard import TaskGuard
from autopc.tasks.state import AttemptState


@pytest.fixture
def guard(tmp_path):
    L = TaskStore(str(tmp_path)).create("rename report.docx to final.docx")
    L.set_criteria(["Explorer shows final.docx and no report.docx"])
    L.set_plan([{"text": "open Documents in Explorer", "status": "in_progress"}])
    L.begin_attempt()
    return TaskGuard(AttemptState(ledger=L), reanchor_every=4, nudge_every=0,
                     max_stop_blocks=2)


def _post(tool, tool_input=None, text="click OK | screen changed (4.0% of pixels)"):
    return {"hook_event_name": "PostToolUse", "tool_name": f"mcp__desktop__{tool}",
            "tool_input": tool_input or {}, "tool_response": {"content": [{"type": "text", "text": text}]},
            "tool_use_id": "t", "session_id": "s", "transcript_path": "", "cwd": ""}


def _ctx(out):
    return out.get("hookSpecificOutput", {}).get("additionalContext", "")


@pytest.mark.asyncio
async def test_stop_is_blocked_with_the_goal_until_done(guard):
    out = await guard.on_stop({"stop_hook_active": False, "last_assistant_message": "All done!"}, None, {})
    assert out["decision"] == "block"
    assert "rename report.docx to final.docx" in out["reason"]
    assert "final.docx and no report.docx" in out["reason"]


@pytest.mark.asyncio
async def test_stop_allowed_once_verified(guard):
    guard.state.ledger.finish("verified")
    assert await guard.on_stop({"stop_hook_active": False}, None, {}) == {}


@pytest.mark.asyncio
async def test_stop_gate_is_bounded(guard):
    for _ in range(2):
        assert (await guard.on_stop({"stop_hook_active": True}, None, {}))["decision"] == "block"
    assert await guard.on_stop({"stop_hook_active": True}, None, {}) == {}
    assert guard.state.stop_blocks == 2


@pytest.mark.asyncio
async def test_question_to_absent_user_is_answered(guard):
    out = await guard.on_stop({"stop_hook_active": False,
                               "last_assistant_message": "Should I overwrite the file?"}, None, {})
    assert "running autonomously" in out["reason"]


@pytest.mark.asyncio
async def test_user_stop_wins(guard):
    from autopc.core.control import CONTROL
    CONTROL.abort()
    try:
        assert await guard.on_stop({"stop_hook_active": False}, None, {}) == {}
        assert guard.state.interrupted
    finally:
        CONTROL.reset()


@pytest.mark.asyncio
async def test_loop_detected_on_repeated_target(guard):
    msgs = [_ctx(await guard.on_post_tool(_post("click_point", {"x": 400 + i, "y": 300}), None, {}))
            for i in range(3)]
    assert "LOOP DETECTED" in msgs[2]
    assert "LOOP" not in msgs[0] + msgs[1]


@pytest.mark.asyncio
async def test_stall_detected_when_screen_does_not_change(guard):
    out = ""
    for i in range(3):
        out = _ctx(await guard.on_post_tool(
            _post("press_key", {"keys": f"k{i}"}, text="pressed | screen UNCHANGED (0.0%)"), None, {}))
    assert "STALLED" in out


@pytest.mark.asyncio
async def test_goal_reanchored_every_n_calls(guard):
    outs = [_ctx(await guard.on_post_tool(_post("observe", text="screenshot"), None, {}))
            for _ in range(4)]
    assert "TASK ANCHOR" in outs[3]
    assert "rename report.docx" in outs[3]


@pytest.mark.asyncio
async def test_compaction_triggers_immediate_reanchor(guard):
    await guard.on_pre_compact({"trigger": "auto", "custom_instructions": None}, None, {})
    out = _ctx(await guard.on_post_tool(_post("observe", text="x"), None, {}))
    assert "TASK ANCHOR" in out
    assert any("compaction" in p["note"].lower() for p in guard.state.ledger.progress)


@pytest.mark.asyncio
async def test_failure_hook_counts_toward_streak(guard):
    out = ""
    for i in range(3):
        out = _ctx(await guard.on_tool_failure(
            {"tool_name": "mcp__desktop__drag", "tool_input": {"start_x": i * 100},
             "error": "boom", "tool_use_id": "t"}, None, {}))
    assert "FAILING" in out


def test_hooks_config_shape(guard):
    h = guard.hooks()
    assert set(h) == {"Stop", "PostToolUse", "PostToolUseFailure", "PreCompact"}
    assert all(m.hooks for matchers in h.values() for m in matchers)
