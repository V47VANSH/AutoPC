"""Task-control tools and the verified completion gate, with the screen and the
verifier stubbed out."""
import pytest

from autopc.agent_brain import tools
from autopc.tasks import TaskStore
from autopc.tasks.state import AttemptState
from autopc.tasks.verifier import Verdict


def _txt(result):
    return result["content"][0]["text"]


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    L = TaskStore(str(tmp_path)).create("type hello in notepad")
    L.set_criteria(["notepad shows hello"])
    L.begin_attempt()
    A = AttemptState(ledger=L)
    tools.bind_attempt(A)
    # Pretend the body is wired, and make the "fresh screenshot" free.
    monkeypatch.setattr(tools, "_PERCEPTION", object())
    monkeypatch.setattr(tools, "_EXECUTOR", object())

    async def fake_evidence():
        return None, "Untitled - Notepad", "hello"
    monkeypatch.setattr(tools, "_screen_evidence", fake_evidence)
    yield A
    tools.bind_attempt(None)


def _verdict(monkeypatch, v):
    async def fake_verify(*a, **k):
        return v
    import autopc.tasks.verifier as tv
    monkeypatch.setattr(tv, "verify", fake_verify)


@pytest.mark.asyncio
async def test_tools_degrade_without_a_task():
    tools.bind_attempt(None)
    assert "No task ledger" in _txt(await tools.task_status.handler({}))


@pytest.mark.asyncio
async def test_update_plan_and_status(attempt):
    await tools.update_plan.handler({"items": [
        {"text": "open notepad", "status": "in_progress"},
        {"text": "type hello", "status": "in_progress"}]})
    assert [p.status for p in attempt.ledger.plan] == ["in_progress", "pending"]
    assert "type hello in notepad" in _txt(await tools.task_status.handler({}))


@pytest.mark.asyncio
async def test_mark_unknown_criterion_is_an_error(attempt):
    r = await tools.mark_criterion.handler({"id": "c7", "met": True, "evidence": "x"})
    assert r.get("isError") and "c1" in _txt(r)


@pytest.mark.asyncio
async def test_giving_up_is_pushed_back_once_then_accepted(attempt):
    r1 = await tools.complete_task.handler({"success": False, "summary": "can't find it"})
    assert "NOT ACCEPTED YET" in _txt(r1)
    assert not attempt.may_end()
    await tools.complete_task.handler({"success": False, "summary": "tried 4 ways"})
    assert attempt.gave_up and attempt.may_end()
    assert attempt.ledger.status == "active"     # the runner decides retry vs fail


@pytest.mark.asyncio
async def test_user_blocker_ends_immediately(attempt):
    await tools.report_blocker.handler({"description": "needs the wifi password",
                                        "needs_user": True})
    await tools.complete_task.handler({"success": False, "summary": "need password"})
    assert attempt.ledger.status == "needs_user" and attempt.may_end()


@pytest.mark.asyncio
async def test_success_claim_rejected_when_verifier_fails(attempt, monkeypatch):
    _verdict(monkeypatch, Verdict(overall="fail", missing="text not typed",
                                  criteria=[{"id": "c1", "met": False, "reason": "editor empty"}]))
    r = await tools.complete_task.handler({"success": True, "summary": "done"})
    assert "NOT VERIFIED" in _txt(r)
    assert attempt.ledger.status == "active" and not attempt.may_end()
    assert attempt.rejected_claims == 1


@pytest.mark.asyncio
async def test_success_claim_accepted_when_verified(attempt, monkeypatch):
    _verdict(monkeypatch, Verdict(overall="pass",
                                  criteria=[{"id": "c1", "met": True, "reason": "hello visible"}]))
    r = await tools.complete_task.handler({"success": True, "summary": "typed hello"})
    assert "complete" in _txt(r).lower()
    assert attempt.ledger.status == "verified" and attempt.may_end()
    assert tools.trace().success is True


@pytest.mark.asyncio
async def test_uncertain_counts_toward_acceptance(attempt, monkeypatch):
    _verdict(monkeypatch, Verdict(overall="uncertain", missing="show it",
                                  criteria=[{"id": "c1", "met": False, "reason": "not visible"}]))
    await tools.complete_task.handler({"success": True, "summary": "done"})
    assert attempt.uncertain_rejections == 1
