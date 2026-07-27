"""Runner orchestration: attempts loop, retry-after-reflection, verified stop,
needs-user stop, and exhaustion -> failed. SDK and LLM calls are stubbed."""
import pytest

from autopc.agent_brain import runner as rn
from autopc.agent_brain import tools as desktop
from autopc.tasks import TaskStore
from autopc.tasks.reflection import Reflection


async def _consume(stream, started, turns):
    async for _ in stream:
        pass
    return 0


def _build_options(model, max_turns, budget, hooks=None):
    return {"hooks": hooks}


def _make_runner(tmp_path, behaviors, max_attempts=3, monkeypatch=None):
    """behaviors[i](ledger) runs during attempt i+1 (simulating the agent)."""
    store = TaskStore(str(tmp_path / "tasks"))
    runner = rn.TaskRunner(store, consume=_consume, build_options=_build_options,
                           model="x", max_attempts=max_attempts)
    calls = {"n": 0}

    def fake_query(*, prompt, options):
        async def gen():
            i = calls["n"]
            calls["n"] += 1
            if i < len(behaviors):
                behaviors[i](runner._current)
            if False:
                yield None
        return gen()

    async def fake_reflect(ledger, outcome, steps=None, generate_json=None, model=None):
        return Reflection(summary=f"reflected on {outcome}",
                          lessons=[f"lesson after attempt about {outcome}"],
                          next_approach="" if outcome == "verified" else "try the menu instead")

    monkeypatch.setattr(rn, "query", fake_query)
    monkeypatch.setattr(rn, "reflect", fake_reflect)
    runner.calls = calls
    return runner, store


async def _drive(runner, store, goal):
    ledger = store.create(goal, status="active")
    ledger.set_criteria(["the outcome is visible on screen"])  # skip LLM contract
    runner._current = ledger
    return await runner._drive(ledger)


@pytest.fixture(autouse=True)
def _clean_desktop():
    desktop.bind_attempt(None)
    yield
    desktop.bind_attempt(None)


@pytest.mark.asyncio
async def test_verifies_on_second_attempt_after_reflection(tmp_path, monkeypatch):
    def nothing(l):
        pass
    def verify(l):
        l.finish("verified", "done on screen")
    runner, store = _make_runner(tmp_path, [nothing, verify], monkeypatch=monkeypatch)
    led = await _drive(runner, store, "do the thing")
    assert led.status == "verified"
    assert runner.calls["n"] == 2                      # two attempts ran
    assert len([a for a in led.attempts if a.ended]) == 2
    assert led.attempts[0].outcome == "exhausted"
    assert led.attempts[1].outcome == "verified"
    assert any("lesson after attempt" in l for l in led.lessons)


@pytest.mark.asyncio
async def test_exhausts_attempt_budget_then_fails(tmp_path, monkeypatch):
    runner, store = _make_runner(tmp_path, [lambda l: None] * 5, max_attempts=2,
                                 monkeypatch=monkeypatch)
    led = await _drive(runner, store, "impossible thing")
    assert led.status == "failed"
    assert runner.calls["n"] == 2                      # stopped at the budget
    assert "resume" in led.final_summary.lower() or led.final_summary


@pytest.mark.asyncio
async def test_needs_user_stops_without_retrying(tmp_path, monkeypatch):
    def needs_user(l):
        l.add_blocker("need the wifi password", needs_user=True)
        l.finish("needs_user", "need the password")
    runner, store = _make_runner(tmp_path, [needs_user, lambda l: l.finish("verified")],
                                 monkeypatch=monkeypatch)
    led = await _drive(runner, store, "log into the portal")
    assert led.status == "needs_user"
    assert runner.calls["n"] == 1                      # did not retry


@pytest.mark.asyncio
async def test_skips_contract_when_criteria_exist(tmp_path, monkeypatch):
    # derive_contract must NOT be called when the ledger already has criteria.
    async def boom(*a, **k):
        raise AssertionError("derive_contract should not run on resume")
    monkeypatch.setattr(rn, "derive_contract", boom)
    runner, store = _make_runner(tmp_path, [lambda l: l.finish("verified")],
                                 monkeypatch=monkeypatch)
    led = await _drive(runner, store, "already-scoped task")
    assert led.status == "verified"
