"""Completion verifier: verdict normalization and the accept/reject policy."""
import pytest

from autopc.tasks import TaskStore
from autopc.tasks.verifier import (MAX_UNCERTAIN_REJECTIONS, Verdict, decide, record,
                            verify)


@pytest.fixture
def ledger(tmp_path):
    t = TaskStore(str(tmp_path)).create("save rain.txt to the desktop")
    t.set_criteria(["notepad shows a haiku", "title bar shows rain.txt, no unsaved marker"])
    t.begin_attempt()
    return t


def _gen(data):
    async def gen(prompt, schema, system="", model=None, images=None):
        gen.images = images
        gen.prompt = prompt
        return data
    return gen


@pytest.mark.asyncio
async def test_pass_requires_every_criterion(ledger):
    gen = _gen({"overall": "pass", "missing": "",
                "criteria": [{"id": "c1", "met": True, "reason": "visible"}]})  # c2 omitted
    v = await verify(ledger, screenshot=object(), claim="done", generate_json=gen)
    assert v.overall == "uncertain"            # a skipped criterion is never 'met'
    assert [c["id"] for c in v.unmet()] == ["c2"]
    assert gen.images and "claim (to test, not to trust)" in gen.prompt.lower()


@pytest.mark.asyncio
async def test_verifier_error_never_raises(ledger):
    async def boom(*a, **k):
        raise RuntimeError("timeout")
    v = await verify(ledger, generate_json=boom)
    assert v.error and v.overall == "uncertain"


def test_decide_pass_accepts(ledger):
    ok, msg = decide(Verdict(overall="pass", criteria=[{"id": "c1", "met": True},
                                                       {"id": "c2", "met": True}]), ledger)
    assert ok and "Verified" in msg


def test_decide_fail_rejects_with_specifics(ledger):
    v = Verdict(overall="fail", missing="file not saved",
                criteria=[{"id": "c1", "met": True, "reason": "ok"},
                          {"id": "c2", "met": False, "reason": "title shows '*Untitled'"}])
    ok, msg = decide(v, ledger)
    assert not ok
    assert "*Untitled" in msg and "file not saved" in msg


def test_uncertain_rejected_until_cap_then_accepted_on_evidence(ledger):
    v = Verdict(overall="uncertain", missing="show the saved file",
                criteria=[{"id": "c1", "met": True, "reason": ""},
                          {"id": "c2", "met": False, "reason": "can't see save state"}])
    assert decide(v, ledger, uncertain_rejections=0)[0] is False
    ledger.mark_criterion("c1", True, "haiku visible")
    ledger.mark_criterion("c2", True, "saved via Ctrl+S, dialog closed")
    assert decide(v, ledger, uncertain_rejections=MAX_UNCERTAIN_REJECTIONS - 1)[0] is False
    assert decide(v, ledger, uncertain_rejections=MAX_UNCERTAIN_REJECTIONS)[0] is True


def test_verifier_down_needs_agent_evidence(ledger):
    v = Verdict(error="api down")
    assert decide(v, ledger)[0] is False
    ledger.mark_criterion("c1", True, "visible")
    ledger.mark_criterion("c2", True, "title shows rain.txt")
    assert decide(v, ledger)[0] is True


def test_record_syncs_criteria_from_judge(ledger):
    v = Verdict(overall="fail", missing="x",
                criteria=[{"id": "c1", "met": True, "reason": "haiku visible"},
                          {"id": "c2", "met": False, "reason": "unsaved marker present"}])
    record(ledger, v, accepted=False)
    assert [c.status for c in ledger.criteria] == ["met", "unmet"]
    assert ledger.verdicts[-1]["overall"] == "fail"
    record(ledger, Verdict(overall="pass", criteria=[]), accepted=True)
    assert all(c.status == "met" for c in ledger.criteria)
