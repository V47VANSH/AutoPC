"""Task contract: sanitizing model output, and never failing a task over it."""
import pytest

from autopc.tasks import TaskStore
from autopc.tasks.contract import (MAX_CRITERIA, apply_contract, derive_contract,
                            fallback_contract)


async def _returns(data):
    async def gen(prompt, schema, system="", model=None, images=None):
        gen.prompt = prompt
        return data
    return gen


@pytest.mark.asyncio
async def test_contract_cleans_dedupes_and_caps():
    gen = await _returns({
        "criteria": ["  Notepad   shows 'hi' ", "notepad shows 'hi'", ""] + [f"c{i}" for i in range(10)],
        "plan": ["open notepad", "type hi"],
        "assumptions": ["default font is fine"],
    })
    c = await derive_contract("open notepad and type hi", generate_json=gen)
    assert c.criteria[0] == "Notepad shows 'hi'"
    assert len(c.criteria) == MAX_CRITERIA          # deduped, empty dropped, capped
    assert c.plan == ["open notepad", "type hi"]
    assert c.assumptions == ["default font is fine"]
    assert not c.fallback


@pytest.mark.asyncio
async def test_contract_falls_back_when_model_errors():
    async def boom(*a, **k):
        raise RuntimeError("api down")
    c = await derive_contract("rename report.docx to final.docx", generate_json=boom)
    assert c.fallback
    assert "rename report.docx to final.docx" in c.criteria[0]


@pytest.mark.asyncio
async def test_contract_falls_back_on_empty_criteria():
    gen = await _returns({"criteria": [], "plan": ["x"]})
    c = await derive_contract("do the thing", generate_json=gen)
    assert c.fallback


@pytest.mark.asyncio
async def test_context_reaches_the_prompt():
    gen = await _returns({"criteria": ["done"], "plan": ["a"]})
    await derive_contract("goal", context="Edge's address bar is at the top", generate_json=gen)
    assert "Edge's address bar" in gen.prompt


def test_apply_contract_sets_first_step_in_progress(tmp_path):
    t = TaskStore(str(tmp_path)).create("goal")
    apply_contract(t, fallback_contract("goal"))
    assert len(t.criteria) == 1
    assert t.plan[0].status == "in_progress"
    assert "fallback" in t.progress[-1]["note"]
