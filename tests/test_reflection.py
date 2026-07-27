"""Reflection: parsing, carrying lessons forward, facts to memory, episodic recall."""
import pytest

from autopc.memory import PagingManager, VectorStore
from autopc.tasks import TaskStore
from autopc.tasks.reflection import apply_reflection, reflect, similar_episodes


@pytest.fixture
def ledger(tmp_path):
    t = TaskStore(str(tmp_path / "tasks")).create("open settings and turn on dark mode")
    t.set_criteria(["Settings shows dark mode enabled"])
    t.begin_attempt()
    return t


@pytest.fixture
def memory(tmp_path):
    vs = VectorStore(str(tmp_path / "vectors"), embedder=None)   # keyword mode
    return PagingManager(vs, data_dir=str(tmp_path))


def _gen(data):
    async def gen(prompt, schema, system="", model=None, images=None):
        gen.prompt = prompt
        return data
    return gen


@pytest.mark.asyncio
async def test_reflect_parses_caps_and_includes_trace(ledger):
    gen = _gen({"summary": "Could not find the toggle quickly.",
                "lessons": ["Dark mode lives under Personalization > Colors"] + ["x"] * 9,
                "facts": [{"key": "settings_dark_mode", "value": "Personalization > Colors"},
                          {"key": "", "value": "dropped"}],
                "next_approach": "Go straight to Personalization > Colors."})
    steps = [{"thought": "open menu", "action": {"type": "click_xy", "x": 1, "y": 2},
              "outcome": "ok"}]
    r = await reflect(ledger, "exhausted", steps, generate_json=gen)
    assert len(r.lessons) == 5 and len(r.facts) == 1
    assert "click_xy" in gen.prompt and "exhausted" in gen.prompt


@pytest.mark.asyncio
async def test_reflect_never_raises(ledger):
    async def boom(*a, **k):
        raise RuntimeError("down")
    r = await reflect(ledger, "error", [], generate_json=boom)
    assert r.error and r.summary


@pytest.mark.asyncio
async def test_apply_reflection_dedupes_lessons_and_updates_attempt(ledger):
    from autopc.tasks.reflection import Reflection
    ledger.lessons.append("existing lesson")
    apply_reflection(ledger, "exhausted",
                     Reflection(summary="ran out of ideas", lessons=["existing lesson", "new one"],
                                next_approach="try the keyboard shortcut"))
    assert ledger.lessons == ["existing lesson", "new one"]
    assert "try the keyboard shortcut" in ledger.attempts[-1].reflection


@pytest.mark.asyncio
async def test_apply_reflection_saves_facts_and_episode(ledger, memory):
    from autopc.tasks.reflection import Reflection
    apply_reflection(ledger, "verified",
                     Reflection(summary="done via Colors page",
                                facts=[{"key": "settings_dark_mode", "value": "Personalization > Colors"}]),
                     memory=memory)
    assert memory.memory_get("settings_dark_mode") == "Personalization > Colors"
    hits = memory.vectors.search("EPISODE dark mode settings", top_k=5)
    assert any((h.get("meta") or {}).get("type") == "episode" for h in hits)


def test_similar_episodes_filters_and_formats(ledger, memory):
    from autopc.tasks.reflection import Reflection
    apply_reflection(ledger, "verified", Reflection(summary="enabled dark mode via Colors"),
                     memory=memory)
    out = similar_episodes(memory, "turn on dark mode in settings", k=3)
    assert "EPISODE" in out and "dark mode" in out
    assert similar_episodes(memory, "completely unrelated spreadsheet pivot table", k=3) == ""


def test_similar_episodes_safe_without_memory():
    assert similar_episodes(None, "anything") == ""
