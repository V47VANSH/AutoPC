"""Skill manager and crystallizer: safe names, no silent overwrites, matching that
needs more than one shared word, and pruning of failing skills."""
import pytest

from autopc.skills import SkillCrystallizer, SkillManager
from autopc.skills.manager import Skill, SkillMetadata


@pytest.fixture
def mgr(tmp_path):
    return SkillManager(skills_dir=str(tmp_path / "skills"))


def _save(mgr, name, description, trigger="", times_used=0, success_rate=0.0):
    mgr.save_skill(Skill(metadata=SkillMetadata(
        name=name, description=description, trigger=trigger,
        times_used=times_used, success_rate=success_rate), body="1. do it"))


def test_skill_view_rejects_paths(mgr, tmp_path):
    (tmp_path / "secret.md").write_text("nope", encoding="utf-8")
    assert mgr.skill_view("../secret") is None
    assert mgr.skill_view("..\\secret") is None


def test_save_rejects_unsafe_names(mgr):
    with pytest.raises(ValueError):
        _save(mgr, "../evil", "x")


def test_one_shared_word_is_not_a_match(mgr):
    _save(mgr, "open_notepad_write_haiku", "open notepad and write a haiku about winter")
    assert mgr.find_relevant_skills("open chrome and check the weather") == []
    hits = mgr.find_relevant_skills("open notepad and write a haiku about summer")
    assert [h["name"] for h in hits] == ["open_notepad_write_haiku"]


def test_trigger_match(mgr):
    _save(mgr, "dark_mode", "turn on dark mode", trigger="dark mode|dark theme")
    assert mgr.find_relevant_skills("switch windows to dark theme")[0]["name"] == "dark_mode"


def test_prune_failing_skills(mgr):
    _save(mgr, "flaky", "a flaky skill", times_used=12, success_rate=0.1)
    _save(mgr, "solid", "a solid skill", times_used=12, success_rate=0.9)
    assert mgr.prune_skills() == ["flaky"]
    assert mgr.has_skill("solid") and not mgr.has_skill("flaky")


@pytest.mark.asyncio
async def test_crystallizer_never_overwrites(mgr):
    async def generate(prompt, system=""):
        return "open|launch" if "trigger" in prompt.lower() else "1. step\n2. step"

    c = SkillCrystallizer(generate, mgr, min_steps=1)
    steps = [{"thought": "t", "action": {"type": "click"}, "outcome": "ok"}]
    a = await c.crystallize("open notepad write haiku about winter", steps)
    b = await c.crystallize("open notepad write haiku about summer", steps)
    assert a.metadata.name == "open_notepad_write_haiku"
    assert b.metadata.name == "open_notepad_write_haiku_2"
    assert mgr.has_skill(a.metadata.name) and mgr.has_skill(b.metadata.name)
