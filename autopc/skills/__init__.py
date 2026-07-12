"""
AutoPC skills subsystem -- procedural memory.

Successful multi-step runs are crystallized into portable SKILL.md playbooks;
relevant ones are replayed at the start of a similar goal. This is the
complement to the declarative memory in memory/.
"""
from __future__ import annotations

import os

from autopc import config
from autopc.skills.crystallizer import SkillCrystallizer
from autopc.skills.manager import Skill, SkillManager, SkillMetadata

__all__ = ["Skill", "SkillManager", "SkillMetadata", "SkillCrystallizer",
           "build_skill_manager"]


def build_skill_manager(data_dir: str = None, vector_store=None) -> SkillManager:
    """Construct the skill manager rooted at the configured data directory."""
    data_dir = data_dir or config.data_dir()
    return SkillManager(skills_dir=os.path.join(data_dir, "skills"),
                        faiss_store=vector_store)
