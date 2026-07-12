"""
Skill manager -- procedural memory.

Skills are portable SKILL.md files with YAML frontmatter, the same open format
the Claude Agent SDK uses for its own skills. Each one is a plain-English
playbook the agent distilled from a successful run; the agent reads the
relevant ones at the start of a similar goal and replays the steps instead of
re-reasoning from scratch.

Progressive loading keeps the agent's context small:
    skills_list()  -> compact index of every skill (~name + description)
    skill_view()   -> the full SKILL.md for one skill

Retrieval is trigger + keyword matching by default; pass a vector store with a
`search_similar_experiences` method to add semantic retrieval.
"""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import yaml

log = logging.getLogger(__name__)

#: Skill names become file names, and the model can ask for any name, so only
#: plain identifiers are accepted -- never a path.
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class SkillMetadata:
    """Parsed YAML frontmatter of a SKILL.md file."""
    name: str = ""
    description: str = ""
    version: int = 1
    platforms: List[str] = field(default_factory=lambda: ["windows", "linux", "macos"])
    trigger: str = ""
    success_rate: float = 0.0
    times_used: int = 0
    last_used: str = ""
    created_by: str = "agent"
    tags: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """A complete loaded skill."""
    metadata: SkillMetadata
    body: str  # the markdown body (steps, verification, pitfalls)
    file_path: str = ""


class SkillManager:
    def __init__(self, skills_dir: str = "data/skills", faiss_store=None):
        self.skills_dir = skills_dir
        self.faiss = faiss_store
        os.makedirs(self.skills_dir, exist_ok=True)
        self._index_cache: Dict[str, SkillMetadata] = {}
        self._rebuild_index()

    # ------------------------------------------------------------ index
    def _rebuild_index(self) -> None:
        self._index_cache.clear()
        for filepath in glob.glob(os.path.join(self.skills_dir, "*.md")):
            try:
                meta, _ = self._parse_skill_file(filepath)
                if meta.name:
                    self._index_cache[meta.name] = meta
            except Exception as exc:
                log.warning("Failed to index skill %s: %s", filepath, exc)
        log.info("Skill index rebuilt: %d skills.", len(self._index_cache))

    def _parse_skill_file(self, filepath: str) -> tuple:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        fm = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not fm:
            return SkillMetadata(), content
        try:
            data = yaml.safe_load(fm.group(1)) or {}
        except yaml.YAMLError:
            data = {}
        meta = SkillMetadata(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", 1),
            platforms=data.get("platforms", ["windows", "linux", "macos"]),
            trigger=data.get("trigger", ""),
            success_rate=data.get("success_rate", 0.0),
            times_used=data.get("times_used", 0),
            last_used=data.get("last_used", ""),
            created_by=data.get("created_by", "agent"),
            tags=data.get("tags", []),
        )
        return meta, fm.group(2).strip()

    # --------------------------------------------- Level 0 / 1 / 2 access
    def skills_list(self) -> str:
        """Compact index of all skills."""
        if not self._index_cache:
            return "No skills learned yet."
        lines = ["Learned skills (replay these for a matching goal):"]
        for name, meta in sorted(self._index_cache.items()):
            trig = f' (triggers: "{meta.trigger}")' if meta.trigger else ""
            rate = f"{meta.success_rate:.0%}" if meta.times_used else "new"
            lines.append(f"  - {name}: {meta.description}{trig} "
                         f"[used {meta.times_used}x, success={rate}]")
        return "\n".join(lines)

    def _path(self, name: str) -> Optional[str]:
        if not _VALID_NAME.match(name or ""):
            return None
        return os.path.join(self.skills_dir, f"{name}.md")

    def skill_view(self, name: str) -> Optional[str]:
        """Full SKILL.md content for one skill."""
        filepath = self._path(name)
        if filepath is None or not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # --------------------------------------------------------- retrieval
    def find_relevant_skills(self, goal: str, top_k: int = 3) -> List[dict]:
        """Trigger + keyword matching (plus optional semantic search)."""
        results: List[dict] = []
        goal_lower = goal.lower()
        goal_words = set(_WORD.findall(goal_lower))
        for name, meta in self._index_cache.items():
            score = 0.0
            if meta.trigger:
                for trig in (t.strip() for t in meta.trigger.split("|")):
                    if trig and trig.lower() in goal_lower:
                        score = 0.9
                        break
            if not score:
                if meta.name.lower() in goal_lower or goal_lower in meta.name.lower():
                    score = 0.7
                else:
                    # Most of the description's meaningful words must appear
                    # in the goal -- one shared word ("open", "file") is not a
                    # match, and a false match both misleads the agent and
                    # blocks a new skill from being learned for this goal.
                    words = {w for w in _WORD.findall(meta.description.lower())
                             if len(w) > 3}
                    if words and len(words & goal_words) / len(words) >= 0.6:
                        score = 0.4
            if score > 0:
                results.append({"name": meta.name, "description": meta.description,
                                "score": score, "success_rate": meta.success_rate,
                                "times_used": meta.times_used})

        if self.faiss and hasattr(self.faiss, "search_similar_experiences"):
            try:
                for fr in self.faiss.search_similar_experiences(f"skill: {goal}", top_k=top_k):
                    m = fr.get("metadata", {})
                    if m.get("type") == "skill_embedding":
                        results.append({"name": m.get("skill_name", "unknown"),
                                        "description": m.get("description", ""),
                                        "score": max(0.0, 1.0 - fr.get("distance", 1.0)),
                                        "success_rate": m.get("success_rate", 0),
                                        "times_used": m.get("times_used", 0)})
            except Exception as exc:
                log.warning("Semantic skill search failed: %s", exc)

        seen, unique = set(), []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique[:top_k]

    # --------------------------------------------------------------- CRUD
    def save_skill(self, skill: Skill) -> None:
        filepath = self._path(skill.metadata.name)
        if filepath is None:
            raise ValueError(f"invalid skill name {skill.metadata.name!r}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._render_skill_file(skill))
        self._index_cache[skill.metadata.name] = skill.metadata
        log.info("Skill saved: %s -> %s", skill.metadata.name, filepath)

    def grade_skill(self, name: str, success: bool) -> None:
        """Update success_rate and times_used after a replay."""
        filepath = self._path(name)
        if filepath is None or not os.path.exists(filepath):
            return
        meta, body = self._parse_skill_file(filepath)
        meta.times_used += 1
        meta.success_rate = (
            (meta.success_rate * (meta.times_used - 1) + (1.0 if success else 0.0))
            / meta.times_used
        )
        meta.last_used = datetime.now().strftime("%Y-%m-%d")
        self.save_skill(Skill(metadata=meta, body=body, file_path=filepath))

    def prune_skills(self, min_uses: int = 10, min_rate: float = 0.3) -> List[str]:
        """Remove skills that keep failing after enough tries."""
        pruned = []
        for name, meta in list(self._index_cache.items()):
            if meta.times_used >= min_uses and meta.success_rate < min_rate:
                filepath = self._path(name)
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                del self._index_cache[name]
                pruned.append(name)
                log.info("Pruned skill %s (rate=%.0f%%, used=%dx).",
                         name, meta.success_rate * 100, meta.times_used)
        return pruned

    def has_skill(self, name: str) -> bool:
        return name in self._index_cache

    # ------------------------------------------------------------ render
    def _render_skill_file(self, skill: Skill) -> str:
        meta_dict = {
            "name": skill.metadata.name,
            "description": skill.metadata.description,
            "version": skill.metadata.version,
            "platforms": skill.metadata.platforms,
            "trigger": skill.metadata.trigger,
            "success_rate": round(skill.metadata.success_rate, 2),
            "times_used": skill.metadata.times_used,
            "last_used": skill.metadata.last_used,
            "created_by": skill.metadata.created_by,
            "tags": skill.metadata.tags,
        }
        yaml_str = yaml.dump(meta_dict, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_str}---\n\n{skill.body}"
