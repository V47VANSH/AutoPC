"""
Skill crystallizer -- learn a reusable skill from a successful run.

After the agent completes a task that took several real steps, the crystallizer
hands the action trace to Claude and asks it to distill a reusable SKILL.md: a
plain-English playbook of the steps, how to verify success, and the pitfalls it
hit. Next time a similar goal arrives, the agent replays the skill instead of
re-reasoning the whole task.

Distillation runs through an injected async `generate_fn(prompt, system) -> str`
(wired to Claude in agent_brain/run.py), so this module stays free of any
provider or SDK import.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Awaitable, Callable, List, Optional

from autopc.skills.manager import Skill, SkillManager, SkillMetadata

log = logging.getLogger(__name__)

GenerateFn = Callable[[str, str], Awaitable[str]]

CRYSTALLIZE_PROMPT = """You are the Skill Crystallizer of an intelligent PC automation agent.

You have just observed the agent successfully complete a task. Distill the
action trace into a reusable skill document.

## Input
- Goal: the original user goal
- Action Trace: the sequence of (thought, action, outcome) tuples that achieved it

## Output Format
Write the skill body in Markdown with these exact sections:

# [Skill Title]

## Steps
1. [First step in plain English]
2. [Second step...]

## Verification
- [How to verify the goal was achieved]

## Known Pitfalls
- [Common failure modes and how to handle them]

## Notes
- [Any useful context or alternatives]

IMPORTANT: Write steps in plain English (e.g., "Click the File menu"), NOT as raw
JSON actions. The agent interprets these steps against the live screen. Output
ONLY the markdown skill body -- no code fences, no preamble."""

TRIGGER_PROMPT = """Given this skill name and description, generate a pipe-separated
list of 3-5 short trigger phrases (2-4 words each) a user might say to invoke it.

Skill: {name}
Description: {description}

Example output: open chrome|launch browser|start chrome|web browser

Output ONLY the pipe-separated triggers, nothing else."""


class SkillCrystallizer:
    def __init__(self, generate_fn: GenerateFn, skill_manager: SkillManager,
                 min_steps: int = 5):
        self.generate = generate_fn
        self.skills = skill_manager
        self.min_steps = min_steps

    def should_crystallize(self, meaningful_steps: List[dict], success: bool) -> bool:
        """Was the run successful and complex enough to be worth saving?"""
        return bool(success) and len(meaningful_steps) >= self.min_steps

    async def crystallize(self, goal: str, meaningful_steps: List[dict]) -> Optional[Skill]:
        """Distill a successful trace into a SKILL.md. Returns the saved Skill, or
        None if distillation failed (which is never fatal -- the goal already
        succeeded; a skill is a bonus)."""
        trace_text = ""
        for i, step in enumerate(meaningful_steps, 1):
            thought = step.get("thought", "")
            action = json.dumps(step.get("action", {}))
            outcome = step.get("outcome", "success")
            trace_text += f"\nStep {i}:\n  Thought: {thought}\n  Action: {action}\n  Outcome: {outcome}\n"

        prompt = (f"Goal: {goal}\n\nAction Trace:\n{trace_text}\n\n"
                  "Now write the skill body following your instructions.")
        try:
            body = await self.generate(prompt, CRYSTALLIZE_PROMPT)
        except Exception as exc:
            log.warning("Crystallization failed for goal %r: %s", goal, exc)
            return None
        if not body or not body.strip():
            log.warning("Crystallization returned nothing for goal %r.", goal)
            return None

        name = base = self._skill_name(goal)
        n = 2
        while self.skills.has_skill(name):   # never overwrite a learned skill
            name = f"{base}_{n}"
            n += 1
        description = goal if len(goal) <= 60 else goal[:57] + "..."
        triggers = await self._triggers(name, description)

        skill = Skill(metadata=SkillMetadata(
            name=name, description=description, version=1, trigger=triggers,
            success_rate=1.0, times_used=1,
            last_used=datetime.now().strftime("%Y-%m-%d"), created_by="agent",
        ), body=body.strip())
        self.skills.save_skill(skill)
        log.info("Crystallized new skill: %s", name)
        return skill

    @staticmethod
    def _skill_name(goal: str) -> str:
        stop = {"the", "a", "an", "to", "for", "in", "on", "at", "and", "or", "is", "my"}
        words = [w for w in goal.lower().split() if w not in stop and w.isalnum()]
        return "_".join(words[:4]) or "unnamed_skill"

    async def _triggers(self, name: str, description: str) -> str:
        try:
            reply = await self.generate(TRIGGER_PROMPT.format(name=name, description=description), "")
        except Exception as exc:
            log.debug("Trigger generation failed: %s", exc)
            return name.replace("_", " ")
        if reply and "|" in reply:
            return reply.strip().replace("\n", "")
        return name.replace("_", " ")
