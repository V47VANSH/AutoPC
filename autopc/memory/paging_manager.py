"""
MemGPT-style paging manager -- three-tier memory with LLM-controlled paging.

    HOT  (Working)  last N actions / current state. In-prompt, ephemeral.
    WARM (Core)     learned facts and preferences. In-prompt, self-editable,
                    persisted to MEMORY.md.
    COLD (Archival) everything older, reached by tools: semantic search over a
                    vector store, plus exact key recall from a JSON KV file.

The agent edits WARM and pages between WARM and COLD using the memory_* tools,
following the MemGPT/Letta design. This is declarative memory; procedural
memory (learned skills) lives in skills/.

No database is needed: exact archival recall uses a small JSON file, and
semantic recall uses the pluggable VectorStore (ONNX embeddings or keyword
fallback, no torch).
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class PagingManager:
    def __init__(
        self,
        vector_store,
        data_dir: str = "data",
        core_memory_file: Optional[str] = None,
        max_working_items: int = 8,
        max_core_entries: int = 20,
    ):
        self.vectors = vector_store
        self.data_dir = data_dir
        self.core_memory_file = core_memory_file or os.path.join(data_dir, "MEMORY.md")
        self.archival_file = os.path.join(data_dir, "archival_kv.json")
        self.max_working_items = max_working_items
        self.max_core_entries = max_core_entries

        self.working_memory: List[dict] = []
        self.core_memory: Dict[str, str] = {}
        self._archival_kv: Dict[str, dict] = {}

        self._load_core_memory()
        self._load_archival()

    # ================================================================ HOT
    def add_to_working(self, event: dict) -> None:
        event["timestamp"] = datetime.now().isoformat()
        self.working_memory.append(event)
        if len(self.working_memory) > self.max_working_items:
            self.working_memory.pop(0)

    def get_working_context(self) -> str:
        if not self.working_memory:
            return "Working Memory: (empty -- no recent actions)"
        lines = ["Working Memory (recent actions, newest last):"]
        for i, ev in enumerate(self.working_memory):
            ts = ev.get("timestamp", "")[:19]
            action = ev.get("action", {})
            thought = ev.get("thought", "")
            outcome = ev.get("outcome", "")
            lines.append(f"  [{i+1}] {ts} | {thought[:60]} -> {json.dumps(action)} -> {outcome}")
        return "\n".join(lines)

    def clear_working(self) -> None:
        self.working_memory.clear()

    # =============================================================== WARM
    #: Overlap above which two facts are treated as the same fact. Core memory
    #: is injected into every planning prompt, so a duplicate crowds out
    #: unrelated facts via oldest-first archiving and reads as false emphasis.
    #: Measured as CONTAINMENT (shared over the smaller fact), not Jaccard, so a
    #: terse restatement of a verbose fact still matches.
    DUPLICATE_SIMILARITY = 0.6
    DUPLICATE_MIN_SHARED = 4

    @staticmethod
    def _fact_tokens(value: str) -> set:
        stop = {"the", "a", "an", "is", "are", "to", "of", "in", "on", "at", "it",
                "be", "will", "may", "should", "must", "and", "or", "if", "when",
                "before", "after", "this", "that", "for", "with", "not", "into",
                "ensure", "always", "use", "using", "via", "being", "avoid"}
        tokens = set()
        for word in re.findall(r"[a-z0-9_]+", value.lower()):
            if word in stop or len(word) <= 2:
                continue
            for suffix in ("ing", "ed", "es", "s"):
                if len(word) > len(suffix) + 2 and word.endswith(suffix):
                    word = word[:-len(suffix)]
                    break
            tokens.add(word)
        return tokens

    def _find_duplicate(self, key: str, value: str) -> Optional[str]:
        if key in self.core_memory:
            return key
        incoming = self._fact_tokens(value)
        if len(incoming) < 4:
            return None
        for existing_key, existing_value in self.core_memory.items():
            existing = self._fact_tokens(existing_value)
            if len(existing) < 4:
                continue
            shared = len(incoming & existing)
            if shared < self.DUPLICATE_MIN_SHARED:
                continue
            if shared / min(len(incoming), len(existing)) >= self.DUPLICATE_SIMILARITY:
                return existing_key
        return None

    def memory_save(self, key: str, value: str) -> None:
        """Save a fact to Core Memory, merging near-duplicates. (LLM tool)"""
        duplicate = self._find_duplicate(key, value)
        if duplicate is not None:
            self.core_memory[duplicate] = value
            self._persist_core_memory()
            log.info("Core Memory MERGE: %s (restating '%s')", key, duplicate)
            return
        if len(self.core_memory) >= self.max_core_entries:
            self.memory_archive(next(iter(self.core_memory)))  # evict oldest
        self.core_memory[key] = value
        self._persist_core_memory()
        log.info("Core Memory SAVE: %s = %s", key, value[:80])

    def memory_update(self, key: str, value: str) -> None:
        if key not in self.core_memory:
            log.warning("Core Memory UPDATE: key %r not found; saving as new.", key)
        self.core_memory[key] = value
        self._persist_core_memory()

    def memory_get(self, key: str) -> Optional[str]:
        return self.core_memory.get(key)

    def get_core_context(self) -> str:
        if not self.core_memory:
            return "Core Memory: (empty -- no learned facts yet)"
        lines = ["Core Memory (persistent facts):"]
        for key, value in self.core_memory.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines)

    # =============================================================== COLD
    def memory_archive(self, key: str) -> None:
        """Move a Core Memory entry to Archival. (LLM tool: page out)"""
        if key not in self.core_memory:
            log.warning("Cannot archive: key %r not in Core Memory.", key)
            return
        value = self.core_memory.pop(key)
        self._archival_kv[key] = {"value": value, "archived_at": datetime.now().isoformat()}
        self.vectors.add(f"core_memory:{key} = {value}",
                         {"type": "archived_core", "key": key})
        self._persist_archival()
        self._persist_core_memory()
        log.info("Core Memory ARCHIVED: %s -> cold storage", key)

    def memory_recall(self, key: str) -> Optional[str]:
        """Bring an Archival entry back into Core Memory. (LLM tool: page in)"""
        entry = self._archival_kv.get(key)
        if entry and "value" in entry:
            self.memory_save(key, entry["value"])
            log.info("Core Memory RECALLED: %s <- cold storage", key)
            return entry["value"]
        log.warning("Cannot recall: key %r not found in Archival.", key)
        return None

    def memory_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Semantic search over Archival Memory. (LLM tool)"""
        return self.vectors.search(query, top_k=top_k)

    def store_experience(self, state: dict, action: dict, reward: float,
                         thought: str = "") -> None:
        """Save a complete experience tuple to episodic (archival) memory."""
        state_text = json.dumps(state) if isinstance(state, dict) else str(state)
        self.vectors.add(
            f"experience: {thought} | action={json.dumps(action)} | reward={reward}",
            {"type": "experience", "action": action, "reward": reward,
             "state": state_text[:500]},
        )

    def recall_similar(self, current_context: str, top_k: int = 3) -> List[dict]:
        return self.vectors.search(current_context, top_k=top_k)

    # =================================================== context assembly
    def get_full_memory_context(self) -> str:
        return "\n".join([
            self.get_working_context(), "",
            self.get_core_context(), "",
            "Archival Memory: use memory_search(query) to recall past experiences.",
        ])

    # ========================================================= persistence
    def _load_core_memory(self) -> None:
        if not os.path.exists(self.core_memory_file):
            return
        try:
            with open(self.core_memory_file, "r", encoding="utf-8") as f:
                for line in f.read().strip().split("\n"):
                    line = line.strip()
                    if line.startswith("- ") and ": " in line:
                        key, _, value = line[2:].partition(": ")
                        self.core_memory[key.strip()] = value.strip()
            log.info("Loaded %d core memory entries.", len(self.core_memory))
        except Exception as exc:
            log.warning("Failed to load core memory: %s", exc)

    def _persist_core_memory(self) -> None:
        os.makedirs(os.path.dirname(self.core_memory_file) or ".", exist_ok=True)
        try:
            lines = [f"- {k}: {v}" for k, v in self.core_memory.items()]
            with open(self.core_memory_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            log.warning("Failed to persist core memory: %s", exc)

    def _load_archival(self) -> None:
        if not os.path.exists(self.archival_file):
            return
        try:
            with open(self.archival_file, "r", encoding="utf-8") as f:
                self._archival_kv = json.load(f)
        except Exception as exc:
            log.warning("Failed to load archival KV: %s", exc)

    def _persist_archival(self) -> None:
        try:
            with open(self.archival_file, "w", encoding="utf-8") as f:
                json.dump(self._archival_kv, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            log.warning("Failed to persist archival KV: %s", exc)

    # ================================================= LLM tool descriptions
    @staticmethod
    def get_memory_tools_description() -> str:
        return """
Memory tools (persist across runs; use them to remember and recall):
- memory_save(key, value)     save a new fact to Core Memory (user prefs, app
                              layouts, task scope/rules) -- survives restarts
- memory_update(key, value)   update an existing Core Memory entry
- memory_search(query)        semantic search over past experiences (Archival)
- memory_archive(key)         move a Core Memory entry to Archival (free space)
- memory_recall(key)          bring an archived entry back into Core Memory
""".strip()
