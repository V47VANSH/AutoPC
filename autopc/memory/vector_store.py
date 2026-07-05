"""
Vector store for archival (episodic) memory and skill retrieval.

Holds short text documents with attached metadata and finds them by meaning.
Two modes, chosen automatically from what is installed:

    semantic   an embedder is available AND faiss is installed -> an L2 index
               over unit-normalized embeddings (cosine similarity).
    keyword    otherwise -> an in-memory token-overlap search. No dependencies,
               lower quality, but never unavailable.

Both modes persist to `index_dir` and expose the same two methods, `add` and
`search`, so nothing above this file cares which one is live. The embedding
model is pluggable (see embedder.py) and nothing here needs torch.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
from typing import Dict, List, Optional

import numpy as np

from autopc.memory.embedder import Embedder

try:
    import faiss
except ImportError:
    faiss = None

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


class VectorStore:
    def __init__(self, index_dir: str, embedder: Optional[Embedder] = None):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.embedder = embedder
        self.semantic = bool(embedder and embedder.available and faiss is not None)

        self.metadata: Dict[int, dict] = {}   # id -> {"text", "meta"}
        self.next_id = 0
        self._index = None                      # faiss index (semantic mode)

        self._faiss_path = os.path.join(index_dir, "episodic.faiss")
        self._meta_path = os.path.join(index_dir, "episodic_meta.pkl")
        self._kw_path = os.path.join(index_dir, "episodic_keyword.json")

        if self.semantic:
            self._load_semantic()
            log.info("VectorStore: semantic mode (dim=%d, %d vectors).",
                     self.embedder.dim, self.next_id)
        else:
            self._load_keyword()
            log.info("VectorStore: keyword mode (%d documents).", len(self.metadata))

    # ------------------------------------------------------------------ add
    def add(self, text: str, meta: Optional[dict] = None) -> int:
        """Store one document; returns its id."""
        doc_id = self.next_id
        self.metadata[doc_id] = {"text": text, "meta": meta or {}}
        self.next_id += 1

        if self.semantic:
            vec = self.embedder.encode([text]).astype(np.float32)
            self._index.add(vec)
            self._save_semantic()
        else:
            self._save_keyword()
        return doc_id

    # --------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Return up to top_k matches, each {text, meta, score} (score 0..1)."""
        if not self.metadata:
            return []
        if self.semantic:
            return self._search_semantic(query, top_k)
        return self._search_keyword(query, top_k)

    def _search_semantic(self, query: str, top_k: int) -> List[dict]:
        qv = self.embedder.encode([query]).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []
        distances, ids = self._index.search(qv, k)
        out = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx < 0 or idx not in self.metadata:
                continue
            doc = self.metadata[idx]
            # Normalized vectors -> L2^2 in [0,4]; map to a 0..1 similarity.
            score = max(0.0, 1.0 - float(dist) / 2.0)
            out.append({"text": doc["text"], "meta": doc["meta"], "score": score})
        return out

    def _search_keyword(self, query: str, top_k: int) -> List[dict]:
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for doc in self.metadata.values():
            d = _tokens(doc["text"])
            if not d:
                continue
            overlap = len(q & d)
            if overlap == 0:
                continue
            score = overlap / len(q | d)   # Jaccard
            scored.append({"text": doc["text"], "meta": doc["meta"], "score": score})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    # ---------------------------------------------------------- persistence
    def _load_semantic(self) -> None:
        if os.path.exists(self._faiss_path) and os.path.exists(self._meta_path):
            try:
                self._index = faiss.read_index(self._faiss_path)
                with open(self._meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                self.next_id = (max(self.metadata) + 1) if self.metadata else 0
                return
            except Exception as exc:
                log.warning("Rebuilding vector index (load failed: %s).", exc)
        self._index = faiss.IndexFlatL2(self.embedder.dim)

    def _save_semantic(self) -> None:
        try:
            faiss.write_index(self._index, self._faiss_path)
            with open(self._meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as exc:
            log.warning("Failed to persist vector index: %s", exc)

    def _load_keyword(self) -> None:
        if not os.path.exists(self._kw_path):
            return
        try:
            with open(self._kw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = {int(k): v for k, v in data.get("docs", {}).items()}
            self.next_id = (max(self.metadata) + 1) if self.metadata else 0
        except Exception as exc:
            log.warning("Failed to load keyword store: %s", exc)

    def _save_keyword(self) -> None:
        try:
            with open(self._kw_path, "w", encoding="utf-8") as f:
                json.dump({"docs": {str(k): v for k, v in self.metadata.items()}},
                          f, ensure_ascii=False)
        except Exception as exc:
            log.warning("Failed to persist keyword store: %s", exc)
