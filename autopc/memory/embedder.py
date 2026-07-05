"""
Text embedders for semantic recall.

The agent's archival memory and skill retrieval both want to find things by
*meaning*, not just keyword overlap. That needs an embedding model -- but the
obvious choice (sentence-transformers) drags in PyTorch, which is hundreds of
megabytes and wrecks the packaged .exe. So embedding is kept behind a small
interface with three interchangeable backends:

    onnx     fastembed (ONNX Runtime, no torch). The default. Local, offline,
             ~130MB for bge-small-en-v1.5. Best recall for the size.
    keyword  no model at all -- signals "unavailable" so the vector store falls
             back to token-overlap search. Zero dependencies, always works.
    none     recall disabled entirely.

Nothing above the interface knows which backend is live: callers ask
`make_embedder()` for one and check `.available`. A missing optional dependency
degrades to keyword search rather than crashing a run.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    """Base interface. `available` is False when no real model is loaded."""

    available: bool = False
    dim: int = 0

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of unit-normalized vectors."""
        raise NotImplementedError


class OnnxEmbedder(Embedder):
    """
    Local embeddings via fastembed (ONNX Runtime, no PyTorch).

    The model is downloaded once on first use into `cache_dir` and reused after.
    Construction is lazy-friendly: if fastembed is not installed, `available`
    stays False and the caller transparently falls back to keyword search.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_dir: Optional[str] = None):
        self.model_name = model_name
        self._model = None
        try:
            from fastembed import TextEmbedding
        except ImportError:
            log.warning(
                "fastembed not installed; semantic recall will use keyword "
                "search. `pip install fastembed` for better recall."
            )
            return
        try:
            self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
            # Probe the dimension once with a throwaway encode.
            self.dim = int(next(iter(self._model.embed(["probe"]))).shape[0])
            self.available = True
            log.info("Embedder ready: %s (dim=%d)", model_name, self.dim)
        except Exception as exc:
            # A bad model id, a download failure, or a broken ONNX runtime must
            # not take down the agent -- recall is a bonus, not a requirement.
            log.warning("Could not initialise embedder %r (%s); using keyword "
                        "search instead.", model_name, exc)
            self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.available or self._model is None:
            raise RuntimeError("OnnxEmbedder.encode called while unavailable.")
        vecs = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        # Normalize so an L2 index behaves like cosine similarity.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def make_embedder(kind: str = "onnx", model: str = "BAAI/bge-small-en-v1.5",
                  cache_dir: Optional[str] = None) -> Embedder:
    """
    Build the configured embedder, degrading gracefully.

    `kind` comes from AUTOPC_EMBEDDER. "onnx" tries fastembed and falls back to
    keyword search if it or the model is unavailable; "keyword"/"none" skip the
    model entirely and return an unavailable Embedder.
    """
    kind = (kind or "onnx").strip().lower()
    if kind in ("keyword", "none", ""):
        return Embedder()  # unavailable -> keyword fallback in the vector store
    if kind == "onnx":
        return OnnxEmbedder(model_name=model, cache_dir=cache_dir)
    log.warning("Unknown AUTOPC_EMBEDDER=%r; using keyword search.", kind)
    return Embedder()
