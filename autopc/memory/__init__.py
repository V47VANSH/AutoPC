"""
AutoPC memory subsystem.

`build_memory()` assembles the whole stack from environment configuration:

    embedder (ONNX / keyword)  ->  vector store (FAISS / keyword)  ->  paging
    manager (HOT working / WARM core / COLD archival)

Everything degrades gracefully: with no fastembed and no faiss installed you
still get working + core memory and keyword-based archival recall.
"""
from __future__ import annotations

import os

from autopc import config
from autopc.memory.embedder import make_embedder
from autopc.memory.paging_manager import PagingManager
from autopc.memory.vector_store import VectorStore

__all__ = ["PagingManager", "VectorStore", "make_embedder", "build_memory"]


def build_memory(data_dir: str = None) -> PagingManager:
    """Construct the memory stack from AUTOPC_* environment variables."""
    data_dir = data_dir or config.data_dir()
    os.makedirs(data_dir, exist_ok=True)

    embedder = make_embedder(
        kind=os.environ.get("AUTOPC_EMBEDDER", "onnx"),
        model=os.environ.get("AUTOPC_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        cache_dir=os.path.join(data_dir, ".embed_cache"),
    )
    vectors = VectorStore(os.path.join(data_dir, "vectors"), embedder=embedder)
    return PagingManager(vectors, data_dir=data_dir)
