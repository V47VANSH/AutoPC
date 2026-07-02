"""
Paths, environment loading and model defaults shared by the whole app.

Every setting is an `AUTOPC_*` environment variable, optionally loaded from a
`.env.local` file (see `.env.template`). Nothing here reads a variable at import
time, so a CLI flag that sets one before the agent boots always takes effect.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: The driver: drives the desktop turn by turn.
DEFAULT_MODEL = "claude-sonnet-5-5"
#: The off-hot-path reasoner: solve, contract, verify, reflect, skill distillation.
DEFAULT_SOLVER_MODEL = "claude-opus-5-5"

_TRUE = ("1", "true", "yes", "on")
_env_loaded = False


def app_dir() -> Path:
    """Where AutoPC keeps its files when nothing else is configured.

    The folder holding the exe for a PyInstaller build, the repository root for a
    source checkout, and a per-user folder for a plain `pip install`.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists():
        return root
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "AutoPC"


def data_dir() -> str:
    """Learned state: core memory, skills, task ledgers and the vector index."""
    return os.environ.get("AUTOPC_DATA_DIR") or str(app_dir() / "data")


def load_env() -> None:
    """Load `.env.local` from the working directory, then from `app_dir()`.

    Variables already set in the environment always win, so a value exported in
    the shell (or set by a CLI flag) overrides the file.
    """
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    from dotenv import load_dotenv
    for folder in (Path.cwd(), app_dir()):
        path = folder / ".env.local"
        if path.is_file():
            load_dotenv(path, override=False)


def env_int(name: str, default: int) -> int:
    """An int setting; unset, blank or malformed values fall back to `default`."""
    try:
        return int((os.environ.get(name) or "").strip())
    except ValueError:
        return default


def env_flag(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return default if not raw else raw in _TRUE


def model() -> str:
    return os.environ.get("AUTOPC_MODEL") or DEFAULT_MODEL


def solver_model() -> str:
    return os.environ.get("AUTOPC_SOLVER_MODEL") or DEFAULT_SOLVER_MODEL
