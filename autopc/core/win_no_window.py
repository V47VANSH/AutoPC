"""Stop spawned console children from opening their own console window.

When AutoPC is built windowed (``console=False`` in the spec) it has no
console of its own. If it then spawns a *console* child process, Windows
allocates a brand-new visible console window for that child -- so the terminal
we removed from the parent simply reappears on every child. The one that matters
here is the SDK's ``claude.exe`` (a Node console app), which
``claude_agent_sdk`` spawns via ``anyio.open_process`` **without** passing
``creationflags``.

Fix: wrap ``anyio.open_process`` so it ORs in ``CREATE_NO_WINDOW`` on Windows.
anyio (>=4) forwards ``creationflags`` straight through to the subprocess, so the
child starts windowless. The SDK looks up ``anyio.open_process`` at call time, so
patching the module attribute is enough.

Call ``apply()`` (or just import this module) once, early, before the SDK spawns
anything. It is a no-op off Windows and safe to call more than once.
"""
import sys

_CREATE_NO_WINDOW = 0x08000000
_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED or sys.platform != "win32":
        return
    import anyio

    _orig_open_process = anyio.open_process

    async def _open_process_no_window(*args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
        return await _orig_open_process(*args, **kwargs)

    anyio.open_process = _open_process_no_window
    _PATCHED = True


apply()
