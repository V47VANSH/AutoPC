"""
Agent SDK tool surface -- the agent's hands and eyes.

The tools drive the desktop through `core/`:

    core/eyes.py         DXCam capture
    core/perception.py   control detection, OCR, window context
    core/executor.py     click snapping, occlusion check, dispatch, change report
    core/hands.py        Interception driver input

## How the model points

`observe` returns the screenshot plus every line of on-screen text tagged with
the pixel coordinate that clicks it. The model clicks text targets by reading
that coordinate rather than estimating it, and clicks everything else by
pointing at the image; a point that lands inside a small detected control is
snapped to its centre by the executor.

## Why observations are stateful

`_LAST` holds the observation the model was actually shown. Actions resolve
against it -- snapping uses its detected controls and change detection uses it
as the "before" frame -- so a click is judged against the picture the model
looked at, not a newer one.

## Dry run

With `DRY_RUN` set (`--dry-run` or `AUTOPC_DRY_RUN=1`), every action that would
touch the mouse or keyboard is reported instead of performed. The check lives in
`_dispatch`, which every input tool goes through, plus `read_screen`, which
scrolls on its own.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Annotated, Any, Optional

from claude_agent_sdk import create_sdk_mcp_server, tool

from autopc import config
from autopc.core.executor import Executor
from autopc.core.perception import Observation, PerceptionEngine
from autopc.core.reader import ScreenReader
from autopc.agent_brain.trace import TraceRecorder

# Wired by `attach()` before the agent runs.
_PERCEPTION: Optional[PerceptionEngine] = None
_EXECUTOR: Optional[Executor] = None
_READER: Optional[ScreenReader] = None

#: The separate Claude Opus reasoner that produces every answer and every line of
#: code. Wired by `attach_solver()`; see agent_brain/solver.py.
_SOLVER: Optional[Any] = None

#: The observation the model was last shown. See "Why observations are stateful".
_LAST: Optional[Observation] = None

#: The persistent memory stack (HOT working / WARM core / COLD archival). Wired
#: by `attach_memory()`. Unlike the conversation, this survives compaction and
#: restarts: learned facts live in MEMORY.md and episodic experiences in a
#: vector store, so `remember`/`recall`/`memory_search` return them intact across
#: runs and across whole sessions. See memory/paging_manager.py.
_MEMORY: Optional[Any] = None


def reset_notes() -> None:
    """Clear the working-memory scratchpad (recent actions).

    Only the HOT tier is wiped -- learned Core facts and Archival experiences
    persist on purpose, so the agent keeps what it has learned.
    """
    if _MEMORY is not None:
        _MEMORY.clear_working()


#: When True, input tools report what they would do instead of doing it.
#: `run.boot()` sets this from AUTOPC_DRY_RUN, which `--dry-run` sets.
DRY_RUN = config.env_flag("AUTOPC_DRY_RUN")


def attach(perception: PerceptionEngine, executor: Executor,
           reader: Optional[ScreenReader] = None) -> None:
    """Bind the live perception, executor and reader before running the agent."""
    global _PERCEPTION, _EXECUTOR, _READER
    _PERCEPTION, _EXECUTOR, _READER = perception, executor, reader


def attach_solver(solver: Any) -> None:
    """Bind the Opus solver (agent_brain/solver.py) before running the agent."""
    global _SOLVER
    _SOLVER = solver


def get_solver() -> Optional[Any]:
    """The attached solver (or None)."""
    return _SOLVER


def attach_memory(memory: Any) -> None:
    """Bind the persistent memory stack (memory.build_memory())."""
    global _MEMORY
    _MEMORY = memory


def memory_preamble() -> str:
    """The agent's current persistent memory, to prepend to a goal so it starts
    from what it already knows. Empty when no memory is attached."""
    if _MEMORY is None:
        return ""
    return _MEMORY.get_full_memory_context()


#: Records the (thought, action, outcome) steps of the current goal, so a
#: successful multi-step run can be crystallized into a reusable skill. See
#: agent_brain/trace.py. One recorder is reused per process; `start_trace`
#: clears it at the start of each goal.
_TRACE = TraceRecorder()


def start_trace(goal: str) -> None:
    """Begin recording a fresh goal's action trace."""
    _TRACE.start(goal)


def note_thought(text: str) -> None:
    """Attach the model's latest reasoning to its next recorded action."""
    _TRACE.note_thought(text)


def trace() -> TraceRecorder:
    """The current goal's trace recorder."""
    return _TRACE


#: Procedural memory -- the learned-skill manager and the crystallizer that
#: distills new skills from successful runs. Wired by `attach_skills()`.
_SKILLS: Optional[Any] = None
_CRYSTALLIZER: Optional[Any] = None
#: Skill names surfaced to the agent for the current goal, so we can grade them
#: by how the goal turned out.
_OFFERED_SKILLS: list = []


def attach_skills(manager: Any, crystallizer: Any = None) -> None:
    """Bind the skill manager and (optionally) the crystallizer."""
    global _SKILLS, _CRYSTALLIZER
    _SKILLS, _CRYSTALLIZER = manager, crystallizer


def get_memory() -> Optional[Any]:
    """The attached persistent memory stack (or None)."""
    return _MEMORY


def get_skills() -> Optional[Any]:
    """The attached skill manager (or None)."""
    return _SKILLS


def skills_preamble(goal: str) -> str:
    """Relevant learned skills for a goal, to prepend so the agent can replay
    them instead of re-reasoning. Records which skills were surfaced (for
    grading) and returns the full steps of the best match."""
    global _OFFERED_SKILLS
    _OFFERED_SKILLS = []
    if _SKILLS is None:
        return ""
    relevant = _SKILLS.find_relevant_skills(goal, top_k=3)
    if not relevant:
        return ""
    _OFFERED_SKILLS = [r["name"] for r in relevant]
    lines = ["## A skill you already learned may apply here",
             "Replay it, adapting each step to what is actually on screen. If it "
             "stops matching reality, abandon it and reason from the screen.",
             "", "Relevant skills:"]
    for r in relevant:
        rate = f"{r['success_rate']:.0%}" if r.get("times_used") else "new"
        lines.append(f"  - {r['name']}: {r['description']} [success={rate}, "
                     f"used {r.get('times_used', 0)}x]")
    body = _SKILLS.skill_view(relevant[0]["name"])
    if body:
        lines += ["", f"Full playbook for the best match "
                  f"({relevant[0]['name']}):", body,
                  "", "Use skill_view(name) to read any of the others."]
    return "\n".join(lines)


async def finish_goal() -> str:
    """Close out a goal: grade any surfaced skills by the outcome, and
    crystallize a new skill from a successful, novel, multi-step run. Called by
    the run loop after the message stream ends. Best-effort."""
    t = _TRACE
    notes = []
    try:
        if _SKILLS and _OFFERED_SKILLS and t.success is not None:
            for name in _OFFERED_SKILLS:
                _SKILLS.grade_skill(name, bool(t.success))
            notes.append(f"graded {len(_OFFERED_SKILLS)} skill(s) "
                         f"({'success' if t.success else 'failure'})")
            pruned = _SKILLS.prune_skills()
            if pruned:
                notes.append(f"pruned failing skill(s): {', '.join(pruned)}")
        if _CRYSTALLIZER and t.success and not _OFFERED_SKILLS:
            steps = t.meaningful_steps()
            if _CRYSTALLIZER.should_crystallize(steps, True):
                sk = await _CRYSTALLIZER.crystallize(t.goal, steps)
                if sk is not None:
                    notes.append(f"crystallized new skill: {sk.metadata.name}")
    except Exception as exc:  # learning must never break the result
        notes.append(f"skill learning skipped ({type(exc).__name__})")
    return "; ".join(notes) or "no skill changes"


async def warm_solver() -> None:
    """Warm the solver's Opus session so the first solve_* call isn't cold."""
    if _SOLVER is not None:
        try:
            await _SOLVER.warm()
        except Exception as exc:
            log = __import__("logging").getLogger("agent_brain")
            log.warning("solver warm failed (non-fatal): %s", exc)


#: The SDK talks to the CLI over stdio with a 1 MiB cap per JSON message, and
#: the whole tool result -- base64 image plus the text list plus framing --
#: has to fit. Exceeding it kills the run with "JSON message exceeded maximum
#: buffer size", not a graceful error.
_MAX_MESSAGE_BYTES = 1_048_576
_IMAGE_BUDGET = int(_MAX_MESSAGE_BYTES * 0.6)   # leave room for the text list


#: **Lossless WebP at native resolution.** Measured against a real 1920x1080
#: desktop frame, comparing bytes on the wire against a
#: decode-and-re-OCR fidelity check (149 reference text lines):
#:
#:     JPEG q85                443 KB base64   LOSSY: 22 of 149 lines lost
#:     JPEG q70 4:4:4          395 KB base64   LOSSY: 26 of 149 lines lost
#:     PNG level 6             388 KB base64   pixel-exact
#:     WebP lossless           184 KB base64   pixel-exact (max channel diff 0)
#:
#: A UI screenshot is thin glyph strokes, flat fills and hard edges -- the worst
#: case for JPEG's DCT ringing and chroma subsampling and the best case for
#: WebP's lossless predictive coding. So there is no size/accuracy trade-off to
#: make here: lossless WebP is both smaller and exact.
#:
#: Why bytes matter more than they look: the API is **stateless**, so every turn
#: re-uploads the entire conversation, including every screenshot taken so far.
#: Frame N is sent again on turn N+1, N+2, ... On a slow uplink the cost of one
#: screenshot is its size multiplied by the number of turns that follow it, which
#: is why a long run gets progressively slower. Token cost is unaffected either
#: way -- image tokens come from the pixel dimensions, not the byte count -- so
#: this is purely an upload win.
#:
#: Resolution stays NATIVE, which is what keeps accuracy at zero loss: the model
#: points in the image's pixel space, so a native frame maps 1:1 to the screen
#: and no downscale can blur a dense UI. Set AUTOPC_IMAGE_WIDTH only to force
#: a smaller frame.
_IMAGE_WIDTH = config.env_int("AUTOPC_IMAGE_WIDTH", 0) or None

#: Encoder effort for lossless WebP (0 fast .. 6 smallest). Measured: size
#: plateaus at 184 KB from method=2 upward, so anything above 2 buys nothing.
_WEBP_METHOD = config.env_int("AUTOPC_WEBP_METHOD", 4)

#: Quality for the LOSSY last-resort encode only -- reached solely when a frame
#: is so busy that even lossless WebP cannot fit the transport budget, which has
#: not been observed. The normal path never reads this.
_IMAGE_QUALITY = config.env_int("AUTOPC_IMAGE_QUALITY", 90)


#: Screen-pixels-per-image-pixel for the last observe frame. The model points
#: in the coordinate space of the image it was shown; if that image was
#: downscaled to fit the transport, click_point divides the model's x/y by this
#: to recover the true screen coordinate. 1.0 when the native frame fit.
_LAST_FRAME_SCALE: float = 1.0

#: Not done here, deliberately: skipping the upload when the screen "has not
#: changed". Measured against a live desktop, byte-identical consecutive frames
#: essentially never occur -- a blinking caret or a ticking clock is enough to
#: differ -- so an exact-match skip never fires. Making it fire requires a
#: perceptual threshold, and that trades away correctness: a frame that differs
#: by a few hundred pixels can be a validation error appearing, a spinner
#: finishing, or a checkbox toggling, and telling the model "nothing changed"
#: there is exactly the silent failure this codebase guards against elsewhere.


#: Pillow is normally built with libwebp, but degrade rather than crash a run if
#: this one is not. Probed once.
def _webp_available() -> bool:
    global _WEBP_OK
    if _WEBP_OK is None:
        try:
            from PIL import Image as _Im
            buf = io.BytesIO()
            _Im.new("RGB", (8, 8)).save(buf, format="WEBP", lossless=True)
            _WEBP_OK = True
        except Exception as exc:
            _WEBP_OK = False
            __import__("logging").getLogger("agent_brain").warning(
                "lossless WebP unavailable (%s); falling back to lossy JPEG, which "
                "is both larger on the wire and worse on small text.", exc)
    return _WEBP_OK


_WEBP_OK: Optional[bool] = None


def _encode_frame(image, prefer_width: Optional[int] = None) -> tuple[bytes, str, int]:
    """
    Encode a frame as small as possible on the wire with NO loss of detail.

    Lossless WebP at native resolution is the normal path: the model sees the
    screen's exact pixels (verified max channel diff 0) at 2.4x fewer bytes than
    lossy JPEG. See the note on `_IMAGE_WIDTH` for the measurements and for why
    upload bytes dominate on a slow uplink.

    Everything below the first attempt is a transport-cap safety net that has not
    been observed to trigger. At each width it tries lossless, then a mild lossy
    pass, before dropping to the next smaller width. Returns the encoded width so
    the caller can map the model's pixel coordinates back to screen space.
    """
    widths = []
    for w in (prefer_width, _IMAGE_WIDTH, 1400, 1100, 900, 700):
        if w and int(w) not in widths:
            widths.append(int(w))
    if not widths:
        widths = [image.width]

    def _render(width):
        frame = image
        if image.width > width:
            frame = image.resize((width, int(image.height * width / image.width)))
        return frame.convert("RGB")

    webp = _webp_available()

    def _attempt(width, lossless):
        frame = _render(width)
        buf = io.BytesIO()
        if webp and lossless:
            frame.save(buf, format="WEBP", lossless=True, method=_WEBP_METHOD)
            mime = "image/webp"
        elif webp:
            frame.save(buf, format="WEBP", quality=_IMAGE_QUALITY, method=_WEBP_METHOD)
            mime = "image/webp"
        else:
            # 4:4:4 -- never subsample chroma on UI text; it is what smears
            # coloured syntax highlighting and small glyphs into mush.
            frame.save(buf, format="JPEG", quality=_IMAGE_QUALITY, subsampling=0)
            mime = "image/jpeg"
        return buf.getvalue(), mime, frame.width

    # Shed QUALITY before RESOLUTION at every step. Resolution is what matters
    # most here: the model points in the image's pixel space, so a downscale both
    # multiplies its pointing error on the way back to screen coordinates and
    # shrinks small glyphs below legibility. A mild WebP lossy pass at native
    # size costs far less accuracy than a lossless one at 1100px.
    data, mime, out_w = b"", "image/webp", image.width
    for width in widths:
        for lossless in (True, False):
            if lossless and not webp:
                continue
            data, mime, out_w = _attempt(width, lossless)
            if len(base64.b64encode(data)) <= _IMAGE_BUDGET:
                return data, mime, out_w
    return data, mime, out_w  # smallest attempt; better than nothing


def _screen_text_block(scale: float, limit: int = 80) -> str:
    """The screen's OCR text, top-to-bottom, each line tagged with the pixel
    coordinate (in image space) to click it. Lets the model click text targets
    pixel-perfectly -- it reads the coordinate rather than estimating it."""
    if _PERCEPTION is None or _LAST is None:
        return ""
    try:
        lines = _PERCEPTION._ocr_lines_for(_LAST.screenshot)
    except Exception:
        return ""
    lines = [l for l in lines if (l.text or "").strip()]
    if not lines:
        return ""
    lines.sort(key=lambda l: (l.rect[1], l.rect[0]))
    out = []
    for l in lines[:limit]:
        cx = int((l.rect[0] + l.rect[2]) / 2 * scale)
        cy = int((l.rect[1] + l.rect[3]) / 2 * scale)
        out.append(f"  ({cx},{cy})  {l.text.strip()}")
    if len(lines) > limit:
        out.append(f"  ... {len(lines) - limit} more text lines")
    return "\n".join(out)


def _observe_text(obs: Observation) -> str:
    """Text half of an observation: window context + readable screen text."""
    s = _LAST_FRAME_SCALE or 1.0
    img_w = int(obs.screenshot.width * s)
    img_h = int(obs.screenshot.height * s)
    parts = [f"Screenshot is {img_w}x{img_h} pixels. All coordinates -- the ones "
             f"below and the x,y you pass to click_point/scroll -- are in this "
             f"pixel space, origin top-left."]
    if obs.foreground:
        parts.append(f"Active window: {obs.foreground.describe()}")
    txt = _screen_text_block(s)
    if txt:
        parts.append("Text currently on screen (each line with the pixel "
                     "coordinate that clicks it):\n" + txt)
    else:
        parts.append("(No text detected on screen -- read the image directly.)")
    return "\n\n".join(parts)


def _require_wired() -> Optional[dict]:
    if _PERCEPTION is None or _EXECUTOR is None:
        return {"content": [{"type": "text", "text":
                             "Perception/executor not attached. Call attach() first."}],
                "isError": True}
    return None


async def _off_loop(fn, *args, **kwargs):
    """
    Run a blocking perception/executor call on a worker thread.

    Two reasons, and the first one bites silently. `core/ocr.py` drives the
    Windows OCR API through `asyncio.run()`; calling that from inside an
    already-running event loop -- which is where every SDK tool handler
    executes -- raises "asyncio.run() cannot be called from a running event
    loop", and `ocr.py` swallows it as non-fatal. The result is a run where
    observe silently returns no screen text, degrading targeting for no
    visible reason. A worker thread has no running loop, so `asyncio.run()`
    works there.

    Second: perception takes a noticeable fraction of a second and the executor
    sleeps through `settle_time`. Blocking the event loop for that stalls the
    SDK's own I/O.
    """
    return await asyncio.to_thread(lambda: fn(*args, **kwargs))


# ---------------------------------------------------------------- perception


@tool(
    "observe",
    "Take a fresh screenshot and look at the screen the way a person would. "
    "Returns the current screen image plus the text visible on it, each line "
    "tagged with the pixel coordinate that clicks it. Decide what to do from the "
    "image, then act by pointing (click_point), typing, or scrolling. Call this "
    "before acting, and again after anything that changes the screen.",
    {},
)
async def observe(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    global _LAST, _LAST_FRAME_SCALE
    _LAST = await _off_loop(_PERCEPTION.observe)
    shot = _LAST.screenshot
    data, mime, out_w = _encode_frame(shot, prefer_width=_IMAGE_WIDTH or shot.width)
    _LAST_FRAME_SCALE = out_w / max(1, shot.width)
    img_block = {"type": "image",
                 "data": base64.b64encode(data).decode("ascii"),
                 "mimeType": mime}
    return {"content": [img_block, {"type": "text", "text": _observe_text(_LAST)}]}


@tool(
    "click_point",
    "Click the screen at a pixel coordinate with the physical mouse -- your main "
    "way to act on anything you can see. Point at the CENTRE of the target. For "
    "text (buttons, links, "
    "menu items, list rows, tabs) pass the exact coordinate shown next to that "
    "line in observe. For icons, checkboxes, or a spot inside a document or code "
    "editor, read the coordinate off the image. You can click empty space too -- "
    "e.g. to place the text cursor between two code lines -- it clicks exactly "
    "where you point.",
    {"x": Annotated[int, "Horizontal pixel position on the screenshot"],
     "y": Annotated[int, "Vertical pixel position on the screenshot"],
     "button": Annotated[str, "'left' (default) or 'right' for a context menu"],
     "double": Annotated[bool, "true to double-click, e.g. to open an item"]},
)
async def click_point(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    x, y = int(args["x"]), int(args["y"])
    s = _LAST_FRAME_SCALE or 1.0
    sx, sy = int(x / s), int(y / s)     # image space -> true screen pixels
    button = str(args.get("button", "left") or "left").lower()
    double = bool(args.get("double", False))
    return await _dispatch({"type": "click_xy", "x": sx, "y": sy,
                            "button": button, "double": double,
                            "snap": True, "label": f"({x},{y})"})


@tool(
    "hover_point",
    "Move the mouse over a pixel WITHOUT clicking. Reveals hover-only UI - "
    "tooltips, dropdown submenus that open on hover, previews - and lets you "
    "confirm you are over the right thing before you commit to a click. Point at "
    "the centre of the target, same coordinate space as click_point.",
    {"x": Annotated[int, "Horizontal pixel position on the screenshot"],
     "y": Annotated[int, "Vertical pixel position on the screenshot"]},
)
async def hover_point(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    x, y = int(args["x"]), int(args["y"])
    s = _LAST_FRAME_SCALE or 1.0
    return await _dispatch({"type": "click_xy", "x": int(x / s), "y": int(y / s),
                            "hover": True, "snap": True, "label": f"({x},{y})"})


@tool(
    "drag",
    "Press the physical mouse at one pixel, move to another with the button "
    "held down, and release -- a real click-and-drag. Use it to move a slider "
    "or scrollbar thumb, reorder a list item, drag a file/card/shape across the "
    "screen, select a range of text, or draw. Point start_x,start_y at where the "
    "drag begins and end_x,end_y at where it ends, both at the CENTRE of the "
    "target, in the same pixel space as click_point. For a slider or scrollbar, "
    "grab the thumb as the start and release at the track position you want.",
    {"start_x": Annotated[int, "Horizontal pixel where the drag begins"],
     "start_y": Annotated[int, "Vertical pixel where the drag begins"],
     "end_x": Annotated[int, "Horizontal pixel where the drag ends"],
     "end_y": Annotated[int, "Vertical pixel where the drag ends"]},
)
async def drag(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    # Image space -> true screen pixels, same mapping click_point/scroll use.
    s = _LAST_FRAME_SCALE or 1.0
    ix0, iy0 = int(args["start_x"]), int(args["start_y"])
    ix1, iy1 = int(args["end_x"]), int(args["end_y"])
    sx, sy = int(ix0 / s), int(iy0 / s)
    ex, ey = int(ix1 / s), int(iy1 / s)
    return await _dispatch({"type": "drag_xy",
                            "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
                            "label": f"({ix0},{iy0})->({ix1},{iy1})"})


# ------------------------------------------------------------------- dispatch


def _describe(action: dict) -> str:
    """One line saying what an action would do, for dry-run reports."""
    t = action.get("type")
    if t == "click_xy":
        if action.get("hover"):
            verb = "hover over"
        else:
            verb = ("double-click" if action.get("double")
                    else f"{action.get('button', 'left')}-click")
        return f"{verb} screen ({action['x']},{action['y']})"
    if t == "drag_xy":
        return (f"drag screen ({action['start_x']},{action['start_y']}) -> "
                f"({action['end_x']},{action['end_y']})")
    if t == "scroll":
        return (f"scroll {action.get('direction', 'down')} x{action.get('amount', 3)} "
                f"at screen ({action.get('x')},{action.get('y')})")
    if t == "type":
        return f"type {len(action.get('text', ''))} chars: {action.get('text', '')[:60]!r}"
    if t == "type_code":
        return (f"type {len(action.get('text', ''))} chars of code "
                f"(replace={action.get('replace', True)})")
    if t == "navigate":
        return f"navigate to {action.get('url', '')}"
    if t == "press":
        return f"press {action.get('key')}"
    if t == "hotkey":
        return f"press {'+'.join(action.get('keys', []))}"
    return f"perform {t}"


async def _dispatch(action: dict) -> dict[str, Any]:
    """
    Run one action against the observation the model was actually shown, and
    report what measurably happened. In a dry run, report the action instead.

    The change signal is returned verbatim rather than interpreted. "The screen
    changed" and "the right thing happened" are different claims, and only the
    model looking at the next observation can settle the second one -- so this
    reports the first and does not editorialise.
    """
    if _LAST is None:
        return {"content": [{"type": "text", "text":
                             "No observation yet -- call observe first."}], "isError": True}
    if DRY_RUN:
        _remember_step(action, "dry run (not performed)", ok=True)
        return _text(f"DRY RUN -- would {_describe(action)}. Nothing was touched; the "
                     "screen is unchanged.")
    result = await _off_loop(_EXECUTOR.execute, action, _LAST)
    if not result.ok:
        _remember_step(action, f"FAILED: {result.detail}", ok=False)
        return {"content": [{"type": "text", "text": f"FAILED: {result.detail}"}],
                "isError": True}
    bits = [result.detail]
    if result.screen_changed is not None:
        bits.append(f"screen {'changed' if result.screen_changed else 'UNCHANGED'} "
                    f"({result.change_ratio:.1%} of pixels)")
    if result.window_before != result.window_after:
        bits.append(f"active window: '{result.window_before}' -> '{result.window_after}'")
    _remember_step(action, result.detail, ok=True)
    bits.append("Call observe to see the result before deciding what is next.")
    return {"content": [{"type": "text", "text": " | ".join(bits)}]}


def _remember_step(action: dict, outcome: str, ok: bool) -> None:
    """Record an executed action to the goal trace and the HOT working tier.

    The trace feeds skill crystallization; the working tier makes the action
    visible to `recall`. Both are best-effort -- a bookkeeping failure must
    never break the action that just succeeded.
    """
    try:
        step = _TRACE.record(action, outcome, ok=ok)
        if _MEMORY is not None:
            _MEMORY.add_to_working({"thought": step["thought"],
                                    "action": action, "outcome": outcome})
    except Exception:  # pragma: no cover - telemetry must not break a run
        pass


@tool(
    "read_screen",
    "Scroll a pane top-to-bottom and read ALL of its text, stitched into one "
    "block. Use this to capture the text of anything that continues below the "
    "fold - an article, a document, instructions, a long form or settings page. "
    "Acting on the visible portion alone misses whatever is below it. Point x,y "
    "at any content inside the pane; wheel events go to whatever the pointer "
    "sits over, so pointing at the wrong pane reads the wrong thing. This tool "
    "is for READING, not for positioning - after it, YOU decide where to go: "
    "the control you need can be anywhere in the pane, so `observe` and then "
    "`scroll` up or down to bring it into view. By "
    "default this returns the pane to the top when done (a clean starting "
    "point); pass return_to_top=false to leave it at the bottom when what you "
    "need is down there.",
    {"x": Annotated[int, "Horizontal pixel inside the pane to read (0 = current pointer)"],
     "y": Annotated[int, "Vertical pixel inside the pane to read (0 = current pointer)"],
     "max_scrolls": Annotated[int, "How far to scroll while reading (default 10)"],
     "return_to_top": Annotated[bool, "After reading, scroll back to the top of the "
                                "pane (default true). Set false to stay at the bottom."]},
)
async def read_screen(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    if _READER is None:
        return {"content": [{"type": "text", "text": "Reader not attached."}],
                "isError": True}
    x, y = int(args.get("x") or 0), int(args.get("y") or 0)
    hover_point = None
    if x and y:
        s = _LAST_FRAME_SCALE or 1.0
        hover_point = (int(x / s), int(y / s))
    return_to_top = args.get("return_to_top", True)
    if isinstance(return_to_top, str):
        return_to_top = return_to_top.strip().lower() not in ("0", "false", "no")
    if DRY_RUN:
        # Reading scrolls the pane, so a dry run reads only what is visible.
        visible = _screen_text_block(_LAST_FRAME_SCALE or 1.0) if _LAST else ""
        return _text("DRY RUN -- would scroll the pane and read it; not scrolling. "
                     "Visible text only:\n" + (visible or "(no text visible)"))
    result = await _off_loop(_READER.read_full, hover_point=hover_point,
                             max_scrolls=int(args.get("max_scrolls", 10) or 10),
                             return_to_top=bool(return_to_top))
    if not result.text:
        return {"content": [{"type": "text", "text":
                             "No text could be read (OCR unavailable, or nothing detected)."}]}
    note = ("\n\n[TRUNCATED - more content remains below; read again from further down "
            "if you need it]" if result.truncated else "")
    return {"content": [{"type": "text", "text":
                         f"{result.summary()}:\n\n{result.text}{note}"}]}


@tool(
    "type_text",
    "Type text into whatever currently holds keyboard focus. To go to a URL, "
    "use `navigate` instead -- typing a URL only works if the address bar "
    "already has focus, and pages routinely steal focus back.",
    {"text": Annotated[str, "The literal text to type"]},
)
async def type_text(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    return await _dispatch({"type": "type", "text": args["text"]})


@tool(
    "type_code",
    "Type CODE into a focused code editor. Use this instead of type_text "
    "whenever you are entering code into an editor (VS Code, a browser-based "
    "editor, or a notebook cell) -- anywhere the editor auto-indents "
    "on Enter, auto-closes brackets/quotes, or pops up suggestions. It types on "
    "the physical keyboard, but neutralizes those editor "
    "behaviours so your indentation and brackets land exactly as written -- no "
    "doubled indentation, no duplicated `)` or `}`, no suggestion accepted in "
    "place of a newline. Pass the code with its real newlines and indentation. "
    "By default it FIRST selects-all and clears the editor, so the code replaces "
    "any template/boilerplate already there -- click into the editor and confirm "
    "it is focused before calling this. Pass `language` whenever you know it (the "
    "editor's language selector tells you): it makes the typing markedly faster "
    "in Java/C++/JS-style languages by letting the editor do the indenting, while "
    "keeping indentation byte-exact in Python and other whitespace-sensitive "
    "languages. For plain text (search boxes, prose, a single value) use "
    "type_text instead.",
    {"code": Annotated[str, "The code to type, with real newlines and indentation"],
     "language": Annotated[str, "The editor's language, e.g. 'java', 'python', "
                           "'cpp', 'javascript'. Optional but recommended."],
     "replace": Annotated[bool, "Clear the editor first so the code replaces existing "
                          "content (default true). Set false only if the editor is "
                          "already empty and focused."]},
)
async def type_code(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    return await _dispatch({"type": "type_code", "text": args["code"],
                            "language": str(args.get("language", "") or ""),
                            "replace": bool(args.get("replace", True))})


@tool(
    "navigate",
    "Go to a URL in the focused browser: focuses the address bar, replaces its "
    "contents, types the URL and presses enter -- all in one uninterruptible "
    "step. Always use this rather than a separate address-bar shortcut followed "
    "by typing; seconds pass between two tool calls and a page can take focus "
    "back in that window, which sends your URL into a search box on the page.",
    {"url": Annotated[str, "The full URL to open"]},
)
async def navigate(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    return await _dispatch({"type": "navigate", "url": args["url"]})


@tool(
    "press_key",
    "Press a single named key (enter, tab, escape, f5, arrows) or a chord. "
    "Pass one key name, or several to press together as a shortcut.",
    {"keys": Annotated[str, "One key name, or a chord joined by '+' e.g. 'ctrl+s'"]},
)
async def press_key(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    keys = [k.strip().lower() for k in str(args["keys"]).split("+") if k.strip()]
    if not keys:
        return {"content": [{"type": "text", "text": "No key given."}], "isError": True}
    action = ({"type": "press", "key": keys[0], "count": 1} if len(keys) == 1
              else {"type": "hotkey", "keys": keys})
    return await _dispatch(action)


@tool(
    "scroll",
    "Scroll the pane under a point to bring off-screen content into view, then "
    "observe again. Scrolls in any direction: 'up'/'down' for content above or "
    "below, and 'left'/'right' for content off the sides -- use the sideways "
    "scroll when a code line runs past the right edge, or a wide table or "
    "timeline is cut off. Point x,y at the pane you want to scroll -- Windows "
    "scrolls whatever sits under the pointer, so the wrong spot scrolls the "
    "wrong pane.",
    {"x": Annotated[int, "Horizontal pixel over the pane to scroll"],
     "y": Annotated[int, "Vertical pixel over the pane to scroll"],
     "direction": Annotated[str, "'up', 'down', 'left', or 'right'"],
     "amount": Annotated[int, "Wheel notches; 3-5 is about a screenful"]},
)
async def scroll(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_wired()
    if err:
        return err
    s = _LAST_FRAME_SCALE or 1.0
    return await _dispatch({"type": "scroll",
                            "x": int(int(args["x"]) / s),
                            "y": int(int(args["y"]) / s),
                            "direction": str(args.get("direction", "down")).lower(),
                            "amount": int(args.get("amount", 3))})


# --------------------------------------------------------------- the solver
# This does not touch the desktop. It hands a self-contained sub-problem to
# the separate Claude Opus session and returns the answer, keeping deep
# reasoning off the fast perception loop.


async def _require_solver() -> Optional[dict]:
    if _SOLVER is None:
        return {"content": [{"type": "text", "text":
                 "Solver not attached. The Opus reasoner is unavailable."}],
                "isError": True}
    return None


@tool(
    "solve",
    "Hand a hard, self-contained sub-problem to the reasoning model (Claude "
    "Opus) and get its answer back. Use it for a focused piece of reasoning, "
    "or a tricky bit of code you would rather have worked out carefully off "
    "the perception loop. It cannot see the screen, so describe the problem "
    "in full and pass any constraints or starter code as context; then act "
    "on the reply yourself.",
    {"problem": Annotated[str, "The full, self-contained problem or question"],
     "context": Annotated[str, "Relevant constraints, data, or starter code (optional)"]},
)
async def solve(args: dict[str, Any]) -> dict[str, Any]:
    err = await _require_solver()
    if err:
        return err
    from autopc.agent_brain.solver import build_solve_prompt
    reply = await _SOLVER.ask(build_solve_prompt(str(args.get("problem", "")),
                                                 str(args.get("context", ""))))
    return {"content": [{"type": "text", "text": reply or "(no answer returned)"}]}


# ----------------------------------------------------------- persistent memory
# These write to the three-tier memory stack (memory/paging_manager.py), so what
# the agent learns here OUTLIVES the conversation AND the process: Core facts go
# to MEMORY.md, episodic experiences to a vector store, both read back on the
# next run. This is why the agent gets better at a recurring task over time.


def _require_memory() -> Optional[dict]:
    if _MEMORY is None:
        return {"content": [{"type": "text", "text":
                 "Memory not attached. Call attach_memory() first."}],
                "isError": True}
    return None


@tool(
    "remember",
    "Save a fact to persistent memory that you must keep for the rest of a task "
    "-- and future tasks. Most importantly details the task depends on that will "
    "scroll out of view (names, values, file paths, steps still to do), and "
    "durable facts about an app or the user's preferences (where a control "
    "lives, that they use dark mode). Stored outside your conversation, so it "
    "survives screens scrolling away, context compaction, AND restarts. Reuse a "
    "key to update it; near-duplicate facts are merged automatically.",
    {"key": Annotated[str, "A short, stable label, e.g. 'invoice_total' or 'edge_menu'"],
     "note": Annotated[str, "The fact to store, in full"]},
)
async def remember(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_memory()
    if err:
        return err
    key = str(args.get("key", "")).strip() or "note"
    _MEMORY.memory_save(key, str(args.get("note", "")))
    return {"content": [{"type": "text", "text":
             f"Saved '{key}' to persistent memory ({len(_MEMORY.core_memory)} "
             f"core facts held)."}]}


@tool(
    "recall",
    "Return your persistent memory -- recent actions (working), and every learned "
    "fact still in Core (task scope/rules, app knowledge, user preferences). Call "
    "this when you start a new task or section to reload what you know. For older "
    "experiences not shown here, use memory_search.",
    {},
)
async def recall(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_memory()
    if err:
        return err
    return {"content": [{"type": "text", "text": _MEMORY.get_full_memory_context()}]}


@tool(
    "memory_search",
    "Semantic search over your ARCHIVAL memory -- past experiences and facts you "
    "archived or that scrolled out of Core. Use it when you suspect you have "
    "solved something like this before, or need a fact you no longer see in "
    "recall. Returns the closest matches with a relevance score.",
    {"query": Annotated[str, "What you are trying to remember, in plain words"]},
)
async def memory_search(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_memory()
    if err:
        return err
    hits = _MEMORY.memory_search(str(args.get("query", "")), top_k=5)
    if not hits:
        return {"content": [{"type": "text", "text": "No matching memories found."}]}
    body = "\n".join(f"  [{h['score']:.2f}] {h['text']}" for h in hits)
    return {"content": [{"type": "text", "text": "Archival matches:\n" + body}]}


@tool(
    "memory_archive",
    "Move a Core fact to Archival storage to free up working context, when it is "
    "no longer immediately relevant but you may want it later (memory_search finds "
    "it, memory_recall brings it back).",
    {"key": Annotated[str, "The Core memory key to archive"]},
)
async def memory_archive(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_memory()
    if err:
        return err
    _MEMORY.memory_archive(str(args.get("key", "")))
    return {"content": [{"type": "text", "text":
             f"Archived '{args.get('key', '')}' (if it existed)."}]}


@tool(
    "memory_recall",
    "Bring a previously archived fact back into Core memory by its exact key.",
    {"key": Annotated[str, "The archived key to restore to Core memory"]},
)
async def memory_recall(args: dict[str, Any]) -> dict[str, Any]:
    err = _require_memory()
    if err:
        return err
    value = _MEMORY.memory_recall(str(args.get("key", "")))
    if value is None:
        return {"content": [{"type": "text", "text":
                 f"No archived fact under key '{args.get('key', '')}'."}]}
    return {"content": [{"type": "text", "text":
             f"Recalled '{args.get('key', '')}' into Core: {value}"}]}


# ------------------------------------------------------------------- skills


@tool(
    "skill_view",
    "Read the full step-by-step playbook of a skill you have learned, by name. "
    "The relevant skills for this goal are listed at the top of your task; the "
    "best match's full steps are already shown there, so use this to read any of "
    "the OTHERS before deciding which to replay.",
    {"name": Annotated[str, "The skill name, exactly as listed"]},
)
async def skill_view(args: dict[str, Any]) -> dict[str, Any]:
    if _SKILLS is None:
        return {"content": [{"type": "text", "text": "No skills are available."}]}
    body = _SKILLS.skill_view(str(args.get("name", "")))
    if not body:
        return {"content": [{"type": "text", "text":
                 f"No skill named '{args.get('name', '')}'."}]}
    return {"content": [{"type": "text", "text": body}]}


# ------------------------------------------------------------- task control
# The task layer (tasks/) gives every goal a ledger: an immutable goal, a
# definition of done, a plan, and a progress journal on disk. These tools are
# how the agent reads and maintains it. The runner binds the current attempt
# with `bind_attempt()`; without one they return a harmless "no task" message.

#: The live attempt (tasks/state.AttemptState) -- ledger + per-attempt counters.
_ATTEMPT: Optional[Any] = None


def bind_attempt(state: Any) -> None:
    """Bind the current task attempt so the task tools and the guard hooks
    operate on the same ledger. Pass None to unbind."""
    global _ATTEMPT
    _ATTEMPT = state


def current_attempt() -> Optional[Any]:
    return _ATTEMPT


def _no_task() -> dict:
    return {"content": [{"type": "text", "text":
             "No task ledger is active for this run."}]}


def _text(msg: str, error: bool = False) -> dict:
    out = {"content": [{"type": "text", "text": msg}]}
    if error:
        out["isError"] = True
    return out


@tool(
    "task_status",
    "Read your task ledger: the goal (your ONLY objective), the definition of done "
    "with each criterion's status, your plan, recent progress, blockers, and lessons "
    "from earlier attempts. Call it whenever you are unsure what is left, after a "
    "long stretch of work, or right after resuming.",
    {"type": "object", "properties": {}},
)
async def task_status(args: dict[str, Any]) -> dict[str, Any]:
    if _ATTEMPT is None:
        return _no_task()
    return _text(_ATTEMPT.ledger.report())


@tool(
    "update_plan",
    "Write or revise your step-by-step plan for the task. Send the WHOLE plan each "
    "time, in order. Keep exactly one step `in_progress` -- the one you are doing "
    "now -- and mark steps `done` as you finish them, `blocked` if stuck. Revise "
    "the plan when reality differs from it; never drop a step the goal requires.",
    {"type": "object",
     "properties": {"items": {"type": "array", "items": {
         "type": "object",
         "properties": {"text": {"type": "string"},
                        "status": {"type": "string",
                                   "enum": ["pending", "in_progress", "done", "blocked"]}},
         "required": ["text", "status"]}}},
     "required": ["items"]},
)
async def update_plan(args: dict[str, Any]) -> dict[str, Any]:
    if _ATTEMPT is None:
        return _no_task()
    L = _ATTEMPT.ledger
    L.set_plan(args.get("items") or [])
    marks = {"done": "[x]", "in_progress": "[>]", "blocked": "[!]", "pending": "[ ]"}
    body = "\n".join(f"  {marks.get(p.status, '[ ]')} {p.text}" for p in L.plan)
    return _text(f"Plan saved:\n{body}")


@tool(
    "mark_criterion",
    "Record your own assessment of one criterion of the definition of done, with "
    "the on-screen evidence (what exactly you see that proves it). This is your "
    "claim; completion is still checked independently against the screen.",
    {"type": "object",
     "properties": {"id": {"type": "string", "description": "criterion id, e.g. c1"},
                    "met": {"type": "boolean"},
                    "evidence": {"type": "string",
                                 "description": "what on screen shows it (or why not)"}},
     "required": ["id", "met", "evidence"]},
)
async def mark_criterion(args: dict[str, Any]) -> dict[str, Any]:
    if _ATTEMPT is None:
        return _no_task()
    L = _ATTEMPT.ledger
    cid = str(args.get("id", "")).strip()
    if not L.mark_criterion(cid, bool(args.get("met")), str(args.get("evidence", ""))):
        ids = ", ".join(c.id for c in L.criteria) or "none"
        return _text(f"No criterion '{cid}'. Valid ids: {ids}.", error=True)
    left = [c.id for c in L.unmet()]
    return _text(f"Marked {cid}. " + (f"Still open: {', '.join(left)}." if left
                                       else "All criteria marked met -- confirm on screen, "
                                            "then call complete_task(success=true)."))


@tool(
    "log_progress",
    "Checkpoint your progress in the task journal: what is now done, what you "
    "learned, where things stand. The journal survives restarts and context "
    "compaction, so a fresh session can pick up exactly here. Log after each "
    "milestone, not after every click.",
    {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
)
async def log_progress(args: dict[str, Any]) -> dict[str, Any]:
    if _ATTEMPT is None:
        return _no_task()
    _ATTEMPT.ledger.log(str(args.get("note", "")))
    return _text("Checkpoint saved.")


@tool(
    "report_blocker",
    "Record something that is blocking the task. Set needs_user=true ONLY when the "
    "obstacle is something only the user can resolve -- a password or 2FA code you "
    "were not given, a permission prompt that needs them, missing information that "
    "cannot be found on the machine, a physical action. Ordinary difficulties (a "
    "control you cannot find, an unexpected dialog) are not user blockers: record "
    "them and keep working around them.",
    {"type": "object",
     "properties": {"description": {"type": "string"},
                    "needs_user": {"type": "boolean"}},
     "required": ["description"]},
)
async def report_blocker(args: dict[str, Any]) -> dict[str, Any]:
    if _ATTEMPT is None:
        return _no_task()
    needs_user = bool(args.get("needs_user", False))
    _ATTEMPT.ledger.add_blocker(str(args.get("description", "")), needs_user=needs_user)
    if needs_user:
        return _text("Blocker recorded as needing the user. If nothing else in the task can "
                     "proceed without them, call complete_task(success=false) with what they "
                     "need to provide.")
    return _text("Blocker recorded. Work around it -- try a different route to the same "
                 "outcome.")


async def _screen_evidence() -> tuple:
    """A FRESH observation for verification: screenshot, window title, OCR text.
    Deliberately not stored in _LAST -- the model has not seen this frame, so
    pointer tools must keep resolving against the one it has."""
    obs = await _off_loop(_PERCEPTION.observe)
    shot = obs.screenshot
    window = getattr(obs.foreground, "title", "") if obs.foreground else ""
    text = ""
    try:
        lines = await _off_loop(_PERCEPTION._ocr_lines_for, shot)
        lines = [l for l in lines if (l.text or "").strip()]
        lines.sort(key=lambda l: (l.rect[1], l.rect[0]))
        text = "\n".join(l.text.strip() for l in lines[:150])
    except Exception:
        pass
    return shot, window, text


#: Giving up is accepted on the second request in an attempt (the first gets a
#: "try another approach" pushback), or immediately with a needs-user blocker.
FAILURE_CLAIMS_TO_GIVE_UP = 2

_GIVE_UP_PUSHBACK = (
    "NOT ACCEPTED YET. Completing this task is your only objective, and giving up "
    "is a last resort. Before giving up:\n"
    "  1. Call task_status and re-read the goal and the definition of done.\n"
    "  2. List the approaches you have NOT tried yet: keyboard shortcuts, a different "
    "menu or app, the Start-menu search, scrolling to hidden controls, closing a "
    "dialog that is in the way, reopening the app, searching the web for how.\n"
    "  3. Try the most promising one now.\n"
    "If the task is blocked by something only the user can provide (a password, a "
    "permission, missing information, a physical action), call "
    "report_blocker(description, needs_user=true) and then complete_task(success=false)."
)


@tool(
    "complete_task",
    "Claim the task is finished. With success=true your claim is INDEPENDENTLY "
    "VERIFIED against a fresh screenshot and the definition of done -- if anything "
    "is not visibly done you will be told exactly what, and you keep working. Only "
    "call it once every criterion is met and visible. success=false is for a task "
    "you genuinely cannot finish; see report_blocker.",
    {"type": "object",
     "properties": {"success": {"type": "boolean"},
                    "summary": {"type": "string",
                                "description": "what you did and the on-screen evidence, "
                                               "or exactly what blocks you"}},
     "required": ["success", "summary"]},
)
async def complete_task(args: dict[str, Any]) -> dict[str, Any]:
    success = bool(args.get("success", False))
    summary = str(args.get("summary", ""))

    if _ATTEMPT is None:
        return _no_task()

    A, L = _ATTEMPT, _ATTEMPT.ledger
    if success:
        err = _require_wired()
        if err:
            return err
        from autopc.tasks.verifier import decide, record, verify
        shot, window, text = await _screen_evidence()
        verdict = await verify(L, screenshot=shot, screen_text=text, window=window,
                               claim=summary,
                               model=os.environ.get("AUTOPC_VERIFIER_MODEL") or None)
        accepted, msg = decide(verdict, L, A.uncertain_rejections)
        record(L, verdict, accepted)
        if accepted:
            L.finish("verified", summary)
            _TRACE.complete(True, summary)
            return _text(msg + "\nThe task is complete. End your turn now with a one-line "
                         "summary -- take no further actions.")
        A.rejected_claims += 1
        if verdict.overall == "uncertain":
            A.uncertain_rejections += 1
        L.save()
        return _text(msg)

    # success == false: a give-up request.
    A.failure_claims += 1
    if L.has_user_blocker():
        L.finish("needs_user", summary)
        _TRACE.complete(False, summary)
        return _text("Recorded: this task needs the user before it can continue. "
                     "End your turn now with what they need to provide.")
    if A.failure_claims >= FAILURE_CLAIMS_TO_GIVE_UP:
        A.gave_up = True
        L.log(f"Agent gave up this attempt: {summary}")
        _TRACE.complete(False, summary)
        return _text("Recorded. This attempt will end; what you tried and why it failed will be "
                     "reviewed before the next attempt. End your turn now.")
    L.log(f"Give-up request pushed back: {summary}")
    return _text(_GIVE_UP_PUSHBACK)


#: The desktop surface: look, then point/type/scroll, plus the `solve` tool that
#: hands hard thinking to the reasoning model, the persistent memory tools
#: (remember/recall + archival paging), the task-control tools that keep a long
#: task anchored to its definition of done, and complete_task -- a verified
#: completion gate.
DESKTOP_TOOLS = [observe, click_point, hover_point, drag, type_text, type_code,
                 press_key, scroll, navigate, read_screen,
                 solve,
                 remember, recall, memory_search, memory_archive, memory_recall,
                 skill_view,
                 task_status, update_plan, mark_criterion, log_progress,
                 report_blocker, complete_task]

#: Tool names as the model sees them once the SDK namespaces the MCP server.
ALLOWED_TOOL_NAMES = [f"mcp__desktop__{t.name}" for t in DESKTOP_TOOLS]


def desktop_server():
    """The in-process MCP server exposing the desktop to the agent."""
    return create_sdk_mcp_server(name="desktop", version="0.1.0", tools=DESKTOP_TOOLS)
