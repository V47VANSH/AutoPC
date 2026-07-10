"""
Run a desktop goal through the Claude Agent SDK.

    python -m autopc.agent_brain.run "open notepad and type hello"

`main.py` runs the same hardware on its own; this module wires it up to the
Agent SDK instead, which brings the agent loop, context compaction, retries and
session state, with the desktop handed to it as in-process MCP tools (see
`agent_brain/tools.py`).

Input is live by default. Pass --dry-run (or set AUTOPC_DRY_RUN=1) to report
each action without performing it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# DPI awareness must be declared before dxcam loads or any window exists, or
# the capture grid and the click grid disagree.
from autopc.core.display import assert_coordinate_spaces_agree, ensure_dpi_aware

ensure_dpi_aware()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env.local"))

from typing import Optional

# Suppress the console window Windows would otherwise allocate for spawned
# console children (esp. the SDK's claude.exe). Must load before the SDK spawns
# anything -- importing it applies the patch.
from autopc.core import win_no_window  # noqa: F401

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
    SystemMessage, TextBlock, ToolUseBlock, query,
)

from autopc import config
from autopc.agent_brain import tools as desktop
from autopc.core.executor import Executor
from autopc.core.eyes import DXCamEyes
from autopc.core.hands import InterceptionHands
from autopc.core.perception import PerceptionEngine
from autopc.core.reader import ScreenReader
from autopc.memory import build_memory

log = logging.getLogger("agent_brain")

SYSTEM_PROMPT = """\
You operate a real Windows desktop the way a person does - you look at the
screen, then move a physical mouse and type on a physical keyboard. You are not
automating an API; you are using the computer by sight and hand.

## Your senses and hands

- `observe` - take a screenshot and look. You get the screen image plus every
  line of text on it, each tagged with the pixel coordinate that clicks it.
  Read it like a person would: find what you want, then act.
- `click_point(x, y)` - click exactly at that pixel with the real mouse. This is
  your main action. For text (buttons, links, menu items, list rows, tabs) use
  the coordinate printed next to that line in `observe`. For icons, checkboxes,
  or a spot inside a document/editor, read the coordinate off the image. You can
  `button="right"` for a context menu, `double=true` to open something, and you
  may click empty space - e.g. to drop the text cursor between two lines.
- `hover_point(x, y)` - move the mouse over a pixel without clicking, to reveal
  tooltips and hover-open submenus, or to confirm you are on the right target.
- `drag(start_x, start_y, end_x, end_y)` - press, move with the button held,
  release: sliders, reordering, selecting a range of text.
- `scroll(x, y, direction, amount)` - wheel-scroll the pane under that point to
  bring off-screen content into view, then `observe` again. Direction is `up`,
  `down`, `left`, or `right` - scroll sideways when a line, table, or timeline
  runs off the edge of the pane.
- `type_text` - type plain text into whatever holds keyboard focus (a search
  box, a form field, prose, a single value).
- `type_code` - type CODE into a code editor. Use this instead of `type_text`
  whenever what you are entering is code and the target is an editor that
  auto-indents, auto-closes brackets, or shows suggestions. It stops the editor
  from doubling your indentation or duplicating your brackets. Pass `language`
  when you know it. See "Typing code into an editor".
- `press_key` - a key or chord (`enter`, `tab`, `escape`, `ctrl+a`, `ctrl+s`).
- `navigate(url)` - go to a URL in the focused browser.
- `read_screen(x, y)` - scroll a pane top-to-bottom and return ALL its text,
  for content taller than the viewport.
- `solve(problem, context)` - hand a hard, self-contained sub-problem to the
  reasoning model and read its answer back, so deep thinking stays off the fast
  loop. Describe the problem in full; it cannot see the screen.
- `remember(key, note)` / `recall()` - your PERSISTENT memory. `remember` saves a
  fact that must outlive the moment: details the task depends on that will
  scroll out of view, or durable knowledge about an app or the user (where a
  control lives, that they prefer dark mode). `recall()` returns your recent
  actions plus every saved fact. This memory survives screens scrolling away,
  context compaction, AND restarts - so save what you learn once and you (and
  future runs) keep it.
- `memory_search(query)` - semantic search over older/archived memories, for when
  you suspect you have done something like this before. `memory_archive(key)` and
  `memory_recall(key)` page a fact out of and back into active memory.

Coordinates are in the pixel space of the screenshot `observe` gave you. You
point where you look, and the mouse lands there precisely.

## Pointing well

Aim for the centre of what you want. When the target has text, the coordinate is
handed to you in the observe list - use it rather than eyeballing. When it does
not (an icon, a colour swatch, a calendar cell), read the point off the image.
A click just inside a small icon is nudged onto its centre for you; a click in
open space or inside text lands exactly where you put it, so you can place a
cursor precisely.

## A changed screen is not a successful action

After each action you are told whether the screen changed and by how much. That
tells you *something* happened, not that the *right* thing happened - a click
that misses and opens an unrelated page repaints most of the screen. Call
`observe` and confirm you are where you intended before continuing.

If the screen did not change at all, the action did nothing: it landed on
nothing, on a disabled control, or the app ignored it. Do not repeat it.

## When you are stuck, work out which kind of stuck

**Wrong target** - right screen, but your clicks keep missing. Look again at the
observe image and aim more carefully; use the exact coordinate from the text
list when the target has a label, or `scroll` to bring it fully into view.

**Wrong state** - you are not where you thought. An earlier action took you
somewhere unintended: a search results page instead of a site, a dialog, a
different app. Look at where you actually are and either use it (if it still
leads to the goal) or back out - `alt+left`, `escape` - and take a different
route.

The tell for wrong state: your actions "work" (the screen changes) but nothing
gets closer to the goal. Repeating a failed approach more carefully is the one
thing guaranteed not to help.

## Filling in forms, fields and editors

Entering text in the right place is where desktop work most often goes wrong.

**1. Map the screen first.** Call `observe` and locate the field or editor the
text goes into, and the control that saves or sends it (Save, Send, OK, Apply).
If you cannot find one of them, `scroll` until you can.

**2. Read everything relevant, not just the visible part.** If instructions or a
document continue below the fold, call `read_screen` on that pane before acting
on it.

**3. Click where the text goes, then VERIFY FOCUS BEFORE TYPING.**

  a. `click_point` the field / editor / option
  b. `observe` again and look: is there a text caret or focus ring in the field,
     or is the option you intended visibly selected?
  c. only then `type_text`

If you cannot see focus, it is not focused. Pages move focus on load, dialogs
steal it, and a click can land just outside the field. Typing into an unfocused
window scatters your text across the page as keyboard shortcuts - which does
damage you then have to undo. One extra observe to check beats that every time.

To replace existing content, select it (`ctrl+a`) before typing so your text
replaces it rather than being spliced into the middle of it.

**4. Confirm the text landed before saving or sending.** `observe` and read back
what is actually in the field. Only then save or send.

**5. Confirm it registered.** A saved file, a sent message, a status change. If
nothing changed, it did not happen - find out why rather than assuming it worked.

## Typing code into an editor

A code editor is not a text box. It auto-indents when you press Enter,
auto-closes a bracket or quote when you open one, and pops up suggestions. If
you type code - which already has its own indentation and its own closing
brackets - with `type_text`, the editor's "help" corrupts it: indentation
doubles and cascades deeper with every level, `)` and `}` come out duplicated,
and a suggestion box can swallow an Enter so two lines merge.

So: **when what you are typing is code and you are in an editor, use
`type_code`, not `type_text`.** It neutralizes those editor behaviours so your
code lands exactly as written. Pass the code with its real newlines and
indentation.

**Always pass `language`** - read it off the editor's language selector or tab
(`java`, `python`, `cpp`, `javascript`, ...). In a brace-delimited language the
tool lets the editor do the indenting and only types the difference, which is
much faster; in Python and other whitespace-sensitive languages it takes the
slower path that reproduces every space exactly.

The workflow:

  1. `click_point` the editor, then `observe` and confirm it is focused (a caret
     or focus ring in it) - `type_code` clears the editor by default before
     typing, so you must be sure the editor, not something else, has focus.
  2. `type_code(code=..., language=...)`. By default it selects-all and replaces
     whatever is there with your code. (Pass `replace=false` to add to it.)
  3. `observe` and read the code back. Confirm the indentation and brackets are
     right before you save or run it.

## Handing off a hard sub-problem

Some steps are worth working out properly rather than improvising on the
perception loop - a fiddly bit of logic, a self-contained piece of code.
When you hit one, read what you need off the screen, call `solve` with a
full, self-contained description of the problem (it cannot see the screen),
and use what comes back. Keep driving the UI yourself; `solve` only thinks.

## Navigation

Use `navigate` for URLs. Never press ctrl+L and then type as two separate steps
- seconds pass between tool calls and pages take focus back in that window,
which sends the URL into a search box on the page instead of the address bar.

Do not guess URL paths you have not seen. A wrong path 404s, repaints the
screen, and looks like success. Go to the site root and click through instead.

## Finishing

When the goal is visibly achieved, confirm it on screen, then call
`complete_task(success=true, summary=...)` as your final action - summarise what
you did and the on-screen evidence that proves it. If you cannot make progress,
call `complete_task(success=false, summary=...)` and say specifically what you
tried and what blocked you - do not keep going once you are out of genuinely
different approaches.

Always finish with `complete_task`: it records the outcome and is how you turn a
successful multi-step run into a reusable skill you can replay next time.
"""


def boot():
    """Bring up the shared hardware layer. Identical to main.py's path."""
    desktop.DRY_RUN = config.env_flag("AUTOPC_DRY_RUN")
    print("Booting perception and hands...")
    eyes = DXCamEyes(target_fps=60)
    geom = assert_coordinate_spaces_agree(eyes.grab_frame())
    print(f"  screen  : {geom.primary_width}x{geom.primary_height} (verified)")

    hands = InterceptionHands(screen_size=geom.primary_size)
    perception = PerceptionEngine(screenshot_fn=eyes.grab_frame, max_candidates=90)
    from autopc.core.vision_grounding import warm_omniparser_async
    warm_omniparser_async()  # background GPU-sidecar warmup; no-op if disabled
    executor = Executor(perception, hands, settle_time=0.35)
    reader = ScreenReader(eyes.grab_frame, hands, settle_time=0.45)
    desktop.attach(perception, executor, reader)
    # The separate reasoning session behind the `solve` tool. Created here,
    # connected (warmed) by boot_runner.
    from autopc.agent_brain.solver import Solver
    solver = Solver()
    desktop.attach_solver(solver)
    # Persistent memory: learned facts (MEMORY.md) + episodic experiences (vector
    # store). Loaded here so what the agent knew last run is available this run.
    memory = build_memory()
    desktop.attach_memory(memory)
    print(f"  solver  : {solver.model} (effort={solver.effort}, adaptive thinking)")
    print(f"  memory  : {len(memory.core_memory)} core facts, "
          f"recall={'semantic' if memory.vectors.semantic else 'keyword'}")
    print(f"  data    : {config.data_dir()}")
    print(f"  input   : {'DRY RUN (nothing is clicked or typed)' if desktop.DRY_RUN else 'LIVE'}")
    return perception, executor


def _build_options(model: str, max_turns: int, budget_usd: float) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"desktop": desktop.desktop_server()},
        # No built-in tools at all. Two reasons: Read/Write/Bash/WebSearch are
        # irrelevant to driving a desktop and cost prefix tokens every turn,
        # and with a large built-in set the CLI defers tool loading, which can
        # spend the first turn on a ToolSearch call before it even looks at the
        # screen. An empty set loads the desktop tools upfront.
        tools=[],
        allowed_tools=desktop.ALLOWED_TOOL_NAMES,
        # Nothing is watching to answer a permission prompt, and a blocked
        # prompt would hang the run mid-task with the mouse parked somewhere.
        # The only tools are the desktop ones above; safety comes from dry run,
        # Ctrl+C, and the caps below, not from interactive approval.
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        # Hard spend ceiling -- the backstop against a loop that decides to
        # observe the screen four hundred times.
        max_budget_usd=budget_usd,
        # Headroom over the 1 MiB stdio default so a busy screen cannot kill a
        # run outright. tools.py still encodes well under this.
        max_buffer_size=4 * 1024 * 1024,
        # Don't load CLAUDE.md / .claude from the working directory -- that
        # context is about developing code, not about driving a desktop.
        setting_sources=[],
    )


async def _consume(stream, started: float, turns: int) -> int:
    """Render one attempt's message stream. Returns the running tool-call count."""
    async for message in stream:
        if isinstance(message, SystemMessage) and message.subtype == "init":
            print(f"session {str(message.data.get('session_id'))[:8]}  "
                  f"model {message.data.get('model')}\n")

        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    # The assistant's reasoning -- attach it to its next action
                    # in the trace so a crystallized skill reads as intent.
                    desktop.note_thought(block.text)
                    print(f"  · {block.text.strip()[:300]}")
                elif isinstance(block, ToolUseBlock):
                    turns += 1
                    name = block.name.replace("mcp__desktop__", "")
                    args = {k: v for k, v in (block.input or {}).items()}
                    print(f"  [{turns:>2}] {name}({args})")

        elif isinstance(message, ResultMessage):
            elapsed = time.time() - started
            u = message.usage or {}
            print("\n" + "=" * 64)
            print(f"RESULT: {str(message.result)[:400]}")
            print(f"tool calls   : {turns}")
            print(f"wall clock   : {elapsed:.1f}s"
                  + (f"  ({elapsed / turns:.1f}s per call)" if turns else ""))
            print(f"tokens       : in={u.get('input_tokens')} out={u.get('output_tokens')} "
                  f"cache_write={u.get('cache_creation_input_tokens')} "
                  f"cache_read={u.get('cache_read_input_tokens')}")
            cost = getattr(message, "total_cost_usd", None)
            if cost is not None:
                print(f"cost (equiv) : ${cost:.4f}")
            print("=" * 64)
    return turns


def _goal_with_memory(goal: str) -> str:
    """Prefix a goal with the agent's current persistent memory, so it begins
    from what it already knows rather than a blank slate each time."""
    preamble = desktop.memory_preamble()
    if not preamble.strip():
        return goal
    return (f"## What you already know (persistent memory)\n{preamble}\n\n"
            f"## Your goal\n{goal}")


async def run_goal(goal: str, model: str = "claude-sonnet-5-5", max_turns: int = 60,
                   budget_usd: float = 2.0):
    """Run a single goal, then tear down. Used for the one-shot CLI form."""
    boot()
    await desktop.warm_solver()  # bring the Opus reasoner up before the goal runs
    options = _build_options(model, max_turns, budget_usd)
    print("=" * 64)
    print(f"GOAL: {goal}")
    print(f"model={model}  max_turns={max_turns}")
    print("=" * 64)
    desktop.start_trace(goal)
    await _consume(query(prompt=_goal_with_memory(goal), options=options),
                   time.time(), 0)


class Session:
    """
    A persistent agent session - one CLI process reused across many goals.

    This exists because `query()` spawns a **fresh Claude Code subprocess on
    every call**, and that spawn dominates the latency. Measured on this
    machine with no tools, no image, and the prompt "say ok" -- work Haiku
    itself answers in well under a second:

        query() round trip   5.4s, then 9.8s on the next call

    There is no warm path with `query()`; the second call is not faster than
    the first because it is a different process. `ClaudeSDKClient` holds one
    process open, so that cost is paid once per session instead of once per
    goal. It also keeps the prompt cache warm across goals, which `query()`
    cannot do -- every measured one-shot run reported `cache_read=0`.

    The hardware boots once here too, rather than per goal.
    """

    def __init__(self, model: str = "claude-sonnet-5-5", max_turns: int = 60,
                 budget_usd: float = 2.0):
        self.model = model
        self._options = _build_options(model, max_turns, budget_usd)
        self._client: Optional[ClaudeSDKClient] = None

    async def __aenter__(self) -> "Session":
        boot()
        self._client = ClaudeSDKClient(options=self._options)
        t0 = time.time()
        await self._client.connect()
        await desktop.warm_solver()  # bring the Opus reasoner up alongside the driver
        print(f"  agent session ready in {time.time() - t0:.1f}s "
              f"(reused for every goal - no per-goal process spawn)\n")
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass

    async def run(self, goal: str) -> None:
        print("=" * 64)
        print(f"GOAL: {goal}")
        print(f"model={self.model}")
        print("=" * 64)
        started = time.time()
        desktop.start_trace(goal)
        await self._client.query(_goal_with_memory(goal))
        await _consume(self._client.receive_response(), started, 0)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m autopc.agent_brain.run",
        description="Run a desktop goal through the Claude Agent SDK.",
        epilog="Ctrl+C stops the run. Pass --dry-run to report actions without "
               "performing them.",
    )
    parser.add_argument("goal", nargs="+", help="What the agent should accomplish")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report each click, key press and scroll instead of "
                             "performing it.")
    parser.add_argument("--model", default=os.environ.get("AUTOPC_MODEL",
                                                          "claude-sonnet-5-5"),
                        help="Model id (default: claude-sonnet-5-5)")
    parser.add_argument("--max-turns", type=int, default=60,
                        help="Hard cap on tool calls (default: 60)")
    parser.add_argument("--budget", type=float, default=2.0,
                        help="Spend ceiling in USD-equivalent (default: 2.0)")
    args = parser.parse_args()

    if args.dry_run:
        # Set before boot() reads it, so a flag and the env var behave
        # identically.
        os.environ["AUTOPC_DRY_RUN"] = "1"

    goal = " ".join(args.goal)
    try:
        asyncio.run(run_goal(goal, model=args.model, max_turns=args.max_turns,
                             budget_usd=args.budget))
    except KeyboardInterrupt:
        print("\ninterrupted - mouse left wherever it was")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
