# AutoPC Architecture

AutoPC is organised as a **body**, a **mind**, and a **learning layer**, wired
together by the Claude Agent SDK. This document explains each part and how a
goal flows through them. All code lives in the `autopc` package; paths below are
relative to it.

```
                         +-------------------------------------------+
                         |             Claude Agent SDK              |
       goal ----------->  |   (the loop: planning, tool use, state)  |
                         +----------------+--------------------------+
                                          | calls tools
             +----------------------------+---------------------------+
             v                            v                           v
      +------------+            +--------------+            +--------------+
      |   BODY     |            |    MIND      |            |   LEARNING   |
      |  core/     |            | agent_brain/ |            | memory/ +    |
      |            |            |              |            | skills/ +    |
      | eyes       |  observe   | tools.py     |  remember  | tasks/       |
      | hands      | <--------  | runner.py    | -------->  | paging mgr   |
      | perception |   click    | solver.py    |  skill     | crystallizer |
      | executor   |   type     | trace.py     |  learned   | ledger       |
      +------------+            +--------------+            +--------------+
```

## Entry point -- `cli.py`

`autopc` (or `python -m autopc`) loads `.env.local`, parses the command, and
boots the agent through `agent_brain/run.py`. Every command that runs the agent
goes through the task runner, so every goal gets a definition of done and
verification. `config.py` holds the settings, file locations and model
defaults.

Ctrl+C is handled per task (`cli.run_interruptible`): it aborts the hands
mid-action through `core/control.py` and cancels the task; the runner saves the
attempt as interrupted, and the shell stays open.

## The body -- `core/`

- **`eyes.py`** -- DXGI desktop-duplication capture (fast, no screenshot API).
- **`hands.py`** -- input through the
  [Interception](https://github.com/oblitum/Interception) driver. Pointer moves
  follow a short Bezier curve (`cursor_path.py`); it is DPI-aware and disables
  "Enhance pointer precision" for the session so relative input lands 1:1.
  Supports click, double/right-click, drag, scroll, and typing. Every step
  passes through the pause/stop gate in `control.py`.
- **`perception.py`** -- one look at the screen: the screenshot, the small
  controls detected in it (`vision_grounding.py`, optionally boosted by an
  OmniParser model), OCR text (`ocr.py`), and Win32 window context. No
  accessibility tree, so it behaves the same in every app.
- **`executor.py`** -- turns a tool call into input: snaps a click that lands on
  a small control to its centre, runs a **pre-flight occlusion check** (is
  another window covering the point?), acts, then reports **what measurably
  changed** on screen and in the active window.
- **`reader.py`** -- scroll-and-stitch OCR for content taller than the viewport.
- **`code_typer.py`** -- types code into editors that auto-indent and
  auto-close brackets without corrupting it.

The model points at pixels in the screenshot it was shown. Text targets come
with exact coordinates from OCR; everything else it reads off the image.

## The mind -- `agent_brain/`

- **`run.py`** -- boots the body, memory, skills and solver; holds the system
  prompt and the SDK options; returns a ready `TaskRunner`.
- **`runner.py`** -- drives one goal to a verified conclusion (see below).
- **`tools.py`** -- the tool surface the model drives: `observe`,
  `click_point`, `hover_point`, `drag`, `type_text`, `type_code`, `press_key`,
  `scroll`, `navigate`, `read_screen`; `solve`; the memory tools;
  `skill_view`; and the task-control tools ending in `complete_task`. Dry run
  is enforced here, in `_dispatch`.
- **`solver.py`** -- a separate, warm reasoning session (Claude Opus by
  default) for hard sub-problems, kept off the fast perception loop.
- **`llm.py`** -- one-shot model calls (vision + structured output) used by the
  contract, the verifier and reflection.
- **`trace.py`** -- records the `(thought, action, outcome)` steps of a goal from
  the SDK message stream, so a successful run can be crystallized.

## The learning layer

### Declarative memory -- `memory/`

A MemGPT/Letta-style three-tier store:

- **HOT (working)** -- the last N actions; ephemeral, shown by `recall`.
- **WARM (core)** -- learned facts and preferences; self-edited by the agent,
  persisted to `MEMORY.md`, and injected into every goal. Near-duplicate facts
  are merged so a re-learned fact doesn't evict useful ones.
- **COLD (archival)** -- older facts and past experiences; exact recall from a
  JSON KV file, semantic recall from the vector store.

`embedder.py` provides embeddings through a pluggable backend -- local ONNX via
`fastembed` (default, no torch), or a keyword fallback. `vector_store.py` uses
FAISS when an embedder is available and token-overlap search otherwise, so the
system never hard-fails on a missing optional dependency.

### Procedural memory -- `skills/`

- **`manager.py`** -- `SKILL.md` files with YAML frontmatter: trigger + keyword
  retrieval, success-rate grading, and pruning of skills that keep failing.
- **`crystallizer.py`** -- distills a successful multi-step trace into a
  `SKILL.md` via an injected async generator (wired to Claude in `run.py`), so
  this package depends on no provider or SDK.

### Goal reinforcement -- `tasks/`

The layer that keeps a long task on target. It makes finishing the task the
agent's only objective and makes "finished" mean *independently verified*.

- **`ledger.py`** -- the task's single source of truth, on disk: the immutable
  goal (verbatim), the definition of done (criteria), an optional plan, a
  progress journal, attempts, verification verdicts and lessons. Atomic
  writes; survives crashes, restarts and context compaction.
- **`store.py`** -- one directory per task; the queue and crash-recovery lookup.
- **`contract.py`** -- turns the goal into the definition of done *before*
  acting: screen-observable criteria covering exactly what was asked. Written
  once; never relaxed by the actor.
- **`verifier.py`** -- an independent judge. On a completion claim it takes a
  fresh screenshot and judges it against the criteria with its own skeptical
  prompt and no access to the agent's conversation. Rejections say exactly what
  is missing. If the verifier call fails, a claim is accepted only with the
  agent's own recorded evidence for every criterion, and the ledger logs it.
- **`guard.py`** -- the SDK hooks that enforce all this: a **Stop** gate (no
  ending the turn until verified / blocked-on-user / gave up after pushback),
  periodic goal **re-anchoring**, **loop/stall/failure** detection, and
  **compaction** recovery.
- **`reflection.py`** -- after every attempt, extracts lessons (for the next
  attempt), durable facts (to core memory) and an episode (to archival memory).
- **`state.py`** -- the per-attempt counters the tools and guard share.

## A goal, end to end

1. **Contract.** `runner.py` creates a task ledger and, for a new goal, derives
   the definition of done (`contract.py`), informed by relevant memory, skills
   and past episodes.
2. **Attempt.** It begins an attempt and runs it as one SDK `query()` in a fresh
   context, with the guard hooks bound (`guard.py`). The attempt prompt carries
   the goal, the definition of done, current memory, relevant skills, and (on a
   retry) the lessons from earlier attempts.
3. **Act.** The agent loops observe -> decide -> act. Each physical action passes
   through `tools._dispatch`, recorded to the trace and the working tier. The
   guard re-anchors the goal periodically and flags loops/stalls; the Stop gate
   refuses a premature end.
4. **Claim + verify.** The agent calls `complete_task(success=true)`; the tool
   takes a fresh screenshot and asks the independent verifier. If any criterion
   is unmet the claim is rejected with specifics and the agent keeps working.
5. **Reflect.** When the stream ends, `finish_goal` grades surfaced skills and --
   on a verified success -- crystallizes a `SKILL.md`; then `reflection.py`
   records lessons, durable facts and an episode.
6. **Retry or finish.** If not verified (and not user-blocked or interrupted),
   the runner starts a fresh attempt that inherits the lessons, up to the attempt
   budget, after which the task is marked `failed` with everything preserved for
   `resume`.

## Known limitations

- Windows only, and the Interception driver needs an admin install.
- Skill retrieval is trigger + keyword based; embedding skills into the vector
  store is not implemented yet.
