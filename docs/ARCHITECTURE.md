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
      |            |            |              |            | skills/      |
      | eyes       |  observe   | tools.py     |  remember  | paging mgr   |
      | hands      | <--------  | solver.py    | -------->  | vector store |
      | perception |   click    | trace.py     |  skill     | crystallizer |
      | executor   |   type     |              |  learned   | skill mgr    |
      +------------+            +--------------+            +--------------+
```

## The body -- `core/`

Vendored intact from the AutoGuard lineage; these modules carry accuracy fixes and are deliberately unchanged.

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

- **`run.py`** -- boots the body, builds memory and skills, wires everything to
  an SDK session, and runs a goal. `Session` keeps one agent process warm across
  many goals (used by the shell); `run_goal` is the one-shot form.
- **`tools.py`** -- the desktop tool surface the model drives: `observe`,
  `click_point`, `hover_point`, `drag`, `type_text`, `type_code`, `press_key`,
  `scroll`, `navigate`, `read_screen`; the solver tool (`solve`); the memory tools; `skill_view`; and
  `complete_task`.
- **`solver.py`** -- a separate, warm Claude Opus session for heavy reasoning
  (answers, code) kept off the fast perception loop.
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

## A goal, end to end

1. `run.py` starts a fresh trace and builds the goal prompt: current **memory**
   (`memory_preamble`) + any **relevant skills** (`skills_preamble`, which also
   records which skills were surfaced) + the goal itself.
2. The SDK agent loops: `observe` -> decide -> act. Each physical action passes
   through `tools._dispatch`, which records it to the trace and the working tier.
3. The agent finishes with `complete_task(success, summary)`.
4. After the stream ends, `tools.finish_goal` grades any surfaced skills by the
   outcome and, for a successful novel run of enough steps, crystallizes a new
   `SKILL.md`.

## Known limitations

- Windows only, and the Interception driver needs an admin install.
- Skill retrieval is trigger + keyword based; embedding skills into the vector
  store is not implemented yet.
