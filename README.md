# AutoPC

**A self-learning Windows desktop agent, built on the Claude Agent SDK.**

AutoPC operates a real Windows desktop the way a person does: it *looks* at the
screen, *reasons* about the next step, and *acts* with the mouse and keyboard.
You give it a goal in plain English instead of a script. It **remembers what it
learns** and **turns successful work into reusable skills**, so recurring tasks
get faster and more reliable.

```
$ autopc
autopc> open notepad and write a haiku about winter
  · I'll open Notepad from the Start menu, then type the haiku.
  [ 1] observe({})
  [ 2] press_key({'keys': 'win'})
  [ 3] type_text({'text': 'Notepad'})
  ...
  [ 9] complete_task({'success': True, 'summary': '...'})
skills      : crystallized new skill: open_notepad_write_haiku
autopc>
```

> **It drives your real mouse and keyboard.** Try a task with `--dry-run` first:
> every click, key press and scroll is reported instead of performed. Press
> **Ctrl+C** at any time to stop the current task immediately.

---

## What it does differently

| | Scripts / RPA | A stateless LLM agent | **AutoPC** |
|---|---|---|---|
| Handles a UI it has never seen | ✗ | ✓ | ✓ |
| Survives a layout change | ✗ | ✓ | ✓ |
| Remembers facts across runs | ✗ | ✗ | ✓ |
| Learns reusable skills from success | ✗ | ✗ | ✓ |
| Drives input at the kernel level | ✗ | usually ✗ | ✓ |

AutoPC pairs a precise **body** (kernel-level input, DPI-correct perception)
with a learning **mind** (persistent memory + skill crystallization) on top of
the Claude Agent SDK.

---

## How it works

Every goal runs through one loop:

1. **Observe** - screenshot the screen and read its text with Windows OCR; each
   line comes with the pixel coordinate that clicks it.
2. **Recall** - the agent is handed what it already knows: persistent memory
   (facts it saved) and any learned skill whose triggers match this goal.
3. **Act** - it clicks, types, scrolls, drags, or navigates, then confirms the
   change on screen. A click that lands on a small icon is snapped to the
   icon's centre.
4. **Finish** - it calls `complete_task`, which is verified as above.
5. **Learn** - a successful, novel, multi-step run is distilled into a reusable
   `SKILL.md`; a replayed skill is graded by how the run went, and skills that
   keep failing are pruned.

### The two kinds of memory

AutoPC follows the split used by MemGPT/Letta and the Hermes agent:

- **Declarative memory** (`autopc/memory/`) - a three-tier store: HOT *working*
  (recent actions), WARM *core* (learned facts, persisted to `MEMORY.md`,
  injected into every goal), and COLD *archival* (older facts and experiences,
  retrieved by semantic search). The agent edits it itself with `remember` /
  `memory_search` / `memory_archive` / `memory_recall`.
- **Procedural memory** (`autopc/skills/`) - portable `SKILL.md` playbooks, each
  distilled from a successful run, retrieved by trigger + keyword matching and
  graded over time.

Embeddings for semantic recall run **locally** through ONNX (`fastembed`) - no
PyTorch, no network after the first model download - and fall back to keyword
search if unavailable.

---

## Install

**Requirements**

- Windows 10/11 and Python 3.10+.
- The [Interception](https://github.com/oblitum/Interception) driver (one-time
  admin install + reboot). AutoPC sends input through it because it reaches
  windows that ignore or block ordinary synthetic input, such as elevated
  windows and remote sessions.
- The Windows OCR language pack for your display language (installed by
  default on most systems).
- Claude credentials for the Agent SDK (see below).

```bash
git clone <repo-url> autopc && cd autopc
python -m venv .venv && .venv\Scripts\activate
pip install -e .                # installs the `autopc` command
copy .env.template .env.local   # optional; every setting has a default
```

### Authentication

The Claude Agent SDK runs the Claude Code CLI under the hood and uses its
credentials: set `ANTHROPIC_API_KEY`, or sign in to Claude Code on this
machine. Anthropic's Agent SDK documentation describes which authentication is
appropriate for what use; check it before you build on or distribute AutoPC.

---

## Usage

```bash
autopc                       # open the interactive shell (default)
autopc run "open notepad and type hello"
autopc run --dry-run "..."   # rehearse: resolve clicks without actuating
autopc skills                # list the skills it has learned
autopc memory                # show its persistent memory
autopc -h                    # full help; `autopc <command> -h` per command
```

**In the shell**, type a task and press Enter. Slash commands: `/help`,
`/skills`, `/memory`, `/clear` (wipe the working scratchpad, keep learned
facts), `/quit`.

Input is **live by default**; `--dry-run` (or `AUTOPC_DRY_RUN=1`) reports every
click, key press and scroll instead of performing it.


## Configuration

All settings are `AUTOPC_*` environment variables, optionally set in
`.env.local` (see [.env.template](.env.template) for every option).

| Variable | Default | Meaning |
|---|---|---|
| `AUTOPC_MODEL` | `claude-sonnet-5-5` | the driver model |
| `AUTOPC_SOLVER_MODEL` | `claude-opus-5-5` | reasoning/code/skill-distillation model |
| `AUTOPC_DRY_RUN` | `0` | `1` = report actions instead of performing them |
| `AUTOPC_DATA_DIR` | `data` | where memory and skills are stored |
| `AUTOPC_EMBEDDER` | `onnx` | `onnx` \| `keyword` \| `none` |
| `AUTOPC_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `AUTOPC_SKILL_MIN_STEPS` | `5` | min steps before a run becomes a skill |

---

## Project layout

```
core/         the body - Interception hands, DXGI capture, perception, executor
agent_brain/  the mind - Claude Agent SDK loop, tools, solver, trace recorder
memory/       declarative memory - 3-tier store, embedder, vector store
skills/       procedural memory - SKILL.md manager + crystallizer
cli.py        the `autopc` command-line interface
data/         runtime state (gitignored): MEMORY.md, skills/, vector index
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## Credits

The memory and skill design draws on [MemGPT](https://github.com/cpacker/MemGPT),
[Letta](https://www.letta.com/), the Hermes agent and the Reflexion pattern. The
skill format follows Anthropic's open `SKILL.md` specification.

## License

[MIT](LICENSE).
