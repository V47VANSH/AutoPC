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
| Completion checked by a separate model | n/a | ✗ | ✓ |
| Saves progress to disk and resumes | ✗ | ✗ | ✓ |

## Finishing the task is the objective

Long tasks fail in predictable ways: the agent drifts as its context fills,
declares victory on a screen that merely *looks* done, loops on a dead approach,
or stops to ask a question nobody will answer. AutoPC addresses each of these
with the SDK's hooks rather than with prompt wording alone:

- **A definition of done, written first.** Before acting, the goal becomes a
  short list of criteria you can check on screen. The acting agent cannot
  change them.
- **Independent verification.** `complete_task` is checked by a *separate*
  model call that judges a fresh screenshot against the criteria, with no
  access to the agent's conversation. A premature claim is rejected with what
  is still missing, and the agent keeps working. If the verifier itself is
  unavailable (for example a network error), the claim is accepted only when
  the agent has recorded on-screen evidence for every criterion, and the
  task's log says so.
- **A stop gate.** The agent cannot end its turn until the task is verified, a
  user-only blocker is recorded, or it has given up after being pushed back.
  The gate gives way after six blocked stops so a stuck agent cannot loop
  forever; the runner then reflects and retries.
- **Re-anchoring and loop/stall detection** throughout, with the task's state
  saved to disk so a crash, restart or Ctrl+C loses at most the current
  attempt.
- **Reflection and retry.** A failed attempt is reviewed for concrete lessons;
  the next attempt runs in a fresh context that inherits the lessons, not the
  dead ends. `autopc resume` picks up an unfinished task later.

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
autopc                          # open the interactive shell
autopc "open notepad and type hello"     # run one task (same as `autopc run ...`)
autopc run --dry-run "..."      # rehearse: report every action, perform none
autopc resume [<id>]            # continue an unfinished task (latest if no id)
autopc queue "..."              # add a task to the queue (does not run it)
autopc work                     # work through the queued tasks, oldest first
autopc tasks                    # list tasks and their status
autopc skills                   # list the skills it has learned
autopc memory                   # show its persistent memory
autopc -h                       # full help; `autopc <command> -h` per command
```

`python -m autopc` works the same way.

**In the shell**, type a task and press Enter; the agent works it to a verified
finish. Slash commands: `/help`, `/tasks`, `/resume [id]`, `/skills`,
`/memory`, `/clear` (wipe the working scratchpad, keep learned facts), `/quit`.

## Safety

- **Dry run.** `--dry-run` (or `AUTOPC_DRY_RUN=1`) turns every input action -
  click, hover, drag, typing, key presses, navigation, scrolling - into a
  report of what would have happened.
- **Ctrl+C** stops the running task at once, including a mouse move or typing
  already in progress, and saves the task so `resume` can continue it. In the
  shell, the shell stays open.
- **Caps.** Each attempt is limited by `--max-turns` (default 80 tool calls) and
  `--budget` (default 4.0 USD); each task by `--max-attempts` (default 3).
- **Unattended by design.** The agent never stops to ask for permission, so it
  is given only its desktop tools: no shell, file or web tools.
- **Mouse acceleration.** While running, AutoPC turns off Windows "Enhance
  pointer precision" for click accuracy and restores it on exit (it is not
  saved, so a crash leaves it off only until you next sign in). Set
  `AUTOPC_DISABLE_MOUSE_ACCEL=0` to opt out.


## Configuration

All settings are `AUTOPC_*` environment variables, optionally set in
`.env.local` (see [.env.template](.env.template) for every option).

| Variable | Default | Meaning |
|---|---|---|
| `AUTOPC_DRY_RUN` | `0` | `1` = report actions instead of performing them |
| `AUTOPC_MODEL` | `claude-sonnet-5-5` | the driver model |
| `AUTOPC_SOLVER_MODEL` | `claude-opus-5-5` | reasoning, definition of done, verifier, reflection, skills |
| `AUTOPC_SOLVER_EFFORT` | `high` | `low` \| `medium` \| `high` \| `xhigh` \| `max` |
| `AUTOPC_MAX_ATTEMPTS` | `3` | fresh attempts before a task is marked failed |
| `AUTOPC_DATA_DIR` | `data` in the AutoPC folder | where memory, skills and tasks are stored |
| `AUTOPC_EMBEDDER` | `onnx` | `onnx` \| `keyword` \| `none` |
| `AUTOPC_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `AUTOPC_SKILL_MIN_STEPS` | `5` | min steps before a run becomes a skill |
| `AUTOPC_IMAGE_WIDTH` | native | force a smaller screenshot width |
| `AUTOPC_DISABLE_MOUSE_ACCEL` | `1` | turn off pointer acceleration while running |
| `AUTOPC_OMNIPARSER` | `0` | optional learned UI detector (see `.env.template`) |

---

## Project layout

```
autopc/
  cli.py        the `autopc` command
  config.py     settings, paths, model defaults
  core/         the body - Interception input, DXGI capture, OCR, control
                detection, executor
  agent_brain/  the mind - Claude Agent SDK wiring, tools, solver, runner
  memory/       declarative memory - 3-tier store, embedder, vector store
  skills/       procedural memory - SKILL.md manager + crystallizer
  tasks/        goal reinforcement - ledger, contract, verifier, guard hooks,
                reflection, store
tests/          unit tests (no screen, driver or network needed)
data/           runtime state for a source checkout (gitignored)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Development

```bash
pip install -e ".[dev]"
pytest
build.cmd                       # standalone build in dist\AutoPC\ (PyInstaller)
```

The tests stub the screen, the input driver and the model, so they run without
Interception or Claude credentials.

---

## Credits

The memory and skill design draws on [MemGPT](https://github.com/cpacker/MemGPT),
[Letta](https://www.letta.com/), the Hermes agent and the Reflexion pattern. The
skill format follows Anthropic's open `SKILL.md` specification.

## License

[MIT](LICENSE).
