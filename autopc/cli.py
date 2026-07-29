"""
AutoPC command-line interface.

The single entry point for the desktop agent. With no subcommand it opens an
interactive shell; it also runs, resumes, and queues tasks, and inspects what
the agent has learned.

    autopc                      open the interactive shell (default)
    autopc "<goal>"             same as `autopc run "<goal>"`
    autopc run "<goal>"         run one task to a verified finish, then exit
    autopc resume [<id>]        continue an unfinished task (latest if no id)
    autopc queue "<goal>"       add a task to the queue (does not run it)
    autopc work                 work through the queued tasks, oldest first
    autopc tasks                list tasks and their status
    autopc skills               list the skills the agent has learned
    autopc memory               show the agent's persistent memory
    autopc --version

Run `autopc -h`, or `autopc <command> -h`, for details.

Every task gets a definition of done, is driven until it is independently
verified on screen, and is reflected on afterwards so the agent improves.
Ctrl+C stops the current task immediately (mouse and keyboard included) and
saves it so `resume` can continue it.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading

from autopc import __version__, config

BANNER = r"""
    _       _        ____   ____
   / \ _  _| |_ ___ |  _ \ / ___|   AutoPC {v}
  / _ \ || |  _/ _ \| |_) | |       self-learning desktop agent
 /_/ \_\_,_|\__\___/|  __/ \____|   on the Claude Agent SDK
                   |_|
""".strip("\n").format(v=__version__)

SHELL_HELP = """\
Type a task in plain English and press Enter. The agent gives it a definition of
done and works until the task is verified complete on screen, e.g.
    open notepad, write a haiku about winter, and save it to the desktop as winter.txt

Memory and learned skills carry across tasks in this session (and future ones).
Ctrl+C stops the running task and keeps the shell open; /resume continues it.

Shell commands:
  /help            show this help
  /tasks           list recent tasks and their status
  /resume [id]     continue an unfinished task (the latest, or one by id)
  /skills          list learned skills
  /memory          show persistent memory (core facts)
  /clear           clear the working-memory scratchpad (keeps learned facts)
  /quit, /exit     leave the shell (Ctrl+Z then Enter also works)
"""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _apply_global_env(args: argparse.Namespace) -> None:
    """Turn CLI flags into the env vars the agent reads when it boots."""
    if getattr(args, "data_dir", None):
        os.environ["AUTOPC_DATA_DIR"] = args.data_dir
    if getattr(args, "dry_run", False):
        os.environ["AUTOPC_DRY_RUN"] = "1"


def _data_dir(args: argparse.Namespace) -> str:
    return getattr(args, "data_dir", None) or config.data_dir()


def _print_skills(data_dir: str) -> None:
    from autopc.skills import build_skill_manager
    print(build_skill_manager(data_dir).skills_list())


def _print_memory(data_dir: str) -> None:
    from autopc.memory import PagingManager, VectorStore
    vs = VectorStore(os.path.join(data_dir, "vectors"), embedder=None)
    print(PagingManager(vs, data_dir=data_dir).get_full_memory_context())


def _print_tasks(data_dir: str, limit: int = 20) -> None:
    from autopc.tasks import build_task_store
    tasks = build_task_store(data_dir).recent(limit)
    if not tasks:
        print("No tasks yet.")
        return
    for t in tasks:
        met = sum(1 for c in t.criteria if c.status == "met")
        tot = len(t.criteria)
        print(f"  {t.id}  [{t.status:>9}]  {met}/{tot}  {t.goal[:60]}")


class Interrupted(Exception):
    """The user pressed Ctrl+C while a task was running."""


async def run_interruptible(coro):
    """Await one task so that Ctrl+C stops just that task.

    Ctrl+C aborts the mouse and keyboard mid-action (`CONTROL.abort()`) and
    cancels the task; the runner records the attempt as interrupted. Raises
    `Interrupted` in that case, so a caller like the shell can carry on.

    This replaces asyncio's own Ctrl+C handling for the duration of the task:
    that handler cancels the *whole* program's main task, which would close the
    shell rather than stop the task.
    """
    from autopc.core.control import CONTROL
    loop = asyncio.get_running_loop()
    CONTROL.reset()
    task = asyncio.ensure_future(coro)
    pressed = False

    def on_sigint(signum, frame):
        nonlocal pressed
        pressed = True
        CONTROL.abort()
        loop.call_soon_threadsafe(task.cancel)

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        return await task
    except asyncio.CancelledError:
        if not pressed:
            raise
        raise Interrupted() from None
    finally:
        signal.signal(signal.SIGINT, previous)


async def _ainput(prompt: str) -> str:
    """`input()` without blocking the event loop or interpreter exit.

    A daemon thread rather than `asyncio.to_thread`: the default executor's
    threads are joined at shutdown, so leaving the shell would wait on an
    `input()` that never returns. Raises EOFError at end of input.
    """
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def deliver(line, eof):
        if fut.done():
            return
        if eof:
            fut.set_exception(EOFError())
        else:
            fut.set_result(line)

    def read():
        try:
            line, eof = input(prompt), False
        except (EOFError, KeyboardInterrupt):
            line, eof = "", True
        try:
            loop.call_soon_threadsafe(deliver, line, eof)
        except RuntimeError:  # the loop is gone: the shell already exited
            pass

    threading.Thread(target=read, name="autopc-input", daemon=True).start()
    return await fut


# --------------------------------------------------------------------------
# commands that run the agent
# --------------------------------------------------------------------------

def _runner_kwargs(args: argparse.Namespace) -> dict:
    kw = dict(model=args.model, max_turns=args.max_turns, budget_usd=args.budget)
    if getattr(args, "max_attempts", None):
        kw["max_attempts"] = args.max_attempts
    return kw


_INTERRUPTED = "\ninterrupted -- progress saved; continue with `autopc resume`"


def cmd_run(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    goal = " ".join(args.goal).strip()
    if not goal:
        print('Nothing to do -- give a goal, e.g. autopc run "open notepad"')
        return 2

    async def go() -> int:
        from autopc.agent_brain.run import boot_runner, shutdown
        try:
            runner = await boot_runner(**_runner_kwargs(args))
            ledger = await run_interruptible(runner.run(goal))
        except Interrupted:
            print(_INTERRUPTED)
            return 130
        finally:
            await shutdown()
        return 0 if ledger.status == "verified" else 1
    try:
        return asyncio.run(go())
    except KeyboardInterrupt:      # Ctrl+C while booting
        print("\ninterrupted")
        return 130


def cmd_resume(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    from autopc.tasks import build_task_store
    store = build_task_store(_data_dir(args))
    ledger = store.load(args.task_id) if args.task_id else (
        store.latest_resumable() or store.next_queued())
    if ledger is None:
        print("Nothing to resume." if not args.task_id else f"No task matching '{args.task_id}'.")
        return 1
    if ledger.finished:
        print(f"Task {ledger.id} is already {ledger.status}.")
        return 0

    async def go() -> int:
        from autopc.agent_brain.run import boot_runner, shutdown
        try:
            runner = await boot_runner(**_runner_kwargs(args))
            result = await run_interruptible(runner.resume(ledger))
        except Interrupted:
            print(_INTERRUPTED)
            return 130
        finally:
            await shutdown()
        return 0 if result.status == "verified" else 1
    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


def cmd_queue(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    from autopc.tasks import build_task_store
    goal = " ".join(args.goal).strip()
    if not goal:
        print("Give a goal to queue.")
        return 2
    t = build_task_store(_data_dir(args)).create(goal, status="queued")
    print(f"Queued {t.id}\n  {goal}\nRun the queue with:  autopc work")
    return 0


def cmd_work(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    from autopc.tasks import build_task_store
    store = build_task_store(_data_dir(args))
    if store.next_queued() is None:
        print("The queue is empty. Add tasks with:  autopc queue \"<goal>\"")
        return 0

    async def go() -> int:
        from autopc.agent_brain.run import boot_runner, shutdown
        done = 0
        try:
            runner = await boot_runner(**_runner_kwargs(args))
            while True:
                task = store.next_queued()
                if task is None:
                    break
                print(f"\n>>> working queued task {task.id}")
                await run_interruptible(runner.resume(task))
                done += 1
        except Interrupted:
            print("\ninterrupted -- that task is saved; `autopc resume` continues it, "
                  "and `autopc work` picks up the rest of the queue")
            return 130
        finally:
            await shutdown()
        print(f"\nQueue drained -- worked {done} task(s).")
        return 0
    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


def cmd_tasks(args: argparse.Namespace) -> int:
    _print_tasks(_data_dir(args), args.limit)
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    _print_skills(_data_dir(args))
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    _print_memory(_data_dir(args))
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    try:
        asyncio.run(_shell(args))
    except KeyboardInterrupt:
        print("\nbye")
    return 0


async def _shell(args: argparse.Namespace) -> None:
    from autopc.agent_brain import tools as desktop
    from autopc.agent_brain.run import boot_runner, shutdown
    from autopc.tasks import build_task_store

    print(BANNER, "\n")
    data_dir = _data_dir(args)
    store = build_task_store(data_dir)
    runner = await boot_runner(**_runner_kwargs(args))
    print(f"\n{'DRY RUN -- nothing will be clicked or typed. ' if desktop.DRY_RUN else ''}"
          "Type /help for commands, or just describe a task.\n")

    async def run_task(coro) -> None:
        try:
            await run_interruptible(coro)
        except Interrupted:
            print("\n(task interrupted -- shell still open; /resume continues it)")
        except Exception as exc:
            print(f"error running task: {type(exc).__name__}: {exc}")

    try:
        while True:
            try:
                line = (await _ainput("autopc> ")).strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            low = line.lower()
            if low in ("/quit", "/exit", ":q"):
                break
            if low in ("/help", "/?", "help"):
                print(SHELL_HELP)
            elif low == "/tasks":
                _print_tasks(data_dir)
            elif low == "/skills":
                _print_skills(data_dir)
            elif low == "/memory":
                _print_memory(data_dir)
            elif low == "/clear":
                desktop.reset_notes()
                print("working memory cleared (learned facts kept)")
            elif low.startswith("/resume"):
                parts = line.split(maxsplit=1)
                ledger = (store.load(parts[1].strip()) if len(parts) > 1
                          else store.latest_resumable())
                if ledger is None:
                    print("Nothing to resume." if len(parts) == 1
                          else "No task matching that id.")
                else:
                    await run_task(runner.resume(ledger))
            elif low.startswith("/"):
                print(f"Unknown command {line.split()[0]!r}. Type /help.")
            else:
                await run_task(runner.run(line))
    finally:
        await shutdown()


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopc",
        description="AutoPC -- a self-learning Windows desktop agent on the Claude "
                    "Agent SDK. It looks at the screen, acts with a real mouse and "
                    "keyboard, works each task to an independently verified finish, "
                    "and turns successful runs into reusable skills.",
        epilog="Examples:\n"
               "  autopc                                  open the interactive shell\n"
               "  autopc \"open notepad and type hi\"        run one task\n"
               "  autopc run --dry-run \"...\"               rehearse without touching anything\n"
               "  autopc queue \"...\"  then  autopc work    batch several tasks\n"
               "  autopc resume                           continue the last unfinished task\n"
               "  autopc tasks / skills / memory          inspect state\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"autopc {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=config.model(),
                        help="driver model id (default: $AUTOPC_MODEL or %(default)s)")
    common.add_argument("--max-turns", type=int, default=80,
                        help="hard cap on tool calls per attempt (default: 80)")
    common.add_argument("--budget", type=float, default=4.0,
                        help="spend ceiling in USD per attempt (default: 4.0)")
    common.add_argument("--max-attempts", type=int, default=None,
                        help="how many fresh attempts before giving up "
                             "(default: $AUTOPC_MAX_ATTEMPTS or 3)")
    common.add_argument("--dry-run", action="store_true",
                        help="report every click, key press and scroll instead of "
                             "performing it")
    common.add_argument("--data-dir", default=None,
                        help="where the agent keeps memory, skills and tasks "
                             f"(default: $AUTOPC_DATA_DIR or {config.data_dir()})")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("shell", parents=[common], help="open the interactive shell (default)"
                   ).set_defaults(func=cmd_shell)

    p_run = sub.add_parser("run", parents=[common], help="run a single task to a verified finish")
    p_run.add_argument("goal", nargs="+", help="the task, in plain English")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", parents=[common], help="continue an unfinished task")
    p_resume.add_argument("task_id", nargs="?", default=None,
                          help="task id or prefix (default: the latest unfinished task)")
    p_resume.set_defaults(func=cmd_resume)

    p_queue = sub.add_parser("queue", parents=[common], help="add a task to the queue")
    p_queue.add_argument("goal", nargs="+", help="the task, in plain English")
    p_queue.set_defaults(func=cmd_queue)

    sub.add_parser("work", parents=[common], help="work through the queued tasks"
                   ).set_defaults(func=cmd_work)

    p_tasks = sub.add_parser("tasks", parents=[common], help="list tasks and their status")
    p_tasks.add_argument("--limit", type=int, default=20, help="how many to show (default: 20)")
    p_tasks.set_defaults(func=cmd_tasks)

    sub.add_parser("skills", parents=[common], help="list learned skills"
                   ).set_defaults(func=cmd_skills)
    sub.add_parser("memory", parents=[common], help="show persistent memory"
                   ).set_defaults(func=cmd_memory)
    return parser


_COMMANDS = {"shell", "run", "resume", "queue", "work", "tasks", "skills", "memory"}
_VALUE_OPTIONS = {"--model", "--max-turns", "--budget", "--max-attempts", "--data-dir",
                  "--limit"}
_TOP_LEVEL = {"-h", "--help", "-v", "--version"}


def route(argv: list) -> list:
    """Put the subcommand first, defaulting to the shell, or `run` for a bare goal.

        autopc                         -> shell
        autopc --dry-run               -> shell --dry-run
        autopc open notepad            -> run open notepad
        autopc --dry-run run "x"       -> run --dry-run "x"
    """
    first = None
    skip = False
    for i, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in _VALUE_OPTIONS
            continue
        first = i
        break
    if first is None:
        return argv if set(argv) & _TOP_LEVEL else ["shell"] + argv
    if argv[first] in _COMMANDS:
        return [argv[first]] + argv[:first] + argv[first + 1:]
    return ["run"] + argv


def main(argv=None) -> int:
    config.load_env()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(route(argv))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
