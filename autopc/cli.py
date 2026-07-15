"""
AutoPC command-line interface.

A single entry point for the self-learning desktop agent. With no subcommand it
opens an interactive shell; it also runs one-off goals and inspects what the
agent has learned.

    autopc                      open the interactive shell (default)
    autopc shell                same, explicit
    autopc run "<goal>"         run one goal, then exit
    autopc skills               list the skills the agent has learned
    autopc memory               show the agent's persistent memory
    autopc --version

Run `autopc -h`, or `autopc <command> -h`, for details.

The agent authenticates through the Claude Agent SDK (your Claude subscription);
there is no API key here.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from autopc import __version__, config

BANNER = r"""
    _       _        ____   ____
   / \ _  _| |_ ___ |  _ \ / ___|   AutoPC {v}
  / _ \ || |  _/ _ \| |_) | |       self-learning desktop agent
 /_/ \_\_,_|\__\___/|  __/ \____|   on the Claude Agent SDK
                   |_|
""".strip("\n").format(v=__version__)

SHELL_HELP = """\
Type a task in plain English and press Enter to run it, e.g.
    open notepad and write a haiku about winter

The agent keeps its memory and learned skills across tasks in this session.

Shell commands:
  /help            show this help
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
    # Lightweight: the skill manager only reads SKILL.md files -- no model load.
    from autopc.skills import build_skill_manager
    print(build_skill_manager(data_dir).skills_list())


def _print_memory(data_dir: str) -> None:
    # Build a keyword-only memory view so inspection never downloads an embedding
    # model; it only reads MEMORY.md and the archival KV.
    from autopc.memory import PagingManager, VectorStore
    vs = VectorStore(os.path.join(data_dir, "vectors"), embedder=None)
    print(PagingManager(vs, data_dir=data_dir).get_full_memory_context())


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    _apply_global_env(args)
    from autopc.agent_brain.run import run_goal
    goal = " ".join(args.goal).strip()
    if not goal:
        print("Nothing to do -- give a goal, e.g. autopc run \"open notepad\"")
        return 2
    try:
        asyncio.run(run_goal(goal, model=args.model, max_turns=args.max_turns,
                             budget_usd=args.budget))
    except KeyboardInterrupt:
        print("\ninterrupted -- mouse left wherever it was")
        return 130
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
    # Imported here so `autopc -h / skills / memory` never trigger the DPI /
    # driver / SDK machinery that booting the hardware requires.
    from autopc.agent_brain import tools as desktop
    from autopc.agent_brain.run import Session

    print(BANNER)
    print()
    data_dir = _data_dir(args)
    async with Session(model=args.model, max_turns=args.max_turns,
                       budget_usd=args.budget) as session:
        while True:
            try:
                line = (await asyncio.to_thread(input, "autopc> ")).strip()
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
                continue
            if low == "/skills":
                _print_skills(data_dir)
                continue
            if low == "/memory":
                _print_memory(data_dir)
                continue
            if low == "/clear":
                desktop.reset_notes()
                print("working memory cleared (learned facts kept)")
                continue
            try:
                await session.run(line)
            except KeyboardInterrupt:
                print("\n(task interrupted -- shell still open)")
            except Exception as exc:  # keep the shell alive on a task error
                print(f"error running task: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopc",
        description="AutoPC -- a self-learning Windows desktop agent on the Claude "
                    "Agent SDK. It looks at the screen, acts with a real mouse and "
                    "keyboard, remembers what it learns, and turns successful runs "
                    "into reusable skills.",
        epilog="Examples:\n"
               "  autopc                                 open the interactive shell\n"
               "  autopc run \"open notepad and type hi\"   run one goal\n"
               "  autopc run --dry-run \"...\"              rehearse without clicking\n"
               "  autopc skills                          list learned skills\n"
               "  autopc memory                          show persistent memory\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version",
                        version=f"autopc {__version__}")

    # Global options shared by the commands that run the agent.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=os.environ.get("AUTOPC_MODEL",
                                                           "claude-sonnet-5-5"),
                        help="driver model id (default: %(default)s)")
    common.add_argument("--max-turns", type=int, default=60,
                        help="hard cap on tool calls per goal (default: 60)")
    common.add_argument("--budget", type=float, default=2.0,
                        help="spend ceiling in USD-equivalent per goal (default: 2.0)")
    common.add_argument("--dry-run", action="store_true",
                        help="report every click, key press and scroll instead of "
                             "performing it")
    common.add_argument("--data-dir", default=None,
                        help="where the agent keeps memory and skills "
                             f"(default: $AUTOPC_DATA_DIR or {config.data_dir()})")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_shell = sub.add_parser("shell", parents=[common],
                             help="open the interactive shell (default)")
    p_shell.set_defaults(func=cmd_shell)

    p_run = sub.add_parser("run", parents=[common],
                           help="run a single goal and exit")
    p_run.add_argument("goal", nargs="+", help="the task, in plain English")
    p_run.set_defaults(func=cmd_run)

    p_skills = sub.add_parser("skills", parents=[common],
                              help="list the skills the agent has learned")
    p_skills.set_defaults(func=cmd_skills)

    p_memory = sub.add_parser("memory", parents=[common],
                              help="show the agent's persistent memory")
    p_memory.set_defaults(func=cmd_memory)

    return parser


_COMMANDS = {"shell", "run", "skills", "memory"}


def main(argv=None) -> int:
    config.load_env()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default to the interactive shell when no command is given, while still
    # honouring any global flags typed before it (e.g. `autopc --dry-run`).
    # -h/--help/-v/--version are left to pass through to the top-level parser.
    passthrough = {"-h", "--help", "-v", "--version"}
    if not any(a in _COMMANDS for a in argv) and not (set(argv) & passthrough):
        argv = ["shell"] + argv
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
