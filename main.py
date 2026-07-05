#!/usr/bin/env python
"""
AutoGuard -- native computer-use agent.

Boots the eyes (DXCam screen capture), hands (Interception kernel driver + human
Bezier movement), perception (vision detection + OCR + optional OmniParser), and
drives them with the Claude Agent SDK. The model operates the desktop the way a
person does: it looks at a screenshot, reads the screen, and points/types/scrolls
-- no numbered regions, no accessibility APIs.

    python main.py "open notepad and type hello"   # run one goal (clicking ARMED)
    python main.py                                  # interactive session
    python main.py --dry-run "..."                  # report actions, perform none
    python main.py --model claude-haiku-4-5-20251001 "..."

Prerequisites (see README.md): the Interception driver, the Claude Code CLI
(subscription auth), and -- optionally, for the recall boost -- an OmniParser
checkout and a CUDA Python for the detector sidecar. Everything degrades
gracefully if the optional pieces are absent.
"""
import argparse
import asyncio
import os
import sys

# Make the AutoGuard folder importable as the project root no matter where it is
# launched from, so the `autopc` package resolves to THIS copy rather than
# anything else on the path.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env.local"))

from autopc.agent_brain.run import Session, run_goal      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AutoGuard -- native computer-use agent.",
        epilog="Clicking is ARMED by default; pass --dry-run to preview without "
               "actuating. Ctrl+C stops a run.",
    )
    parser.add_argument("goal", nargs="*",
                        help="Goal to run immediately. Omit for an interactive prompt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report each click, key press and scroll instead of performing it.")
    parser.add_argument("--model", default=os.environ.get("AUTOPC_MODEL", "claude-sonnet-5-5"),
                        help="Model id (default: claude-sonnet-5-5)")
    parser.add_argument("--max-turns", type=int, default=60,
                        help="Hard cap on tool calls per goal (default: 60)")
    parser.add_argument("--budget", type=float, default=2.0,
                        help="USD-equivalent spend ceiling per goal (default: 2.0)")
    args = parser.parse_args()

    # Authoritative for this process: run.boot() reads AUTOPC_DRY_RUN.
    if args.dry_run:
        os.environ["AUTOPC_DRY_RUN"] = "1"
    print("=" * 64)
    print("AutoGuard -- native computer-use agent")
    print(f"  model    : {args.model}")
    print(f"  clicking : {'DRY RUN (--dry-run)' if args.dry_run else 'ARMED'}")
    print("=" * 64)

    goal = " ".join(args.goal) if args.goal else None

    if goal:
        try:
            asyncio.run(run_goal(goal, model=args.model,
                                 max_turns=args.max_turns, budget_usd=args.budget))
        except KeyboardInterrupt:
            print("\ninterrupted -- mouse left wherever it was")
        return

    async def interactive() -> None:
        async with Session(model=args.model, max_turns=args.max_turns,
                           budget_usd=args.budget) as session:
            while True:
                try:
                    text = input("\nGoal (or 'exit'): ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nshutting down AutoGuard...")
                    return
                if not text:
                    continue
                if text.lower() in ("exit", "quit"):
                    return
                try:
                    await session.run(text)
                except KeyboardInterrupt:
                    print("\ninterrupted -- mouse left wherever it was")
                except Exception as exc:  # a bad goal should not kill the session
                    print(f"\nRun failed: {type(exc).__name__}: {exc}\n")

    try:
        asyncio.run(interactive())
    except KeyboardInterrupt:
        print("\nshutting down AutoGuard...")


if __name__ == "__main__":
    main()
