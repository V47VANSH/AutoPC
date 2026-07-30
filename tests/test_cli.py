"""CLI routing, and Ctrl+C stopping a task without ending the program."""
import asyncio
import signal

import pytest

from autopc import cli
from autopc.core.control import CONTROL


@pytest.mark.parametrize("argv,expected", [
    ([], ["shell"]),
    (["--dry-run"], ["shell", "--dry-run"]),
    (["-h"], ["-h"]),
    (["--version"], ["--version"]),
    (["open", "notepad"], ["run", "open", "notepad"]),
    (["open notepad and type hi"], ["run", "open notepad and type hi"]),
    (["tasks"], ["tasks"]),
    (["--dry-run", "run", "x"], ["run", "--dry-run", "x"]),
    (["--model", "m", "resume"], ["resume", "--model", "m"]),
    (["--data-dir", "d", "open", "the", "tasks", "app"],
     ["run", "--data-dir", "d", "open", "the", "tasks", "app"]),
])
def test_route(argv, expected):
    assert cli.route(argv) == expected


def test_routed_args_parse():
    args = cli.build_parser().parse_args(cli.route(["--dry-run", "open", "notepad"]))
    assert args.command == "run" and args.dry_run and args.goal == ["open", "notepad"]


@pytest.mark.asyncio
async def test_ctrl_c_interrupts_the_task_only():
    cancelled = []

    async def long_task():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    before = signal.getsignal(signal.SIGINT)
    asyncio.get_running_loop().call_later(0.1, signal.raise_signal, signal.SIGINT)
    try:
        with pytest.raises(cli.Interrupted):
            await cli.run_interruptible(long_task())
        assert cancelled == [True]
        assert CONTROL.aborted                      # mouse/keyboard told to stop
        assert signal.getsignal(signal.SIGINT) is before

        # The next task starts clean and runs normally.
        async def quick():
            return "ok"
        assert await cli.run_interruptible(quick()) == "ok"
        assert not CONTROL.aborted
    finally:
        CONTROL.reset()
