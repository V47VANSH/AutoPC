"""`python -m autopc` -- same as the `autopc` command. Also the PyInstaller entry."""
import sys

from autopc.cli import main

if __name__ == "__main__":
    sys.exit(main())
