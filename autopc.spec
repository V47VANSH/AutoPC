# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AutoPC (onedir).

Build:   pyinstaller autopc.spec     (or: build.cmd)
Output:  dist/AutoPC/AutoPC.exe      (+ an _internal/ folder beside it)

The entry point is the CLI (autopc/__main__.py), so the built exe behaves
exactly like the `autopc` command:

    AutoPC.exe                      # open the interactive shell
    AutoPC.exe run "open notepad"   # run one goal, then exit
    AutoPC.exe skills               # list learned skills
    AutoPC.exe -h                   # help

It is a CONSOLE app on purpose -- it IS a terminal tool.

## What is NOT inside the exe (must exist on the target machine)
  * the Interception driver (system install + reboot; can't be bundled)
  * Claude credentials for the Agent SDK (see README "Authentication")
  * .env.local -- put it NEXT TO the .exe (editable); it is read from there.
    Learned state goes in a `data` folder next to the exe unless
    AUTOPC_DATA_DIR says otherwise.

The SDK's bundled CLI is collected (no separate Node/CLI needed). The embedding
model is downloaded at first run into the data dir, not bundled.
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

sys.path.insert(0, SPECPATH)                          # noqa: F821 (SPECPATH injected)

APP_NAME = "AutoPC"

datas, binaries, hiddenimports = [], [], []

# Packages PyInstaller cannot fully trace on its own:
#   claude_agent_sdk  -- bundled CLI + dynamic imports
#   comtypes          -- dxcam's COM codegen
#   dxcam, winsdk     -- capture + WinRT OCR
#   cv2               -- OpenCV
#   faiss             -- compiled _swigfaiss extension
#   fastembed         -- ONNX embeddings; pulls onnxruntime + tokenizers
#   onnxruntime       -- native runtime + providers
#   tokenizers        -- rust extension
#   huggingface_hub   -- fastembed uses it to fetch the model on first run
# NOTE: do NOT collect_all("mcp") -- mcp/cli/cli.py sys.exit(1)s at import when
# typer is missing. The SDK only needs mcp's server side; it is traced normally.
for pkg in ("claude_agent_sdk", "comtypes", "dxcam", "winsdk", "cv2",
            "faiss", "fastembed", "onnxruntime", "tokenizers", "huggingface_hub"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:
        print(f"[autopc.spec] collect_all({pkg!r}) skipped: {exc}")

# The Interception user-mode helper DLL, bundled if present in the repo.
_dll = os.path.join(SPECPATH, "interception.dll")     # noqa: F821
if os.path.exists(_dll):
    binaries += [(_dll, ".")]

a = Analysis(
    [os.path.join("autopc", "__main__.py")],
    pathex=[SPECPATH],                               # noqa: F821
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + collect_submodules("autopc") + [
        "winsdk.windows.media.ocr",
        "winsdk.windows.graphics.imaging",
        "winsdk.windows.storage.streams",
        "comtypes.stream",
        "faiss", "faiss._swigfaiss",
        "yaml",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "mcp.cli", "typer",   # mcp's CLI sys.exit(1)s at import; SDK doesn't need it
        "tkinter", "matplotlib", "IPython", "pytest", "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # onedir: deps go in the _internal folder
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
