"""
OmniParser GPU sidecar -- run the YOLOv9-E interactable-region detector under a
CUDA-enabled Python without disturbing the main venv.

AutoPC itself needs no torch. Rather than force a multi-GB CUDA torch into its
environment, the optional detector runs in a separate GPU venv as a persistent
subprocess, and we talk to it over stdin/stdout. The frame is handed across as
a temp JPEG; the reply is a JSON list of boxes. Actuation is unaffected -- this
only reads pixels we already captured.

Two roles live in this one file:

  * serve()            runs UNDER the GPU python (`python omniparser_client.py
                       --serve`). Loads the model once, then answers one JSON
                       request per line. torch is imported only here, so the
                       main venv can import this module for the client half
                       without pulling a second torch into its process.

  * OmniParserSidecar  runs in the MAIN venv. Owns the subprocess, ships frames,
                       returns boxes, and respawns the child if it dies.

Main-process env:
    AUTOPC_OMNIPARSER_SIDECAR=1
    AUTOPC_OMNIPARSER_PYTHON = <path to the GPU venv's python.exe>
    AUTOPC_OMNIPARSER_HOME / AUTOPC_OMNIPARSER_WEIGHTS   (forwarded to the server)
    AUTOPC_OMNIPARSER_DEVICE   (default "cuda")
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

_READY = "__OMNI_SIDECAR_READY__"
_DEFAULT_HOME = ""


# ── server half (executes under the GPU python) ────────────────────────────
def serve() -> None:
    import torch  # GPU torch; imported ONLY in the child process
    from PIL import Image

    home = os.environ.get("AUTOPC_OMNIPARSER_HOME", _DEFAULT_HOME)
    weights = os.environ.get(
        "AUTOPC_OMNIPARSER_WEIGHTS",
        os.path.join(home, "weights", "icon_detect_v3", "model.pt"),
    )
    device = os.environ.get(
        "AUTOPC_OMNIPARSER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if home not in sys.path:
        sys.path.insert(0, home)
    from util.yolov9 import YOLOv9Detector

    det = YOLOv9Detector(model_path=weights, device=device)
    # Warm the kernels so the first real request isn't paying compile cost.
    try:
        det.predict(source=Image.new("RGB", (640, 640)), conf=0.5, imgsz=640)
    except Exception:
        pass

    sys.stdout.write(_READY + " " + json.dumps({"device": str(device)}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            img = Image.open(req["image_path"]).convert("RGB")
            res = det.predict(
                source=img,
                conf=float(req.get("conf", 0.05)),
                imgsz=int(req.get("imgsz", 1280)),
                iou=float(req.get("iou", 0.7)),
                max_det=int(req.get("max_det", 300)),
            )
            boxes = [[int(v) for v in b] for b in res[0].boxes.xyxy.tolist()]
            reply = {"ok": True, "boxes": boxes}
        except Exception as exc:  # never crash the loop on one bad frame
            reply = {"ok": False, "error": repr(exc)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


# ── client half (executes in the main venv) ────────────────────────────────
class OmniParserSidecar:
    """Owns the GPU child process and marshals one frame at a time to it."""

    def __init__(
        self,
        python_exe: str,
        home: Optional[str] = None,
        weights: Optional[str] = None,
        device: str = "cuda",
    ):
        self.python_exe = python_exe
        home = home or os.environ.get("AUTOPC_OMNIPARSER_HOME", _DEFAULT_HOME)
        weights = weights or os.environ.get(
            "AUTOPC_OMNIPARSER_WEIGHTS",
            os.path.join(home, "weights", "icon_detect_v3", "model.pt"),
        )
        self.env = dict(os.environ)
        self.env["AUTOPC_OMNIPARSER_HOME"] = home
        self.env["AUTOPC_OMNIPARSER_WEIGHTS"] = weights
        self.env["AUTOPC_OMNIPARSER_DEVICE"] = device
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._tmp = os.path.join(
            tempfile.gettempdir(), f"autopc_omni_{os.getpid()}.jpg"
        )

    def _ensure(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        script = os.path.abspath(__file__)
        try:
            self.proc = subprocess.Popen(
                [self.python_exe, script, "--serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                env=self.env,
            )
        except Exception as exc:
            log.warning("Could not launch OmniParser sidecar: %s", exc)
            self.proc = None
            return False
        # Block until the model is loaded and warmed.
        while True:
            line = self.proc.stdout.readline()
            if not line:
                log.warning("OmniParser sidecar exited before ready.")
                self.proc = None
                return False
            if line.startswith(_READY):
                log.info("OmniParser sidecar ready: %s", line.strip())
                return True

    def predict_boxes(
        self, pil_image, imgsz: int, conf: float, iou: float = 0.7
    ) -> List[Box]:
        with self._lock:
            if not self._ensure():
                return []
            try:
                pil_image.convert("RGB").save(self._tmp, "JPEG", quality=90)
                req = {
                    "image_path": self._tmp,
                    "imgsz": int(imgsz),
                    "conf": float(conf),
                    "iou": float(iou),
                }
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    self.proc = None
                    return []
                resp = json.loads(line)
                if not resp.get("ok"):
                    log.warning("OmniParser sidecar error: %s", resp.get("error"))
                    return []
                return [tuple(int(v) for v in b) for b in resp["boxes"]]
            except Exception as exc:
                log.warning("OmniParser sidecar call failed: %s", exc)
                self.proc = None  # force respawn next time
                return []

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
