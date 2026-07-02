"""
Vision-based element proposal: find the clickable controls in a screenshot.

The agent points at pixels it reads off the screenshot. That is reliable for
large targets and for text (whose exact coordinates come from OCR), but a
model's pixel estimate can be several pixels off, which is enough to miss a
24px icon. So the executor snaps a click that lands inside a small detected
control to that control's exact centre (see `Executor._snap_to_element`). This
module supplies those detections from the pixels alone -- no accessibility
tree -- so it behaves the same in every app, including canvas apps, games and
remote sessions that do not expose one.

## Proposal strategy

UI is not natural imagery, so generic object detection is the wrong tool.
Interface controls have exploitable structure: hard edges, axis-aligned
rectangles, flat fills, and text on uniform backgrounds. We run several cheap
detectors that fail in different ways and merge them:

  * `_propose_edges`     -- Canny + morphological close, for bordered controls
  * `_propose_mser`      -- maximally stable regions, for text runs and glyph
                            icons on flat backgrounds
  * `_propose_uniform`   -- flat-fill segmentation, for borderless flat-design
                            buttons that carry no edge signal at all
  * `_propose_gradient`  -- local contrast peaks, for small dense icon groups
                            (toolbars, window controls, three-dot menus)

Merging is non-maximum suppression with containment collapsing, so a button
found by three detectors yields one box, and a toolbar containing eight icons
does not swallow them.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # (left, top, right, bottom)


# ── OmniParser: a learned interactable-element detector ────────────────────
#
# Optional 5th proposal source. The 4 classical detectors find nearly every
# control but cannot tell a button from a decorative edge, so junk competes with
# real controls for the capped candidate list. OmniParser's YOLOv9-E model is
# trained to find *interactable* regions specifically, so it raises the
# precision of what makes the cut. It runs on the same captured frame.
#
# Off by default. Enable with AUTOPC_OMNIPARSER=1. Everything degrades to the
# classical CV path if torch / the weights / the repo are absent -- exactly like
# OCR degrades when winsdk is missing.
_OMNI_HOME_DEFAULT = ""
_OMNI_SINGLETON = {"tried": False, "detector": None}


def _omniparser_detector():
    """Lazy, process-wide singleton. Returns the detector or None (degrade)."""
    if _OMNI_SINGLETON["tried"]:
        return _OMNI_SINGLETON["detector"]
    _OMNI_SINGLETON["tried"] = True
    try:
        home = os.environ.get("AUTOPC_OMNIPARSER_HOME", _OMNI_HOME_DEFAULT)
        weights = os.environ.get(
            "AUTOPC_OMNIPARSER_WEIGHTS",
            os.path.join(home, "weights", "icon_detect_v3", "model.pt"),
        )
        try:
            import torch
            has_cuda = bool(torch.cuda.is_available())
        except Exception:
            has_cuda = False
        device = os.environ.get("AUTOPC_OMNIPARSER_DEVICE", "cuda" if has_cuda else "cpu")
        if home not in sys.path:
            sys.path.insert(0, home)
        from util.yolov9 import YOLOv9Detector  # MIT, minimal deps (torch/tv/hf)

        det = YOLOv9Detector(model_path=weights, device=device)
        _OMNI_SINGLETON["detector"] = det
        log.info("OmniParser detector loaded (%s, device=%s).", weights, device)
    except Exception as exc:  # pragma: no cover -- environment dependent
        log.warning("OmniParser unavailable; using classical CV only: %s", exc)
        _OMNI_SINGLETON["detector"] = None
    return _OMNI_SINGLETON["detector"]


# The detector can also run out-of-process under a CUDA-enabled Python (see
# core/omniparser_client.py) so the main venv stays CPU-only. Toggled with
# AUTOPC_OMNIPARSER_SIDECAR=1 + AUTOPC_OMNIPARSER_PYTHON=<gpu venv python>.
_OMNI_SIDECAR = {"tried": False, "client": None}


def _omniparser_sidecar():
    if _OMNI_SIDECAR["tried"]:
        return _OMNI_SIDECAR["client"]
    _OMNI_SIDECAR["tried"] = True
    try:
        gpu_python = os.environ.get("AUTOPC_OMNIPARSER_PYTHON", "")
        if not gpu_python or not os.path.isfile(gpu_python):
            raise FileNotFoundError(
                f"AUTOPC_OMNIPARSER_PYTHON not set or missing: {gpu_python!r}")
        from autopc.core.omniparser_client import OmniParserSidecar

        client = OmniParserSidecar(
            python_exe=gpu_python,
            device=os.environ.get("AUTOPC_OMNIPARSER_DEVICE", "cuda"),
        )
        import atexit
        atexit.register(client.close)  # don't leak the GPU child on exit
        _OMNI_SIDECAR["client"] = client
        log.info("OmniParser sidecar configured (%s).", gpu_python)
    except Exception as exc:  # pragma: no cover -- environment dependent
        log.warning("OmniParser sidecar unavailable; using classical CV only: %s", exc)
        _OMNI_SIDECAR["client"] = None
    return _OMNI_SIDECAR["client"]


def _omni_predict_boxes(pil_image, imgsz: int, conf: float, iou: float = 0.7) -> List[Box]:
    """Interactable-region boxes from OmniParser, via sidecar or in-process."""
    if os.environ.get("AUTOPC_OMNIPARSER_SIDECAR", "0") == "1":
        client = _omniparser_sidecar()
        if client is None:
            return []
        return client.predict_boxes(pil_image, imgsz=imgsz, conf=conf, iou=iou)
    det = _omniparser_detector()
    if det is None:
        return []
    res = det.predict(source=pil_image.convert("RGB"), conf=conf, imgsz=imgsz,
                      iou=iou, max_det=300)
    return [tuple(int(v) for v in b) for b in res[0].boxes.xyxy.tolist()]


def warm_omniparser() -> bool:
    """Trigger backend load (sidecar spawn or in-process weights). Returns ready.

    For the sidecar this forces the actual subprocess spawn + model load now, so
    the one-time cold start does not land inside the first real observe.
    """
    if os.environ.get("AUTOPC_OMNIPARSER_SIDECAR", "0") == "1":
        client = _omniparser_sidecar()
        if client is None:
            return False
        if Image is not None:  # force spawn + load + kernel warmup up front
            client.predict_boxes(Image.new("RGB", (64, 64)), imgsz=640, conf=0.9)
        return client.proc is not None
    return _omniparser_detector() is not None


def warm_omniparser_async() -> None:
    """Warm the backend in a daemon thread so boot and the first observe never
    wait on the one-time model load. No-op when the detector is disabled."""
    if os.environ.get("AUTOPC_OMNIPARSER", "0") != "1":
        return
    import threading

    threading.Thread(target=warm_omniparser, name="omniparser-warm",
                     daemon=True).start()


@dataclass
class VisionCandidate:
    """A proposed clickable region. Geometry is exact; semantics are the VLM's job."""

    id: int
    rect: Box
    score: float = 0.0
    sources: Tuple[str, ...] = ()
    hint: str = ""          # coarse shape guess (icon/button/text-run), never authoritative
    text: str = ""          # OCR-read label, when confidently attributed -- see core/ocr.py

    def __post_init__(self):
        # OpenCV hands back numpy integers, which survive all the way into
        # action payloads and then fail json.dumps ("np.int32 is not JSON
        # serializable") when the trace is logged or sent to the model.
        # Normalise once, here, rather than at every consumer.
        self.rect = tuple(int(v) for v in self.rect)  # type: ignore[assignment]

    @property
    def center(self) -> Tuple[int, int]:
        l, t, r, b = self.rect
        return (l + r) // 2, (t + b) // 2

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _contains(outer: Box, inner: Box, slack: int = 2) -> bool:
    return (outer[0] - slack <= inner[0] and outer[1] - slack <= inner[1]
            and outer[2] + slack >= inner[2] and outer[3] + slack >= inner[3])


class VisionGrounder:
    """
    Proposes clickable regions from a screenshot alone.

    Tuned for desktop UI at 1080p-1440p. Thresholds are expressed relative to
    image size where it matters so other resolutions degrade gracefully.
    """

    # A control smaller than this is almost certainly noise; larger than this
    # is a layout container, not a target.
    MIN_SIDE = 8
    MAX_SIDE_FRAC = 0.6
    MIN_AREA = 90

    def __init__(self, max_candidates: int = 60, use_omniparser: Optional[bool] = None):
        self.max_candidates = max_candidates
        if use_omniparser is None:
            use_omniparser = os.environ.get("AUTOPC_OMNIPARSER", "0") == "1"
        self.use_omniparser = bool(use_omniparser)

    # -- public API ---------------------------------------------------------

    def propose(self, screenshot: "Image.Image") -> List[VisionCandidate]:
        """Run every detector, merge, rank, and number the survivors."""
        if cv2 is None:
            log.error("OpenCV unavailable; vision grounding cannot run.")
            return []

        bgr = self._to_bgr(screenshot)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        proposals: List[Tuple[Box, str]] = []
        for name, fn in (
            ("edge", self._propose_edges),
            ("mser", self._propose_mser),
            ("flat", self._propose_uniform),
            ("icon", self._propose_gradient),
        ):
            try:
                for box in fn(gray, bgr):
                    proposals.append((box, name))
            except Exception as exc:
                log.warning("Vision detector %s failed: %s", name, exc)

        # Learned interactable detector (optional). Its boxes carry the "ui"
        # source tag, which _rank rewards heavily -- a control the model was
        # trained to recognise is a far stronger prior than any single classical
        # detector firing. Runs on the same frame; no OS/accessibility calls.
        if self.use_omniparser:
            try:
                for box in self._propose_omniparser(screenshot, (w, h)):
                    proposals.append((box, "ui"))
            except Exception as exc:
                log.warning("OmniParser source failed: %s", exc)

        merged = self._merge(proposals, (w, h))
        ranked = self._rank(merged, (w, h))
        for idx, cand in enumerate(ranked, start=1):
            cand.id = idx
        return ranked

    # -- detectors ----------------------------------------------------------

    def _propose_omniparser(self, screenshot: "Image.Image", size: Tuple[int, int]) -> List[Box]:
        """
        Learned interactable-region proposals from the OmniParser YOLOv9-E model.

        Boxes come back in the image's pixel space, so they slot straight into
        the same NMS/merge path as the classical detectors. Detection only --
        we never load the caption model, so VRAM and latency stay low (weights
        are ~280MB; box inference is a single forward pass).

        Tunable via env: AUTOPC_OMNIPARSER_IMGSZ (default 1280 -- a higher size recovers
        small icons that a 640 pass shrinks below detectability), AUTOPC_OMNIPARSER_CONF
        (default 0.05).
        """
        if Image is None:
            return []
        w, h = size
        imgsz = int(os.environ.get("AUTOPC_OMNIPARSER_IMGSZ", "1280"))
        conf = float(os.environ.get("AUTOPC_OMNIPARSER_CONF", "0.05"))
        out: List[Box] = []
        for x1, y1, x2, y2 in _omni_predict_boxes(screenshot, imgsz, conf):
            if x2 <= x1 or y2 <= y1:
                continue
            if self._plausible(x2 - x1, y2 - y1, (h, w)):
                out.append((x1, y1, x2, y2))
        return out

    def _propose_edges(self, gray: np.ndarray, bgr: np.ndarray) -> List[Box]:
        """Bordered controls: buttons, fields, framed panels."""
        # Median-derived thresholds adapt to light and dark themes alike,
        # which fixed thresholds do not.
        med = float(np.median(gray))
        lo = int(max(0, 0.66 * med))
        hi = int(min(255, 1.33 * med))
        edges = cv2.Canny(gray, lo, hi, apertureSize=3)

        # Close small gaps so a dashed or anti-aliased border forms one contour.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Box] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not self._plausible(w, h, gray.shape):
                continue
            # Reject wildly non-rectangular blobs: real controls fill their box.
            fill = cv2.contourArea(c) / float(w * h) if w * h else 0
            if fill < 0.12:
                continue
            out.append((x, y, x + w, y + h))
        return out

    def _propose_mser(self, gray: np.ndarray, bgr: np.ndarray) -> List[Box]:
        """Text runs and glyph icons -- stable regions on flat backgrounds."""
        h, w = gray.shape
        mser = cv2.MSER_create()
        try:
            mser.setMinArea(self.MIN_AREA)
            mser.setMaxArea(int(w * h * 0.05))
        except AttributeError:
            pass

        regions, _ = mser.detectRegions(gray)
        raw: List[Box] = []
        for pts in regions:
            x, y, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if not self._plausible(rw, rh, gray.shape):
                continue
            raw.append((x, y, x + rw, y + rh))

        # Individual glyphs are useless as click targets; merge characters that
        # sit on a shared baseline into word/label boxes.
        return self._merge_text_runs(raw)

    def _propose_uniform(self, gray: np.ndarray, bgr: np.ndarray) -> List[Box]:
        """
        Borderless flat-design controls.

        Modern UI frequently draws a button as a solid fill with no border, so
        it produces no edge signal at all. Those regions are still detectable
        as locally uniform patches that differ from their surroundings.
        """
        # Variance-based flatness: low local stddev means a flat fill.
        blur = cv2.blur(gray, (7, 7))
        sq = cv2.blur(gray.astype(np.float32) ** 2, (7, 7))
        var = np.clip(sq - blur.astype(np.float32) ** 2, 0, None)
        flat = (var < 12).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
        out: List[Box] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not self._plausible(w, h, gray.shape):
                continue
            if area < self.MIN_AREA:
                continue
            # A flat patch filling its bounding box is a control-like slab.
            if area / float(w * h) < 0.6:
                continue
            out.append((x, y, x + w, y + h))
        return out

    def _propose_gradient(self, gray: np.ndarray, bgr: np.ndarray) -> List[Box]:
        """
        Dense small icons -- toolbars, window controls, overflow menus.

        These are the targets the old grid approach missed most often, so they
        get a dedicated pass tuned for small high-contrast glyphs rather than
        being left to the general detectors.
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, strong = cv2.threshold(mag, 60, 255, cv2.THRESH_BINARY)
        # Dilate so the strokes of one glyph coalesce, but not across icons.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blobs = cv2.dilate(strong, kernel, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
        out: List[Box] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < self.MIN_SIDE or h < self.MIN_SIDE:
                continue
            # Icons are compact and roughly square.
            if w > 96 or h > 96:
                continue
            ar = w / float(h)
            if ar < 0.35 or ar > 2.8:
                continue
            out.append((x, y, x + w, y + h))
        return out

    # -- merging and ranking ------------------------------------------------

    def _plausible(self, w: int, h: int, shape) -> bool:
        ih, iw = shape
        if w < self.MIN_SIDE or h < self.MIN_SIDE:
            return False
        if w * h < self.MIN_AREA:
            return False
        if w > iw * self.MAX_SIDE_FRAC and h > ih * self.MAX_SIDE_FRAC:
            return False
        ar = w / float(h)
        return 0.05 <= ar <= 30

    @staticmethod
    def _merge_text_runs(boxes: List[Box], gap: int = 10) -> List[Box]:
        """Join glyph boxes sharing a baseline into label-sized boxes."""
        if not boxes:
            return []
        # Group by row band, then sweep left-to-right joining near neighbours.
        boxes = sorted(boxes, key=lambda b: (b[1] // 12, b[0]))
        merged: List[Box] = []
        cur = list(boxes[0])
        for b in boxes[1:]:
            same_row = abs(b[1] - cur[1]) <= 8 and abs(b[3] - cur[3]) <= 10
            near = b[0] - cur[2] <= gap
            if same_row and near:
                cur[0] = min(cur[0], b[0])
                cur[1] = min(cur[1], b[1])
                cur[2] = max(cur[2], b[2])
                cur[3] = max(cur[3], b[3])
            else:
                merged.append(tuple(cur))  # type: ignore[arg-type]
                cur = list(b)
        merged.append(tuple(cur))  # type: ignore[arg-type]
        return merged

    def _merge(self, proposals: List[Tuple[Box, str]], size) -> List[VisionCandidate]:
        """
        Non-maximum suppression with source tracking.

        A region found by several independent detectors is more likely to be a
        real control, so agreement raises its score rather than being discarded
        as a duplicate.
        """
        if not proposals:
            return []

        # Largest first so containment collapsing is well-defined.
        proposals = sorted(proposals, key=lambda p: -((p[0][2] - p[0][0]) * (p[0][3] - p[0][1])))

        kept: List[VisionCandidate] = []
        for box, src in proposals:
            hit = None
            for cand in kept:
                if self._same_target(cand.rect, box):
                    hit = cand
                    break
            if hit is not None:
                if src not in hit.sources:
                    hit.sources = hit.sources + (src,)
                    hit.score += 1.0
                continue
            kept.append(VisionCandidate(id=0, rect=box, score=1.0, sources=(src,)))

        return kept

    @staticmethod
    def _same_target(a: Box, b: Box) -> bool:
        """
        Do two boxes point at the same control?

        Plain IoU is not enough. A 14x14 box nested inside a 19x19 box on the
        identical centre scores IoU 0.54 -- just under a 0.55 threshold -- so
        both survive and the model is offered two different numbers for one
        button. Observed live: the Chrome close button appeared twice, as
        19x19 and 14x14 regions both centred on (1891, 24).

        Duplicates are worse than they look: they waste scarce candidate slots,
        and they split the model's choice between two equally-correct answers,
        which reads as indecision in the trace.

        So we also collapse when the centres nearly coincide and one box is
        substantially contained in the other -- concentric detections of the
        same glyph at different tightnesses.
        """
        if _iou(a, b) >= 0.55:
            return True

        acx, acy = (a[0] + a[2]) // 2, (a[1] + a[3]) // 2
        bcx, bcy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        if abs(acx - bcx) > 6 or abs(acy - bcy) > 6:
            return False

        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        inner, outer = (a, b) if area_a <= area_b else (b, a)
        inner_area = min(area_a, area_b)

        ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return overlap / inner_area >= 0.8

    def _rank(self, cands: List[VisionCandidate], size) -> List[VisionCandidate]:
        """
        Score for clickability, then select for *spatial coverage*.

        Pure score-ordering is a trap here, and it was measured: ranking by
        score alone and taking the top 60 gave 17.6% centre-recall against accessibility-tree
        ground truth, because the 60 highest scorers were 60 near-identical
        toolbar icons clustered in two strips. The detectors themselves find
        98.9% of real controls -- the ranking was discarding the answers.

        So selection is diversified over a coarse spatial grid: repeatedly take
        the best remaining candidate from each occupied cell. Coverage of the
        screen matters more than the marginal quality of any one box, because
        a target the model never sees is a guaranteed miss while a slightly
        loose box is still clickable at its centre.
        """
        w, h = size
        img_area = w * h

        for c in cands:
            score = c.score * 2.0                       # detector agreement

            # A learned interactable detector is a far stronger prior than any
            # single classical detector firing, so a box it proposed should win
            # a scarce candidate slot over an unconfirmed edge blob. This bonus is
            # the mechanism by which OmniParser raises wide-view precision; it is
            # inert when the "ui" source is disabled.
            if "ui" in c.sources:
                score += float(os.environ.get("AUTOPC_OMNIPARSER_RANK_BONUS", "6.0"))

            ar = c.width / float(max(c.height, 1))

            if 0.7 <= ar <= 1.4 and c.width <= 64:
                score += 2.0
                c.hint = "icon"
            elif 1.4 < ar <= 8 and 14 <= c.height <= 56:
                score += 2.0
                c.hint = "button/label"
            elif ar > 8:
                score += 0.5
                c.hint = "text-run"
            else:
                c.hint = "region"

            # Containers are rarely the target and crowd out real controls.
            frac = c.area / float(img_area)
            if frac > 0.12:
                score -= 4.0
            elif frac > 0.04:
                score -= 1.5

            if c.area < 4000:
                score += 0.75

            # Continuous tie-breakers. Without these the score is built purely
            # from discrete bumps and collides constantly -- measured: ten
            # candidates in one cell all at exactly 6.75, so ordering within
            # the cell fell back to insertion order and the real target lost
            # arbitrarily. These terms are small enough not to reorder
            # genuinely different candidates, but large enough to break ties.
            #
            # Compactness: real controls fill their box; stray edge fragments
            # are thin slivers.
            compact = min(c.width, c.height) / max(c.width, c.height, 1)
            score += 0.4 * compact

            # Prefer tighter boxes among otherwise-equal candidates, since a
            # tight box centres more accurately on its control.
            score += 0.3 * (1.0 - min(1.0, c.area / 8000.0))

            c.score = score

        # Explicit secondary keys so equal scores still order deterministically
        # rather than by whatever the detectors happened to emit first.
        cands.sort(key=lambda c: (-c.score, -len(c.sources), c.area))
        if len(cands) <= self.max_candidates:
            return cands
        return self._select_diverse(cands, size)

    # Desktop UI is organised in horizontal strips -- title bar, tab strip,
    # toolbar, status bar -- stacked only tens of pixels apart. A square grid
    # lumps several strips into one cell and then keeps a single winner from
    # all of them. Measured failure: VS Code's title bar (y=21) and the
    # toolbar beneath it (y=65) shared one 192x108 cell, so the toolbar took
    # the slot and the window's Close button was dropped despite being
    # detected. More rows than columns keeps strips in separate cells.
    GRID_COLS = 12
    GRID_ROWS = 20

    # Share of the budget reserved for small controls that lost their cell.
    SMALL_RESERVE = 0.25
    SMALL_AREA = 2600

    def _select_diverse(self, ranked: List[VisionCandidate], size) -> List[VisionCandidate]:
        """
        Select for screen coverage, then rescue starved small controls.

        Two phases, because they fail in opposite directions:

        **Phase A -- coverage.** Round-robin the best candidate from each grid
        cell. Without this, score-ordering returns 60 near-identical toolbar
        icons from one strip (measured: 17.6% centre-recall vs 48% with it).

        **Phase B -- small-control rescue.** Coverage alone starves dense
        clusters of genuinely distinct controls, because a cell yields roughly
        `budget / occupied_cells` candidates -- about 1.2 in practice. Window
        controls, overflow menus and close buttons are both small AND
        clustered, which is exactly the losing combination, and they are the
        targets that matter most. So a slice of the budget is reserved for the
        highest-scoring small candidates regardless of which cell they are in.
        """
        w, h = size
        cw = max(1, w // self.GRID_COLS)
        ch = max(1, h // self.GRID_ROWS)

        buckets: Dict[Tuple[int, int], List[VisionCandidate]] = {}
        for c in ranked:
            cx, cy = c.center
            buckets.setdefault((cx // cw, cy // ch), []).append(c)

        reserve = int(self.max_candidates * self.SMALL_RESERVE)
        coverage_budget = self.max_candidates - reserve

        chosen: List[VisionCandidate] = []
        picked: set = set()

        # Phase A -- one per cell per pass, cells already score-ordered.
        depth = 0
        while len(chosen) < coverage_budget:
            progressed = False
            for key in sorted(buckets.keys()):
                group = buckets[key]
                if depth < len(group):
                    cand = group[depth]
                    chosen.append(cand)
                    picked.add(id(cand))
                    progressed = True
                    if len(chosen) >= coverage_budget:
                        break
            if not progressed:
                break
            depth += 1

        # Phase B -- best remaining small candidates, cell quotas ignored.
        for c in ranked:
            if len(chosen) >= self.max_candidates:
                break
            if id(c) in picked or c.area > self.SMALL_AREA:
                continue
            chosen.append(c)
            picked.add(id(c))

        # Phase C -- any leftover budget goes to the best remaining anything.
        for c in ranked:
            if len(chosen) >= self.max_candidates:
                break
            if id(c) in picked:
                continue
            chosen.append(c)
            picked.add(id(c))

        chosen.sort(key=lambda c: -c.score)
        return chosen[: self.max_candidates]

    @staticmethod
    def _to_bgr(screenshot) -> np.ndarray:
        arr = np.array(screenshot)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if arr.ndim == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
