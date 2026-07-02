"""
OCR -- native Windows text recognition (Windows.Media.Ocr via winsdk).

## Why this exists

Vision region proposal (`core/vision_grounding.py`) finds *where* things are
but has no idea *what they say*. A region labelled only `[14] icon @(880,210)`
forces the model to recognise "Settings" from a gear glyph; a region labelled
`[14] icon "Settings" @(880,210)` lets it read the answer. That is the
difference between recognition and OCR, and OCR is the reliable one whenever
the control actually has a text label.

## Why Windows.Media.Ocr and not a Python OCR library

Measured on a typical desktop: 0.27s for a full 1920x1080 screenshot, 10ms for a
200x60 crop, and it correctly read UI chrome, browser tab titles and body text
with no false "confident nonsense" in a quick check. It ships with Windows,
needs no model download, runs fully offline, and costs nothing per call --
unlike routing every screenshot through a vision LLM, which is both slower and
priced per image. It fits the same "prefer native OS APIs" posture as the
DXGI capture and Interception driver already in this project.

## The confusion constraint

The user's explicit requirement: OCR labels must not confuse the model. Two
disciplines enforce that:

  1. **Attribution, not replacement.** OCR text is attached to a candidate's
     `.text` field, separate from `.hint` (the shape-based guess). The
     detector's geometry is never altered by what OCR reads.
  2. **Confidence gating.** A line is only attached if it plausibly labels a
     control: short, mostly overlapping the candidate's box, not a stray
     fragment of unrelated body text sitting nearby. See `_looks_like_label`.
     When OCR finds nothing trustworthy, the candidate is left unlabelled
     rather than guessing -- an absent label costs nothing; a wrong one costs
     a misclick.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]

try:
    import winsdk.windows.media.ocr as _wocr
    import winsdk.windows.graphics.imaging as _wimg
    import winsdk.windows.storage.streams as _wstreams
    import winsdk.windows.globalization as _wglob
    _WINSDK_AVAILABLE = True
except ImportError:
    _WINSDK_AVAILABLE = False
    log.warning("winsdk not installed. OCR labelling and read_screen are disabled. "
                "Run 'pip install winsdk' to enable (Windows only).")


@dataclass
class OcrWord:
    text: str
    rect: Box


@dataclass
class OcrLine:
    text: str
    rect: Box                    # union of its words' rects
    words: List[OcrWord]


class OcrEngine:
    """
    Thin synchronous wrapper over Windows' native OCR.

    winsdk's calls are async (WinRT), and this codebase is otherwise
    synchronous, so each call spins a private event loop via
    `asyncio.run(...)`. A lock serialises calls: the underlying
    OcrEngine/BitmapDecoder objects are not documented as thread-safe, and
    this is called both from the main planning loop and from background
    critic threads.
    """

    def __init__(self, language: str = "en-US"):
        self.available = _WINSDK_AVAILABLE
        self._engine = None
        self._lock = threading.Lock()
        self._language = language
        if self.available:
            self._engine = self._create_engine(language)
            self.available = self._engine is not None

    @staticmethod
    def _create_engine(language: str):
        try:
            engine = _wocr.OcrEngine.try_create_from_user_profile_languages()
            if engine is not None:
                return engine
            return _wocr.OcrEngine.try_create_from_language(_wglob.Language(language))
        except Exception as exc:
            log.warning("OCR engine creation failed: %s", exc)
            return None

    def read(self, image) -> List[OcrLine]:
        """
        Recognise all text in a PIL image. Returns [] if OCR is unavailable
        or the call fails -- callers must treat that as "no labels", not an
        error, since OCR is an enhancement layer, not a required dependency.
        """
        if not self.available:
            return []
        with self._lock:
            try:
                return asyncio.run(self._read_async(image))
            except Exception as exc:
                log.warning("OCR read failed (non-fatal): %s", exc)
                return []

    async def _read_async(self, image) -> List[OcrLine]:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()

        stream = _wstreams.InMemoryRandomAccessStream()
        writer = _wstreams.DataWriter(stream.get_output_stream_at(0))
        writer.write_bytes(bytearray(data))
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)

        decoder = await _wimg.BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await self._engine.recognize_async(bitmap)

        lines: List[OcrLine] = []
        for line in result.lines:
            words = []
            for w in line.words:
                r = w.bounding_rect
                words.append(OcrWord(
                    text=w.text,
                    rect=(int(r.x), int(r.y), int(r.x + r.width), int(r.y + r.height)),
                ))
            if not words:
                continue
            l = min(w.rect[0] for w in words)
            t = min(w.rect[1] for w in words)
            r = max(w.rect[2] for w in words)
            b = max(w.rect[3] for w in words)
            lines.append(OcrLine(text=line.text, rect=(l, t, r, b), words=words))
        return lines


# A process-wide singleton: engine construction has measurable overhead and
# there is no reason for perception and the reader to each build one.
_default_engine: Optional[OcrEngine] = None
_default_lock = threading.Lock()


def get_default_engine() -> OcrEngine:
    global _default_engine
    if _default_engine is None:
        with _default_lock:
            if _default_engine is None:
                _default_engine = OcrEngine()
    return _default_engine


# ---------------------------------------------------------------------------
# Region labelling
# ---------------------------------------------------------------------------

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
    return inter / max(1, area_a + area_b - inter)


def _overlap_fraction_of_line(line_rect: Box, cand_rect: Box) -> float:
    """What fraction of the OCR line sits inside the candidate's box."""
    lx1, ly1, lx2, ly2 = line_rect
    cx1, cy1, cx2, cy2 = cand_rect
    ix1, iy1 = max(lx1, cx1), max(ly1, cy1)
    ix2, iy2 = min(lx2, cx2), min(ly2, cy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    line_area = max(1, (lx2 - lx1) * (ly2 - ly1))
    return inter / line_area


def _looks_like_label(text: str) -> bool:
    """
    Reject text that is not plausibly a control's label.

    This is the confusion guard: a stray fragment of a paragraph brushing
    against an icon's box must not become that icon's name. Labels are short
    and word-like; body text is long or punctuation-heavy.

    A bare single letter or a short run of digits is rejected outright.
    Measured live against a VS Code + file-explorer desktop: git status
    badges ("o", "U"), gutter line numbers ("4", "6", "11") and a temperature
    reading ("280C" truncated to digits) all satisfied the earlier looser
    check and attached themselves to unrelated 14-20px icons as if they were
    real names. A single symbol ("+", "x", "-") is kept, since those genuinely
    label real controls (add/close/collapse) and are not the noise pattern
    observed. Pagination buttons labelled by a bare digit ("2") are an
    accepted false negative -- given an explicit instruction that labels must
    not confuse the model, silently dropping a real short label is the safer
    failure than attaching a wrong one.
    """
    t = text.strip()
    if not t or len(t) > 28:
        return False
    if t.count(" ") > 3:
        return False
    if len(t) == 1 and t.isalpha():
        return False
    if t.isdigit() and len(t) <= 2:
        return False
    alnum = sum(1 for c in t if c.isalnum())
    return alnum >= max(1, len(t) // 2)


def label_candidates(
    candidates,
    image=None,
    engine: Optional[OcrEngine] = None,
    min_overlap: float = 0.55,
    min_candidate_coverage: float = 0.06,
    lines: Optional[List[OcrLine]] = None,
) -> int:
    """
    Attach OCR text to candidates in place, where it plausibly labels them.

    For each OCR line, find the single best-overlapping candidate (if any
    clears `min_overlap` of the *line's own area* inside the candidate's box)
    and attach the line's text to `.text` -- but only when it survives
    `_looks_like_label`, and only when no stronger claim from another line
    already won that candidate (highest overlap wins, so a candidate never
    ends up with two conflicting labels).

    `min_candidate_coverage` guards a second, different confusion case: a
    large container (a file-tree panel, a whole toolbar) will trivially
    satisfy `min_overlap` for the first text line it happens to contain,
    which would make a 545x384 panel appear to be labelled "PROJECT" as if
    that were the name of the whole clickable region rather than one header
    inside it. Requiring the line to also cover a minimum fraction of the
    *candidate's* area means only a line that is label-sized relative to its
    box can attach -- a snug button label passes easily; a stray line inside
    a much bigger container does not.

    A third case -- watermarks and other repeated overlays -- is guarded
    after assignment: if the exact same text ends up attached to several
    unrelated candidates in one frame (a tiled "CONFIDENTIAL"/"SAMPLE" stamp,
    a repeated document footer), real distinct controls
    essentially never share one identical label that many times, so every
    instance is dropped rather than trusted as three coincidentally identical
    button names.

    Mutates `candidates` (each must expose `.rect` and a settable `.text`
    attribute) and returns the number labelled.

    Deliberately many-lines-to-one-candidate, not the reverse: several OCR
    words can belong to one button ("Save", "As...") but one line should never
    be smeared across two different candidates.

    Pass `lines` directly to skip a fresh OCR pass when the caller already has
    one for this exact frame (see `PerceptionEngine`'s per-screenshot cache) --
    otherwise `image` is read via `engine.read()`.
    """
    if not candidates:
        return 0
    if lines is None:
        eng = engine or get_default_engine()
        if not eng.available or image is None:
            return 0
        lines = eng.read(image)
    if not lines:
        return 0

    best_for: dict = {}   # id(candidate) -> (overlap, text)
    for line in lines:
        if not _looks_like_label(line.text):
            continue
        best_cand = None
        best_score = 0.0
        for c in candidates:
            frac = _overlap_fraction_of_line(line.rect, c.rect)
            if frac < min_overlap:
                continue
            cand_area = max(1, (c.rect[2] - c.rect[0]) * (c.rect[3] - c.rect[1]))
            line_area = max(1, (line.rect[2] - line.rect[0]) * (line.rect[3] - line.rect[1]))
            if (line_area / cand_area) < min_candidate_coverage:
                continue
            # Prefer the tighter-fitting candidate when several qualify --
            # a small icon nested in a large toolbar should win over the
            # toolbar itself.
            score = frac / cand_area
            if score > best_score:
                best_score, best_cand = score, c
        if best_cand is None:
            continue
        key = id(best_cand)
        if key not in best_for or best_score > best_for[key][0]:
            best_for[key] = (best_score, line.text.strip())

    # Watermark/overlay guard: the same exact text attached to several
    # different candidates in one frame is a repeating-pattern signal, not
    # coincidence -- drop every instance.
    text_counts: dict = {}
    for _, text in best_for.values():
        norm = " ".join(text.lower().split())
        text_counts[norm] = text_counts.get(norm, 0) + 1
    repeated = {t for t, n in text_counts.items() if n >= 3}
    if repeated:
        dropped = [k for k, (_, t) in best_for.items() if " ".join(t.lower().split()) in repeated]
        for k in dropped:
            del best_for[k]
        log.debug("OCR labelling: dropped %d candidate(s) sharing repeated text %s "
                   "(likely watermark/overlay).", len(dropped), sorted(repeated))

    count = 0
    for c in candidates:
        hit = best_for.get(id(c))
        if hit:
            c.text = hit[1]
            count += 1
    return count


def read_screen_text(image, engine: Optional[OcrEngine] = None) -> str:
    """
    Read all text on screen, top-to-bottom, as a plain block.

    Used by the scroll-and-read primitive (core/reader.py) to assemble long
    content that does not fit in one viewport.
    """
    eng = engine or get_default_engine()
    lines = eng.read(image)
    if not lines:
        return ""
    lines_sorted = sorted(lines, key=lambda l: (l.rect[1], l.rect[0]))
    return "\n".join(l.text for l in lines_sorted if l.text.strip())
