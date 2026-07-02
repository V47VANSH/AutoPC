"""
ScreenReader -- capture content that does not fit in one viewport.

## The problem this solves

An article, a long document, or a multi-paragraph instructions screen
routinely exceeds one screenshot. An agent that only ever sees the current
viewport will act on a partial read and miss whatever is stated below the fold.

## The approach: scroll, OCR, stitch

Each pass captures the screen, reads its text with the native OCR engine
(`core/ocr.py`), then scrolls down and captures again. Consecutive captures
overlap -- the bottom of frame N is the top of frame N+1 -- so the two are
stitched by finding the longest run of lines common to both, the classic
scrolling-screenshot-diff technique. Only the genuinely new tail is appended,
so the assembled text reads as one continuous document rather than repeating
every overlapping line once per scroll.

Stopping conditions, checked after every scroll:

  * the new capture is identical to the last one (nothing moved -- the
    container does not scroll, or a modal/tooltip is eating the wheel)
  * every line in the new capture already appeared in the previous one (the
    scroll moved the pixels but revealed no new text -- end of content)
  * `max_scrolls` reached, in which case the result is marked `truncated`
    so the caller knows the read may be incomplete

## Watermarks and other static overlays

A watermark pinned to the viewport (a sticky footer or page stamp,
a "SAMPLE"/"CONFIDENTIAL" tile) does not scroll with the content, so it
reappears at roughly the same text every single frame -- but not necessarily
at the head or tail of the sorted line list where the per-frame overlap match
looks, so it can slip past that check and get appended as "new" once per
scroll, littering the assembled document with N repeats of the same stamp.

`read_full` guards against this after the whole pass completes: any line that
appeared in an unusually large fraction of the frames scanned (the threshold
scales with how many frames were taken) almost certainly is not unique
scrolling content -- a real sentence appears in the 1-3 frames it was visible
in, not in most of them -- and is collapsed to a single mention. See
`_strip_repeating_overlays`.

This is deliberately a text-only capability: it answers "what does this
content say", not "where is the button" -- targeting stays with
`core/vision_grounding.py`'s region proposals, which is a different kind of
question with a different, exact answer.
"""
from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from autopc.core import ocr as _ocr

log = logging.getLogger(__name__)


@dataclass
class ReadResult:
    text: str
    scrolls_performed: int = 0
    truncated: bool = False
    duration: float = 0.0

    def summary(self) -> str:
        n_lines = self.text.count("\n") + (1 if self.text else 0)
        flag = " (TRUNCATED -- more content likely remains below)" if self.truncated else ""
        return f"{n_lines} lines captured over {self.scrolls_performed} scroll(s){flag}"


class ScreenReader:
    """Assembles the full text of scrollable content via OCR + scroll + stitch."""

    def __init__(self, screenshot_fn, hands, ocr_engine=None, settle_time: float = 0.45):
        self.screenshot_fn = screenshot_fn
        self.hands = hands
        self.ocr = ocr_engine or _ocr.get_default_engine()
        self.settle_time = settle_time

    def read_full(
        self,
        hover_point: Optional[Tuple[int, int]] = None,
        max_scrolls: int = 10,
        scroll_amount: int = 4,
        return_to_top: bool = True,
    ) -> ReadResult:
        """
        Capture the full content, scrolling as needed.

        `hover_point` positions the cursor before scrolling -- required
        because Windows delivers wheel events to whatever the cursor sits
        over, not to whatever "looks" scrollable. Defaults to a point in the
        lower-middle of the screen (below any header/toolbar strip, inside
        the likely content column) when the caller has no better candidate.
        """
        started = time.perf_counter()

        if not self.ocr.available:
            log.warning("OCR unavailable -- read_full cannot capture text.")
            return ReadResult(text="", duration=time.perf_counter() - started)

        shot = self.screenshot_fn()
        w, h = shot.size
        point = hover_point or (int(w * 0.5), int(h * 0.55))

        lines = self._read_lines(shot)
        assembled: List[str] = list(lines)
        prev_lines = lines
        scrolls = 0
        truncated = False

        # Tracks, per DISTINCT frame, which normalised lines appeared in it --
        # the input to the end-of-pass watermark/overlay strip.
        frame_count = 1
        line_frame_counts: dict = {}
        for l in {self._normalise(x) for x in lines}:
            line_frame_counts[l] = 1

        for _ in range(max_scrolls):
            if not self._scroll(point, scroll_amount):
                break
            scrolls += 1

            shot = self.screenshot_fn()
            new_lines = self._read_lines(shot)
            frame_count += 1
            for l in {self._normalise(x) for x in new_lines}:
                line_frame_counts[l] = line_frame_counts.get(l, 0) + 1

            if new_lines == prev_lines:
                log.debug("read_full: identical capture after scroll -- stopping (likely bottom).")
                break

            overlap = self._find_overlap(prev_lines, new_lines)
            new_tail = new_lines[overlap:]

            # Safety net beneath the primary overlap match. OCR line
            # segmentation is not perfectly deterministic across two crops of
            # the same content -- a line straddling a viewport edge can split
            # differently between frames -- so the exact/fuzzy match above
            # occasionally under-counts the overlap at one boundary. Rather
            # than trust that every boundary is clean, drop any "new" line
            # that duplicates something already captured recently; this
            # catches an under-count without needing it to never happen.
            recent = {self._normalise(l) for l in assembled[-40:]}
            new_tail = [l for l in new_tail if self._normalise(l) not in recent]

            if not new_tail:
                log.debug("read_full: scroll revealed no new lines -- stopping.")
                break

            assembled.extend(new_tail)
            prev_lines = new_lines

            if scrolls >= max_scrolls:
                truncated = True

        assembled = self._strip_repeating_overlays(assembled, line_frame_counts, frame_count)

        # Reading leaves the pane scrolled to the bottom. The caller almost
        # always needs to act from the top next -- pick the first option, click
        # into the editor, re-read the problem statement -- so scroll back up to
        # where the read began. Without this the agent is stranded at the bottom
        # with no cheap way back to the start of the question.
        if return_to_top and scrolls > 0:
            self._return_to_top(point, scroll_amount, scrolls)

        return ReadResult(
            text="\n".join(assembled),
            scrolls_performed=scrolls,
            truncated=truncated,
            duration=time.perf_counter() - started,
        )

    def _return_to_top(self, point: Tuple[int, int], amount: int, down_scrolls: int) -> None:
        """Scroll the pane back up to the start after a top-to-bottom read.

        Scrolls up a little more than we scrolled down, and stops early once the
        view stops moving -- that means the top is reached, so any extra wheel-up
        is a harmless no-op. Symmetric with the bottom-detection in the read loop.
        """
        prev: Optional[List[str]] = None
        for _ in range(down_scrolls + 3):
            if not self._scroll(point, amount, direction="up"):
                break
            lines = self._read_lines(self.screenshot_fn())
            if lines == prev:
                break  # nothing moved -- we are at the top
            prev = lines

    def _strip_repeating_overlays(
        self, assembled: List[str], line_frame_counts: dict, frame_count: int,
    ) -> List[str]:
        """
        Collapse lines that showed up in most frames to a single mention.

        A watermark or sticky footer is pinned to the viewport, so it is
        read fresh in nearly every frame regardless of scroll position --
        unlike real content, which each appears in only the handful of frames
        it was actually visible in. Anything clearing the threshold is kept
        once (so its presence is not hidden entirely -- a persistent status
        line can occasionally be genuinely relevant context) but not repeated.
        """
        import math
        if frame_count < 3:
            return assembled  # Too few frames for frequency to mean anything.

        threshold = max(3, math.ceil(0.5 * frame_count))
        overlay_norms = {n for n, c in line_frame_counts.items() if c >= threshold}
        if not overlay_norms:
            return assembled

        log.debug("read_full: collapsing %d likely watermark/overlay line(s) "
                   "seen in >=%d/%d frames.", len(overlay_norms), threshold, frame_count)

        kept: List[str] = []
        seen_overlay: set = set()
        for line in assembled:
            norm = self._normalise(line)
            if norm in overlay_norms:
                if norm in seen_overlay:
                    continue
                seen_overlay.add(norm)
            kept.append(line)
        return kept

    # -- internals ------------------------------------------------------

    @staticmethod
    def _normalise(line: str) -> str:
        """Loose key for the duplicate safety net -- case/space-insensitive."""
        return " ".join(line.lower().split())

    def _read_lines(self, image) -> List[str]:
        ocr_lines = self.ocr.read(image)
        ocr_lines.sort(key=lambda l: (l.rect[1], l.rect[0]))
        return [l.text.strip() for l in ocr_lines if l.text.strip()]

    def _scroll(self, point: Tuple[int, int], amount: int, direction: str = "down") -> bool:
        try:
            self.hands.move_to(*point)
            time.sleep(0.05)
            self.hands.execute_action({
                "type": "scroll", "direction": direction, "amount": amount,
            })
            time.sleep(self.settle_time)
            return True
        except Exception as exc:
            log.warning("read_full: scroll failed (%s); stopping.", exc)
            return False

    @staticmethod
    def _find_overlap(prev_lines: List[str], new_lines: List[str], min_overlap: int = 1) -> int:
        """
        How many leading lines of `new_lines` already appeared at the tail of
        `prev_lines`? Tries an exact match first (OCR is deterministic for
        identical pixels), then falls back to fuzzy character-level matching
        for near-identical captures where anti-aliasing or a 1px scroll
        offset nudged a line's text apart.
        """
        max_k = min(len(prev_lines), len(new_lines))
        for k in range(max_k, min_overlap - 1, -1):
            if prev_lines[-k:] == new_lines[:k]:
                return k

        prev_tail = "\n".join(prev_lines[-8:])
        new_head = "\n".join(new_lines[:8])
        if not prev_tail or not new_head:
            return 0
        sm = difflib.SequenceMatcher(None, prev_tail, new_head)
        match = sm.find_longest_match(0, len(prev_tail), 0, len(new_head))
        if match.size < 20:
            return 0
        covered = new_head[: match.b + match.size]
        return covered.count("\n") + 1
