"""
The Eyes: dxcam (DXGI Desktop Duplication)

Extremely fast screen capture utilizing Windows Desktop Window Manager.
Zero-copy GPU buffer reading achieves 144+ FPS with 1-2ms latency.
"""
import logging
from PIL import Image

# Imported before dxcam on purpose: the capture must be initialised in a
# DPI-aware process so its frames are in the same pixel space the mouse uses.
from autopc.core.display import ensure_dpi_aware

ensure_dpi_aware()

try:
    import dxcam
except ImportError:
    dxcam = None
    logging.warning("dxcam not installed. Captures will fall back to Pillow.")

log = logging.getLogger(__name__)


class DXCamEyes:
    """
    High-speed vision module using DXGI Desktop Duplication.
    Provides frames to perception and the reader as PIL Images.
    """

    def __init__(self, target_fps: int = 60, region: tuple = None):
        """
        Args:
            target_fps: Frame rate to target (unused for single grab, useful if starting a stream).
            region: Optional (left, top, right, bottom) bounding box to capture.
        """
        if dxcam is None:
            raise ImportError("dxcam is required. Run 'pip install dxcam'")

        self.region = region
        self._camera = dxcam.create(output_color="RGB")
        self._last_frame: Image.Image = None
        log.info(f"DXCam initialized. Capturing region: {region or 'Full Screen'}")

    def grab_frame(self) -> Image.Image:
        """
        Capture the current screen.

        Note on `None` returns: DXGI Desktop Duplication only hands over a
        frame when the desktop has *changed* since the last grab. On a static
        screen -- which is most of the time while the agent is thinking --
        `grab()` legitimately returns None. That is not an error; it means
        "identical to the frame you already have".

        The previous implementation treated None as failure and fell back to
        PIL ImageGrab on every such call. That was both slow (a full GDI blit
        per observation instead of a GPU buffer read) and noisy, logging a
        warning on every static frame.

        So we cache the last good frame and return it when DXGI reports no
        change. This is faster *and* semantically more accurate: the cached
        frame is exactly what is on screen. PIL is used only when we have no
        cached frame at all, i.e. the very first capture.
        """
        if not self._camera:
            from PIL import ImageGrab
            return ImageGrab.grab(bbox=self.region)

        frame = self._camera.grab(region=self.region)

        if frame is None:
            # One short retry covers a genuine timeout during a rapid repaint.
            import time
            time.sleep(0.01)
            frame = self._camera.grab(region=self.region)

        if frame is not None:
            self._last_frame = Image.fromarray(frame)
            return self._last_frame

        if self._last_frame is not None:
            # No new frame => the screen is unchanged => the cache is current.
            return self._last_frame

        log.warning("dxcam returned no frame and no cache exists; using PIL fallback.")
        from PIL import ImageGrab
        self._last_frame = ImageGrab.grab(bbox=self.region)
        return self._last_frame

    def start_stream(self):
        """Start continuous capturing (useful for recording or real-time streaming)."""
        if self._camera:
            self._camera.start(target_fps=60)

    def stop_stream(self):
        """Stop continuous capturing."""
        if self._camera:
            self._camera.stop()
