"""
Threaded video capture with buffering, reconnection, and FPS measurement.
Supports webcam (Windows CAP_DSHOW), USB, file, RTSP.
"""

import cv2
import time
import logging
import threading
from typing import Optional, Tuple
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class VideoStream:
    """Threaded camera / video source reader."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 2,
        use_dshow: bool = True,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 2.0,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.buffer_size = buffer_size
        self.use_dshow = use_dshow
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread: Optional[threading.Thread] = None

        # FPS measurement
        self._frame_count = 0
        self._start_time = time.time()
        self.fps = 0.0

        self._open()

    def _open(self) -> bool:
        """Open the capture source."""
        if self.cap is not None:
            self.cap.release()

        # Windows webcam prefers CAP_DSHOW
        if isinstance(self.source, int) and self.use_dshow:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        # Set resolution (may be ignored by some cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce latency

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: source={self.source}, resolution={actual_w}x{actual_h}")
        return True

    def start(self) -> "VideoStream":
        """Start the background capture thread."""
        if self.thread is not None and self.thread.is_alive():
            return self

        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("VideoStream thread started")
        return self

    def _update(self):
        """Background thread: continuously read frames."""
        consecutive_failures = 0

        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                consecutive_failures += 1
                if consecutive_failures > self.reconnect_attempts:
                    logger.error("Max reconnection attempts reached. Stopping stream.")
                    self.stopped = True
                    break
                logger.warning(f"Camera disconnected. Reconnecting... ({consecutive_failures})")
                time.sleep(self.reconnect_delay)
                self._open()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning("Failed to read frame")
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Drop old frames if queue is full (keep only latest)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass

            self.frame_queue.put(frame)

            # Update FPS
            self._frame_count += 1
            elapsed = time.time() - self._start_time
            if elapsed >= 1.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._start_time = time.time()

    def read(self) -> Tuple[bool, Optional[any]]:
        """
        Get the latest frame.
        Returns (success, frame)
        """
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except Empty:
            return False, None

    def get_fps(self) -> float:
        return round(self.fps, 1)

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened() and not self.stopped

    def stop(self):
        """Graceful shutdown."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logger.info("VideoStream stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
