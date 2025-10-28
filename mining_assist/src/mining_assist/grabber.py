"""Screen capture module with fallback support for dxcam and mss.

This module provides a unified interface for capturing game windows using
either dxcam (preferred for performance) or mss (fallback).
"""

import logging
import time

import cv2
import numpy as np

# Try to import win32gui for window focus checking
try:
    import win32gui  # noqa: F401

    WIN32GUI_AVAILABLE = True
except ImportError:
    win32gui = None
    WIN32GUI_AVAILABLE = False
    logging.warning("win32gui not available, window focus checking disabled")

# Try to import dxcam, fall back to mss if not available
try:
    import dxcam

    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    logging.warning("dxcam not available, falling back to mss")

try:
    import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.error("Neither dxcam nor mss available for screen capture")

if not DXCAM_AVAILABLE and not MSS_AVAILABLE:
    raise ImportError("No screen capture backend available. Install dxcam or mss.")


class ScreenGrabber:
    """Screen capture class with automatic fallback between dxcam and mss.

    Provides frame capture, window detection, and preprocessing capabilities.
    """

    def __init__(
        self,
        target_window: str | None = None,
        target_size: tuple[int, int] | None = None,
        backend: str = "auto",
        region: tuple[int, int, int, int] | None = None,
    ):
        """Initialize screen grabber.

        Args:
            target_window: Name of the target window to capture (None for full screen)
            target_size: Resize captured frames to (width, height)
            backend: "dxcam", "mss", or "auto" for automatic selection
            region: Specific region to capture (x, y, width, height)
        """
        self.target_window = target_window
        self.target_size = target_size
        self.region = region
        self.backend = backend

        # Initialize capture objects
        self.dxcam_camera = None
        self.mss_instance = None
        self.window_rect = None
        self.target_hwnd = None  # Handle for the target window

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate capture backend."""
        if self.backend == "dxcam" or (self.backend == "auto" and DXCAM_AVAILABLE):
            try:
                self._init_dxcam()
                self.active_backend = "dxcam"
                logging.info("Using dxcam backend")
            except Exception as e:
                logging.warning(f"Failed to initialize dxcam: {e}")
                if MSS_AVAILABLE:
                    self._init_mss()
                    self.active_backend = "mss"
                    logging.info("Falling back to mss backend")
                else:
                    raise RuntimeError("No working capture backend available") from e
        else:
            if MSS_AVAILABLE:
                self._init_mss()
                self.active_backend = "mss"
                logging.info("Using mss backend")
            else:
                raise RuntimeError("MSS backend not available") from None

    def _init_dxcam(self):
        """Initialize dxcam capture."""
        if not DXCAM_AVAILABLE:
            raise ImportError("dxcam not available")

        self.dxcam_camera = dxcam.create()  # type: ignore
        if self.target_window:
            self._update_window_region()

    def _init_mss(self):
        """Initialize mss capture."""
        if not MSS_AVAILABLE:
            raise ImportError("mss not available")

        self.mss_instance = mss.mss()  # type: ignore
        if self.target_window:
            self._update_window_region()

    def _window_name_matches(self, target: str, actual: str) -> bool:
        """Check if window names match, handling whitespace and case differences.

        Args:
            target: The target window name to search for
            actual: The actual window title found

        Returns:
            True if the names match (case-insensitive, whitespace-tolerant)
        """
        # Normalize both strings: strip whitespace and convert to lowercase
        target_normalized = target.strip().lower()
        actual_normalized = actual.strip().lower()

        # Check for exact match or substring match in either direction
        return (
            target_normalized == actual_normalized
            or target_normalized in actual_normalized
            or actual_normalized in target_normalized
        )

    def _update_window_region(self):
        """Find and update the target window region."""
        if not self.target_window:
            return

        # Use Windows API to find window
        if not WIN32GUI_AVAILABLE:
            logging.warning("Cannot find window without pywin32")
            return

        # Type ignore because win32gui might be None
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):  # type: ignore
                window_text = win32gui.GetWindowText(hwnd)  # type: ignore
                if (
                    window_text
                    and self.target_window
                    and self._window_name_matches(self.target_window, window_text)
                ):
                    windows.append((hwnd, window_text))
            return True

        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)  # type: ignore

        if windows:
            hwnd = windows[0][0]  # Take first match
            rect = win32gui.GetWindowRect(hwnd)  # type: ignore
            self.window_rect = {
                "left": rect[0],
                "top": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            }
            self.target_hwnd = hwnd  # Save the window handle
            logging.info(f"Found window '{windows[0][1]}' at {self.window_rect}")
        else:
            logging.warning(f"Window '{self.target_window}' not found")
            self.window_rect = None

    def capture_frame(self) -> np.ndarray | None:
        """Capture a single frame.

        Returns:
            Frame as numpy array (H, W, C) in BGR format, or None if capture failed
        """
        try:
            if self.active_backend == "dxcam":
                frame = self._capture_dxcam()
            else:
                frame = self._capture_mss()

            if frame is not None:
                frame = self._preprocess_frame(frame)
                self._update_fps()

            return frame

        except Exception as e:
            logging.error(f"Frame capture failed: {e}")
            return None

    def _capture_dxcam(self) -> np.ndarray | None:
        """Capture frame using dxcam."""
        if self.dxcam_camera is None:
            return None

        if self.window_rect:
            region = (
                self.window_rect["left"],
                self.window_rect["top"],
                self.window_rect["left"] + self.window_rect["width"],
                self.window_rect["top"] + self.window_rect["height"],
            )
            frame = self.dxcam_camera.grab(region=region)
        elif self.region:
            frame = self.dxcam_camera.grab(region=self.region)
        else:
            frame = self.dxcam_camera.grab()

        if frame is not None:
            # dxcam returns RGB, convert to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def _capture_mss(self) -> np.ndarray | None:
        """Capture frame using mss."""
        if self.mss_instance is None:
            return None

        if self.window_rect:
            monitor = self.window_rect
        elif self.region:
            monitor = {
                "left": self.region[0],
                "top": self.region[1],
                "width": self.region[2],
                "height": self.region[3],
            }
        else:
            monitor = self.mss_instance.monitors[1]  # Primary monitor

        screenshot = self.mss_instance.grab(monitor)
        frame = np.array(screenshot)

        # mss returns BGRA, convert to BGR
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def _preprocess_frame(self, frame: np.ndarray | None) -> np.ndarray | None:
        """Apply preprocessing to captured frame."""
        if frame is None:
            return None

        # Resize if target size specified
        if self.target_size:
            frame = cv2.resize(frame, self.target_size)

        return frame

    def _update_fps(self):
        """Update FPS tracking."""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps

    def is_window_focused(self) -> bool:
        """Check if target window is currently focused."""
        if not self.target_window:
            return True

        if not WIN32GUI_AVAILABLE:
            logging.warning("Cannot check window focus without pywin32")
            return True

        foreground_hwnd = win32gui.GetForegroundWindow()  # type: ignore
        window_text = win32gui.GetWindowText(foreground_hwnd)  # type: ignore
        return self._window_name_matches(self.target_window, window_text)

    def bring_window_to_foreground(self) -> bool:
        """Bring the target window to the foreground.

        Returns:
            True if successful, False otherwise
        """
        if not self.target_window or not WIN32GUI_AVAILABLE:
            logging.warning(
                "Cannot bring window to foreground without target window or pywin32"
            )
            return False

        if not hasattr(self, "target_hwnd") or not self.target_hwnd:
            logging.warning("No window handle available to bring to foreground")
            return False

        try:
            # Try to bring window to foreground
            win32gui.SetForegroundWindow(self.target_hwnd)  # type: ignore
            win32gui.ShowWindow(self.target_hwnd, 9)  # type: ignore  # SW_RESTORE = 9
            logging.info(f"Brought window '{self.target_window}' to foreground")
            return True
        except Exception as e:
            logging.warning(f"Failed to bring window to foreground: {e}")
            return False

    def close(self):
        """Clean up capture resources."""
        if self.dxcam_camera:
            self.dxcam_camera.release()
            self.dxcam_camera = None

        if self.mss_instance:
            self.mss_instance.close()
            self.mss_instance = None

        logging.info("Screen grabber closed")


def test_grabber():
    """Simple test function for the grabber."""
    grabber = ScreenGrabber(target_size=(640, 480))

    print(f"Active backend: {grabber.active_backend}")

    for i in range(10):
        frame = grabber.capture_frame()
        if frame is not None:
            print(f"Frame {i}: {frame.shape}, FPS: {grabber.get_fps():.1f}")
        else:
            print(f"Frame {i}: Failed to capture")
        time.sleep(0.1)

    grabber.close()


if __name__ == "__main__":
    test_grabber()
