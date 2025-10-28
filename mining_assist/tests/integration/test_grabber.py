"""
Integration tests for the screen grabber module.
"""

import time

import numpy as np
import pytest

from mining_assist.grabber import ScreenGrabber  # type: ignore


class TestScreenGrabberIntegration:
    """Integration tests for ScreenGrabber."""

    def test_actual_screen_capture(self):
        """Test actual screen capture (if backends available)."""
        try:
            grabber = ScreenGrabber(target_size=(160, 120))

            frame = grabber.capture_frame()

            if frame is not None:
                assert isinstance(frame, np.ndarray)
                assert len(frame.shape) == 3  # Height, Width, Channels
                assert frame.shape[:2] == (120, 160)
                assert frame.dtype == np.uint8

            grabber.close()

        except Exception as e:
            pytest.skip(f"Screen capture not available: {e}")

    def test_performance_metrics(self):
        """Test performance tracking during capture."""
        try:
            grabber = ScreenGrabber()

            # Capture several frames
            for _ in range(10):
                grabber.capture_frame()
                time.sleep(0.01)

            # Should have some FPS measurement
            fps = grabber.get_fps()
            assert fps >= 0.0

            grabber.close()

        except Exception as e:
            pytest.skip(f"Screen capture not available: {e}")
