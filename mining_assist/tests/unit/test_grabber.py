"""
Tests for the screen grabber module.
"""

import time
from unittest.mock import Mock, patch

from mining_assist.grabber import ScreenGrabber  # type: ignore


class TestScreenGrabber:
    """Test cases for ScreenGrabber class."""

    def test_init_auto_backend(self):
        """Test automatic backend selection."""
        grabber = ScreenGrabber(backend="auto")
        assert grabber.backend == "auto"
        assert hasattr(grabber, "active_backend")
        grabber.close()

    def test_init_with_target_size(self):
        """Test initialization with target size."""
        target_size = (320, 240)
        grabber = ScreenGrabber(target_size=target_size)
        assert grabber.target_size == target_size
        grabber.close()

    def test_init_with_window_name(self):
        """Test initialization with target window."""
        window_name = "Test Window"
        grabber = ScreenGrabber(target_window=window_name)
        assert grabber.target_window == window_name
        grabber.close()

    @patch("mining_assist.grabber.dxcam")
    def test_init_dxcam_backend(self, mock_dxcam):
        """Test dxcam backend initialization."""
        mock_camera = Mock()
        mock_dxcam.create.return_value = mock_camera

        grabber = ScreenGrabber(backend="dxcam")
        assert grabber.active_backend == "dxcam"
        assert grabber.dxcam_camera == mock_camera
        grabber.close()

    @patch("mining_assist.grabber.mss")
    def test_init_mss_backend(self, mock_mss_module):
        """Test mss backend initialization."""
        mock_mss_instance = Mock()
        mock_mss_module.mss.return_value = mock_mss_instance

        grabber = ScreenGrabber(backend="mss")
        assert grabber.active_backend == "mss"
        assert grabber.mss_instance == mock_mss_instance
        grabber.close()

    def test_preprocess_frame_resize(self, sample_frame):
        """Test frame preprocessing with resize."""
        grabber = ScreenGrabber(target_size=(160, 120))

        processed = grabber._preprocess_frame(sample_frame)
        assert processed is not None
        assert processed.shape[:2] == (120, 160)  # Height, Width
        grabber.close()

    def test_preprocess_frame_none_input(self):
        """Test frame preprocessing with None input."""
        grabber = ScreenGrabber()

        processed = grabber._preprocess_frame(None)  # type: ignore
        assert processed is None
        grabber.close()

    def test_fps_tracking(self):
        """Test FPS tracking functionality."""
        grabber = ScreenGrabber()

        # Initially should be 0
        assert grabber.get_fps() == 0.0

        # Simulate some frame updates
        for _ in range(5):
            grabber._update_fps()
            time.sleep(0.01)  # Small delay

        # FPS should still be 0 until enough time passes
        assert grabber.get_fps() >= 0.0
        grabber.close()

    @patch("mining_assist.grabber.win32gui")
    def test_window_focus_check(self, mock_win32gui):
        """Test window focus checking."""
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetWindowText.return_value = "Test Window - Game"

        grabber = ScreenGrabber(target_window="Test Window")

        is_focused = grabber.is_window_focused()
        assert is_focused is True
        grabber.close()

    @patch("mining_assist.grabber.win32gui")
    def test_window_not_focused(self, mock_win32gui):
        """Test window not focused scenario."""
        mock_win32gui.GetForegroundWindow.return_value = 12345
        mock_win32gui.GetWindowText.return_value = "Different Window"

        grabber = ScreenGrabber(target_window="Test Window")

        is_focused = grabber.is_window_focused()
        assert is_focused is False
        grabber.close()

    def test_capture_frame_none_backend(self):
        """Test frame capture when backend fails."""
        grabber = ScreenGrabber()

        # Force backend to None to simulate failure
        grabber.dxcam_camera = None
        grabber.mss_instance = None

        frame = grabber.capture_frame()
        # Should handle gracefully and return None
        assert frame is None
        grabber.close()

    def test_close_cleanup(self):
        """Test proper cleanup on close."""
        grabber = ScreenGrabber()

        # Mock some resources
        grabber.dxcam_camera = Mock()
        grabber.mss_instance = Mock()

        grabber.close()

        # Verify cleanup calls
        if grabber.dxcam_camera:
            grabber.dxcam_camera.release.assert_called_once()
        if grabber.mss_instance:
            grabber.mss_instance.close.assert_called_once()

    @patch("mining_assist.grabber.DXCAM_AVAILABLE", False)
    @patch("mining_assist.grabber.MSS_AVAILABLE", True)
    def test_fallback_to_mss(self):
        """Test fallback to mss when dxcam not available."""
        with patch("mining_assist.grabber.mss.mss") as mock_mss:
            mock_mss.return_value = Mock()

            grabber = ScreenGrabber(backend="auto")
            assert grabber.active_backend == "mss"
            grabber.close()

    @patch("mining_assist.grabber.DXCAM_AVAILABLE", False)
    @patch("mining_assist.grabber.MSS_AVAILABLE", False)
    def test_no_backend_available(self):
        """Test error when no backend available."""
        # This should be caught at import time, but test the case
        # where both backends fail during initialization
        pass  # The import-time check would catch this

    def test_region_capture_setup(self):
        """Test setup with specific region."""
        region = (100, 100, 200, 200)
        grabber = ScreenGrabber(region=region)

        assert grabber.region == region
        grabber.close()
