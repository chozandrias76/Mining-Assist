"""
Tests for the teleoperation recording script.
"""

import time
from unittest.mock import Mock, patch

from mining_assist.scripts.teleop_record import (  # type: ignore
    InputRecorder,
    ModeDetector,
    TeleopRecorder,
)


class TestInputRecorder:
    """Test cases for InputRecorder class."""

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_init(self, mock_mouse_listener, mock_keyboard_listener):
        """Test InputRecorder initialization."""
        mock_kb_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard_listener.return_value = mock_kb_instance
        mock_mouse_listener.return_value = mock_mouse_instance

        recorder = InputRecorder()

        assert recorder.current_keys == set()
        assert recorder.current_mouse_pos == (0, 0)
        assert recorder.current_mouse_buttons == set()
        assert recorder.events == []

        # Verify listeners were started
        mock_kb_instance.start.assert_called_once()
        mock_mouse_instance.start.assert_called_once()

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_get_current_state(
        self, mock_mouse_listener, mock_keyboard_listener
    ):
        """Test getting current input state."""
        mock_keyboard_listener.return_value = Mock()
        mock_mouse_listener.return_value = Mock()

        recorder = InputRecorder()

        # Simulate some input state
        recorder.current_keys = {"a", "shift"}
        recorder.current_mouse_pos = (100, 200)
        recorder.current_mouse_buttons = {"left"}

        state = recorder.get_current_state()

        assert set(state["keys"]) == {"a", "shift"}
        assert state["mouse_pos"] == (100, 200)
        assert set(state["mouse_buttons"]) == {"left"}

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_key_press(
        self, mock_mouse_listener, mock_keyboard_listener
    ):
        """Test key press handling."""
        mock_keyboard_listener.return_value = Mock()
        mock_mouse_listener.return_value = Mock()

        recorder = InputRecorder()

        # Create a mock key
        mock_key = Mock()
        mock_key.char = "a"

        # Simulate key press
        recorder._on_key_press(mock_key)

        assert "a" in recorder.current_keys
        assert len(recorder.events) == 1
        assert recorder.events[0]["type"] == "key_press"
        assert recorder.events[0]["key"] == "a"

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_mouse_move(
        self, mock_mouse_listener, mock_keyboard_listener
    ):
        """Test mouse movement handling."""
        mock_keyboard_listener.return_value = Mock()
        mock_mouse_listener.return_value = Mock()

        recorder = InputRecorder()

        # Simulate mouse movement
        recorder._on_mouse_move(150, 250)

        assert recorder.current_mouse_pos == (150, 250)
        assert len(recorder.events) == 1
        assert recorder.events[0]["type"] == "mouse_move"
        assert recorder.events[0]["x"] == 150
        assert recorder.events[0]["y"] == 250

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_stop(self, mock_mouse_listener, mock_keyboard_listener):
        """Test stopping input recording."""
        mock_kb_instance = Mock()
        mock_mouse_instance = Mock()
        mock_keyboard_listener.return_value = mock_kb_instance
        mock_mouse_listener.return_value = mock_mouse_instance

        recorder = InputRecorder()
        recorder.stop()

        # Verify listeners were stopped
        mock_kb_instance.stop.assert_called_once()
        mock_mouse_instance.stop.assert_called_once()

    @patch("mining_assist.scripts.teleop_record.keyboard.Listener")
    @patch("mining_assist.scripts.teleop_record.mouse.Listener")
    def test_input_recorder_clear_events(
        self, mock_mouse_listener, mock_keyboard_listener
    ):
        """Test clearing recorded events."""
        mock_keyboard_listener.return_value = Mock()
        mock_mouse_listener.return_value = Mock()

        recorder = InputRecorder()

        # Add some events
        recorder.events = ["event1", "event2", "event3"]

        recorder.clear_events()

        assert recorder.events == []


class TestModeDetector:
    """Test cases for ModeDetector class."""

    def test_mode_detector_init(self):
        """Test ModeDetector initialization."""
        detector = ModeDetector()

        assert "MENU" in detector.mode_keywords
        assert "FLIGHT" in detector.mode_keywords
        assert "LOADING" in detector.mode_keywords

    def test_mode_detector_menu_detection(self, sample_frame):
        """Test menu mode detection."""
        detector = ModeDetector()

        mode = detector.detect_mode(sample_frame, "Game Menu - Settings")

        assert mode == "MENU"

    def test_mode_detector_flight_detection(self, sample_frame):
        """Test flight mode detection."""
        detector = ModeDetector()

        mode = detector.detect_mode(sample_frame, "Star Citizen - Cockpit View")

        assert mode == "FLIGHT"

    def test_mode_detector_loading_detection(self, sample_frame):
        """Test loading mode detection."""
        detector = ModeDetector()

        mode = detector.detect_mode(sample_frame, "Game Loading...")

        assert mode == "LOADING"

    def test_mode_detector_default_fallback(self, sample_frame):
        """Test default mode fallback."""
        detector = ModeDetector()

        mode = detector.detect_mode(sample_frame, "Unknown Window")

        assert mode == "FLIGHT"  # Default fallback


class TestTeleopRecorder:
    """Test cases for TeleopRecorder class."""

    @patch("mining_assist.scripts.teleop_record.ScreenGrabber")
    @patch("mining_assist.scripts.teleop_record.InputRecorder")
    @patch("mining_assist.scripts.teleop_record.ModeDetector")
    def test_teleop_recorder_init(
        self, mock_mode_detector, mock_input_recorder, mock_screen_grabber, temp_dir
    ):
        """Test TeleopRecorder initialization."""
        mock_grabber = Mock()
        mock_recorder = Mock()
        mock_detector = Mock()

        mock_screen_grabber.return_value = mock_grabber
        mock_input_recorder.return_value = mock_recorder
        mock_mode_detector.return_value = mock_detector

        teleop = TeleopRecorder(
            game_window="Test Window", output_dir=str(temp_dir), fps=10.0
        )

        assert teleop.game_window == "Test Window"
        assert teleop.output_dir == temp_dir
        assert teleop.fps == 10.0
        assert teleop.frame_interval == 0.1
        assert not teleop.is_recording
        assert teleop.frame_count == 0

    @patch("mining_assist.scripts.teleop_record.ScreenGrabber")
    @patch("mining_assist.scripts.teleop_record.InputRecorder")
    @patch("mining_assist.scripts.teleop_record.ModeDetector")
    @patch("mining_assist.scripts.teleop_record.cv2.imwrite")
    def test_teleop_recorder_capture_frame(
        self,
        mock_imwrite,
        mock_mode_detector,
        mock_input_recorder,
        mock_screen_grabber,
        temp_dir,
        sample_frame,
    ):
        """Test frame capture with data."""
        # Setup mocks
        mock_grabber = Mock()
        mock_recorder = Mock()
        mock_detector = Mock()

        mock_screen_grabber.return_value = mock_grabber
        mock_input_recorder.return_value = mock_recorder
        mock_mode_detector.return_value = mock_detector

        mock_grabber.capture_frame.return_value = sample_frame
        mock_recorder.get_current_state.return_value = {
            "keys": ["w", "a"],
            "mouse_pos": (100, 200),
            "mouse_buttons": [],
        }
        mock_detector.detect_mode.return_value = "FLIGHT"
        mock_imwrite.return_value = True

        teleop = TeleopRecorder(
            game_window="Test Window", output_dir=str(temp_dir), fps=10.0
        )

        # Capture frame
        success = teleop._capture_frame_with_data()

        assert success
        assert teleop.frame_count == 1
        assert len(teleop.session_data["frames"]) == 1
        assert len(teleop.session_data["inputs"]) == 1
        assert len(teleop.session_data["modes"]) == 1

        # Verify frame data
        frame_data = teleop.session_data["frames"][0]
        assert frame_data["frame_id"] == 0
        assert frame_data["mode"] == "FLIGHT"
        assert frame_data["shape"] == sample_frame.shape

    @patch("mining_assist.scripts.teleop_record.ScreenGrabber")
    @patch("mining_assist.scripts.teleop_record.InputRecorder")
    @patch("mining_assist.scripts.teleop_record.ModeDetector")
    def test_teleop_recorder_capture_frame_failure(
        self, mock_mode_detector, mock_input_recorder, mock_screen_grabber, temp_dir
    ):
        """Test frame capture failure handling."""
        # Setup mocks
        mock_grabber = Mock()
        mock_recorder = Mock()
        mock_detector = Mock()

        mock_screen_grabber.return_value = mock_grabber
        mock_input_recorder.return_value = mock_recorder
        mock_mode_detector.return_value = mock_detector

        # Simulate capture failure
        mock_grabber.capture_frame.return_value = None

        teleop = TeleopRecorder(
            game_window="Test Window", output_dir=str(temp_dir), fps=10.0
        )

        # Attempt to capture frame
        success = teleop._capture_frame_with_data()

        assert not success
        assert teleop.frame_count == 0
        assert len(teleop.session_data["frames"]) == 0

    @patch("mining_assist.scripts.teleop_record.ScreenGrabber")
    @patch("mining_assist.scripts.teleop_record.InputRecorder")
    @patch("mining_assist.scripts.teleop_record.ModeDetector")
    def test_teleop_recorder_stop_recording(
        self, mock_mode_detector, mock_input_recorder, mock_screen_grabber, temp_dir
    ):
        """Test stopping recording and saving data."""
        # Setup mocks
        mock_grabber = Mock()
        mock_recorder = Mock()
        mock_detector = Mock()

        mock_screen_grabber.return_value = mock_grabber
        mock_input_recorder.return_value = mock_recorder
        mock_mode_detector.return_value = mock_detector

        mock_recorder.events = []

        teleop = TeleopRecorder(
            game_window="Test Window", output_dir=str(temp_dir), fps=10.0
        )

        # Simulate recording state
        teleop.is_recording = True
        teleop.session_data["metadata"]["start_time"] = time.time()

        # Stop recording
        teleop.stop_recording()

        assert not teleop.is_recording
        assert teleop.session_data["metadata"]["end_time"] is not None

        # Verify cleanup calls
        mock_recorder.stop.assert_called_once()
        mock_grabber.close.assert_called_once()

        # Verify files are created
        session_file = temp_dir / "session.json"
        events_file = temp_dir / "input_events.json"
        summary_file = temp_dir / "summary.json"

        assert session_file.exists()
        assert events_file.exists()
        assert summary_file.exists()
