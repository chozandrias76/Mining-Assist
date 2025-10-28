#!/usr/bin/env python3
"""Teleoperation recording script for mining assist.

Records frames, keyboard/mouse inputs, and mode labels for training data collection.
This is typically the first script used to gather training data.
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from pynput import keyboard, mouse

from mining_assist.grabber import ScreenGrabber
from mining_assist.utils import (
    MovingAverage,
    PerformanceTimer,
    create_run_directory,
    ensure_dir,
    save_json,
    setup_logging,
)


class InputRecorder:
    """Records keyboard and mouse inputs during teleoperation."""

    def __init__(self):
        """Initialize the input listener."""
        self.current_keys = set()
        self.current_mouse_pos = (0, 0)
        self.current_mouse_buttons = set()
        self.events = []

        # Start listeners
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press, on_release=self._on_key_release
        )
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )

        self.keyboard_listener.start()
        self.mouse_listener.start()

        logging.info("Input recording started")

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            key_name = key.char if hasattr(key, "char") and key.char else str(key)
            self.current_keys.add(key_name)

            event = {"timestamp": time.time(), "type": "key_press", "key": key_name}
            self.events.append(event)

        except Exception as e:
            logging.error(f"Error in key press handler: {e}")

    def _on_key_release(self, key):
        """Handle key release events."""
        try:
            key_name = key.char if hasattr(key, "char") and key.char else str(key)
            self.current_keys.discard(key_name)

            event = {"timestamp": time.time(), "type": "key_release", "key": key_name}
            self.events.append(event)

        except Exception as e:
            logging.error(f"Error in key release handler: {e}")

    def _on_mouse_move(self, x, y):
        """Handle mouse movement events."""
        self.current_mouse_pos = (x, y)

        event = {"timestamp": time.time(), "type": "mouse_move", "x": x, "y": y}
        self.events.append(event)

    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        button_name = str(button)

        if pressed:
            self.current_mouse_buttons.add(button_name)
        else:
            self.current_mouse_buttons.discard(button_name)

        event = {
            "timestamp": time.time(),
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": button_name,
            "pressed": pressed,
        }
        self.events.append(event)

    def _on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll events."""
        event = {
            "timestamp": time.time(),
            "type": "mouse_scroll",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }
        self.events.append(event)

    def get_current_state(self) -> dict:
        """Get current input state."""
        return {
            "keys": list(self.current_keys),
            "mouse_pos": self.current_mouse_pos,
            "mouse_buttons": list(self.current_mouse_buttons),
        }

    def stop(self):
        """Stop input recording."""
        self.keyboard_listener.stop()
        self.mouse_listener.stop()
        logging.info("Input recording stopped")

    def clear_events(self):
        """Clear recorded events."""
        self.events.clear()


class ModeDetector:
    """Simple heuristic-based mode detection for labeling."""

    def __init__(self):
        """Initialize the mode detector with keyword mappings."""
        self.mode_keywords = {
            "MENU": ["menu", "settings", "options", "inventory"],
            "FLIGHT": ["cockpit", "space", "flight", "pilot"],
            "LOADING": ["loading", "connecting", "please wait"],
        }

    def detect_mode(self, frame: np.ndarray, window_title: str = "") -> str:  # noqa: ARG002
        """Detect current game mode using simple heuristics.

        This is a placeholder implementation. In practice, you might:
        - Use template matching for UI elements
        - OCR to read text on screen
        - Color histogram analysis
        - Manual labeling during recording

        Args:
            frame: Current frame
            window_title: Current window title

        Returns:
            Detected mode: 'MENU', 'FLIGHT', or 'LOADING'
        """
        # Simple heuristic based on window title
        window_title_lower = window_title.lower()

        for mode, keywords in self.mode_keywords.items():
            if any(keyword in window_title_lower for keyword in keywords):
                return mode

        # Default fallback - could be improved with actual image analysis
        # For now, assume FLIGHT mode if no specific indicators
        return "FLIGHT"


class TeleopRecorder:
    """Main teleoperation recording class."""

    def __init__(
        self,
        game_window: str,
        output_dir: str,
        fps: float = 10.0,
        frame_size: tuple | None = None,
    ):
        """Initialize the teleoperation recorder.

        Args:
            game_window: Name of the game window to capture
            output_dir: Directory to save recordings
            fps: Recording frame rate
            frame_size: Optional frame size (width, height)
        """
        self.game_window = game_window
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # Create output directory
        ensure_dir(self.output_dir)

        # Initialize components
        target_size = None
        if frame_size and len(frame_size) == 2:
            target_size = (int(frame_size[0]), int(frame_size[1]))
        self.grabber = ScreenGrabber(
            target_window=game_window,
            target_size=target_size,
        )
        self.input_recorder = InputRecorder()
        self.mode_detector = ModeDetector()

        # Recording state
        self.is_recording = False
        self.frame_count = 0
        self.window_focused = True  # Track window focus state
        self.frames_skipped_unfocused = 0  # Count frames skipped due to unfocus
        self.session_data = {
            "metadata": {
                "game_window": game_window,
                "fps": fps,
                "frame_size": frame_size,
                "start_time": None,
                "end_time": None,
            },
            "frames": [],
            "inputs": [],
            "modes": [],
        }

        # Performance tracking
        self.fps_tracker = MovingAverage(window_size=30)

        logging.info(f"Teleop recorder initialized for '{game_window}'")

    def start_recording(self):
        """Start recording session."""
        # Try to bring the target window to foreground before starting
        if self.grabber.target_window:
            logging.info(
                f"Attempting to bring '{self.grabber.target_window}' to foreground..."
            )
            if self.grabber.bring_window_to_foreground():
                # Give the system a moment to bring the window forward
                time.sleep(0.5)
            else:
                logging.warning(
                    "Failed to bring window to foreground, continuing anyway..."
                )

        self.is_recording = True
        self.session_data["metadata"]["start_time"] = time.time()

        # Create subdirectories
        frames_dir = self.output_dir / "frames"
        ensure_dir(frames_dir)

        logging.info("Recording started - Press 'q' to stop")

        last_frame_time = 0

        try:
            while self.is_recording:
                current_time = time.time()

                # Check if target window is focused
                window_focused = self.grabber.is_window_focused()

                # Handle focus state changes
                if window_focused != self.window_focused:
                    if window_focused:
                        logging.info(
                            f"Window '{self.game_window}' gained focus - resuming recording"
                        )
                    else:
                        logging.info(
                            f"Window '{self.game_window}' lost focus - pausing recording"
                        )
                    self.window_focused = window_focused

                # Capture frame at specified FPS (only if window is focused)
                if current_time - last_frame_time >= self.frame_interval:
                    if self.window_focused:
                        with PerformanceTimer() as timer:
                            success = self._capture_frame_with_data()

                        if success:
                            last_frame_time = current_time
                            actual_fps = 1.0 / timer.get_elapsed()
                            avg_fps = self.fps_tracker.update(actual_fps)

                            if self.frame_count % 30 == 0:  # Log every 30 frames
                                logging.info(
                                    f"Frame {self.frame_count}, FPS: {avg_fps:.1f}"
                                )
                    else:
                        # Window not focused - skip this frame but still update timing
                        self.frames_skipped_unfocused += 1
                        last_frame_time = current_time

                        # Log occasionally when paused
                        if (
                            self.frames_skipped_unfocused % 50 == 0
                        ):  # Every ~5 seconds at 10 FPS
                            logging.info(
                                f"Recording paused - {self.frames_skipped_unfocused} frames skipped (window not focused)"
                            )

                # Check for quit condition
                if "Key.q" in self.input_recorder.current_keys:
                    logging.info("Quit key pressed")
                    break

                # Small sleep to prevent 100% CPU usage
                time.sleep(0.001)

        except KeyboardInterrupt:
            logging.info("Recording interrupted by user")

        self.stop_recording()

    def _capture_frame_with_data(self) -> bool:
        """Capture frame and associated data."""
        # Capture frame
        frame = self.grabber.capture_frame()
        if frame is None:
            return False

        # Get current input state
        input_state = self.input_recorder.get_current_state()

        # Detect mode (placeholder implementation)
        detected_mode = self.mode_detector.detect_mode(frame, self.game_window)

        # Save frame
        frame_filename = f"frame_{self.frame_count:06d}.png"
        frame_path = self.output_dir / "frames" / frame_filename
        cv2.imwrite(str(frame_path), frame)

        # Record frame data
        frame_data = {
            "frame_id": self.frame_count,
            "timestamp": time.time(),
            "filename": frame_filename,
            "shape": frame.shape,
            "mode": detected_mode,
        }

        # Store data
        self.session_data["frames"].append(frame_data)
        self.session_data["inputs"].append(
            {
                "frame_id": self.frame_count,
                "timestamp": frame_data["timestamp"],
                **input_state,
            }
        )
        self.session_data["modes"].append(
            {
                "frame_id": self.frame_count,
                "timestamp": frame_data["timestamp"],
                "mode": detected_mode,
            }
        )

        self.frame_count += 1
        return True

    def stop_recording(self):
        """Stop recording and save data."""
        self.is_recording = False
        self.session_data["metadata"]["end_time"] = time.time()

        # Stop input recording
        self.input_recorder.stop()

        # Save session data
        self._save_session_data()

        # Close grabber
        self.grabber.close()

        duration = (
            self.session_data["metadata"]["end_time"]
            - self.session_data["metadata"]["start_time"]
        )

        logging.info(f"Recording stopped. {self.frame_count} frames in {duration:.1f}s")
        if self.frames_skipped_unfocused > 0:
            logging.info(
                f"Skipped {self.frames_skipped_unfocused} frames due to window not being focused"
            )
        logging.info(f"Data saved to: {self.output_dir}")

    def _save_session_data(self):
        """Save session data to files."""
        # Save main session file
        session_file = self.output_dir / "session.json"
        save_json(self.session_data, session_file)

        # Save input events
        events_file = self.output_dir / "input_events.json"
        events_data = {"events": self.input_recorder.events}
        save_json(events_data, events_file)

        # Save summary
        duration = (
            self.session_data["metadata"]["end_time"]
            - self.session_data["metadata"]["start_time"]
        )
        average_fps = self.frame_count / duration if duration > 0 else 0.0

        summary = {
            "total_frames": self.frame_count,
            "frames_skipped_unfocused": self.frames_skipped_unfocused,
            "duration": duration,
            "average_fps": average_fps,
            "modes_detected": list(
                {item["mode"] for item in self.session_data["modes"]}
            ),
            "input_events": len(self.input_recorder.events),
        }

        summary_file = self.output_dir / "summary.json"
        save_json(summary, summary_file)


def main():
    """Main function for teleop recording script."""
    parser = argparse.ArgumentParser(
        description="Record teleoperation data for mining assist"
    )
    parser.add_argument("--game-window", required=True, help="Target game window name")
    parser.add_argument("--fps", type=float, default=10.0, help="Recording FPS")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--frame-size", nargs=2, type=int, help="Frame size (width height)"
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    log_file = Path(args.out) / "recording.log"
    setup_logging(args.log_level, str(log_file))

    # Parse frame size
    frame_size = tuple(args.frame_size) if args.frame_size else None

    # Create unique output directory
    base_dir = Path(args.out)
    output_dir = create_run_directory(base_dir, "teleop_session")

    # Initialize recorder
    recorder = TeleopRecorder(
        game_window=args.game_window,
        output_dir=str(output_dir),
        fps=args.fps,
        frame_size=frame_size,
    )

    print(f"Starting teleoperation recording for '{args.game_window}'")
    print(f"Recording at {args.fps} FPS to: {output_dir}")
    print("Press 'q' to stop recording")

    # Start recording
    recorder.start_recording()


if __name__ == "__main__":
    main()
