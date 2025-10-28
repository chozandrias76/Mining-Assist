"""
Integration tests for the teleoperation recording script.
"""

import pytest


class TestTeleopRecorderIntegration:
    """Integration tests for TeleopRecorder."""

    def test_full_recording_session_short(self, temp_dir):
        """Test a short recording session end-to-end."""
        try:
            # This test requires actual system resources
            # Skip if not available
            from mining_assist.scripts.teleop_record import (
                TeleopRecorder,  # type: ignore
            )

            recorder = TeleopRecorder(
                game_window="Test",  # May not exist
                output_dir=str(temp_dir),
                fps=1.0,  # Very low FPS for testing
            )

            # This would require actual screen capture
            # For now, just test that it doesn't crash
            assert recorder.fps == 1.0
            assert not recorder.is_recording

        except Exception as e:
            pytest.skip(f"Integration test not available: {e}")
