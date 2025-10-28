"""
Tests for the utilities module.
"""

import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from mining_assist.utils import (  # type: ignore
    FrameBuffer,
    MovingAverage,
    PerformanceTimer,
    create_run_directory,
    ensure_dir,
    get_project_root,
    load_config,
    load_json,
    preprocess_frame,
    save_config,
    save_frame_sequence,
    save_json,
    stack_frames,
)


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_ensure_dir(self, temp_dir):
        """Test directory creation."""
        test_path = temp_dir / "test" / "nested" / "dir"

        result = ensure_dir(test_path)

        assert result.exists()
        assert result.is_dir()
        assert result == test_path

    def test_ensure_dir_existing(self, temp_dir):
        """Test ensure_dir with existing directory."""
        test_path = temp_dir / "existing"
        test_path.mkdir()

        result = ensure_dir(test_path)

        assert result.exists()
        assert result == test_path

    def test_save_and_load_json(self, temp_dir):
        """Test JSON save and load operations."""
        test_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        json_file = temp_dir / "test.json"

        # Save
        save_json(test_data, json_file)
        assert json_file.exists()

        # Load
        loaded_data = load_json(json_file)
        assert loaded_data == test_data

    def test_load_json_nonexistent(self, temp_dir):
        """Test loading nonexistent JSON file."""
        nonexistent_file = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(nonexistent_file)

    def test_save_and_load_config(self, temp_dir, sample_config):
        """Test config save and load operations."""
        config_file = temp_dir / "test_config.yaml"

        # Convert to DictConfig
        config = DictConfig(sample_config)

        # Save
        save_config(config, config_file)
        assert config_file.exists()

        # Load
        loaded_config = load_config(config_file)
        assert loaded_config is not None
        assert loaded_config.model == sample_config["model"]
        assert loaded_config.train.batch_size == sample_config["train"]["batch_size"]

    def test_load_config_nonexistent(self, temp_dir):
        """Test loading nonexistent config file."""
        nonexistent_file = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(nonexistent_file)

    def test_preprocess_frame_resize(self, sample_frame):
        """Test frame preprocessing with resize."""
        target_size = (160, 120)

        processed = preprocess_frame(sample_frame, target_size=target_size)

        assert processed is not None
        assert processed.shape[:2] == (120, 160)  # Height, Width
        assert processed.dtype == sample_frame.dtype

    def test_preprocess_frame_grayscale(self, sample_frame):
        """Test frame preprocessing with grayscale conversion."""
        processed = preprocess_frame(sample_frame, grayscale=True)

        assert processed is not None
        assert len(processed.shape) == 3
        assert processed.shape[2] == 1  # Single channel

    def test_preprocess_frame_normalize(self, sample_frame):
        """Test frame preprocessing with normalization."""
        processed = preprocess_frame(sample_frame, normalize=True)

        assert processed is not None
        assert processed.dtype == np.float32
        assert np.all(processed >= 0.0)
        assert np.all(processed <= 1.0)

    def test_preprocess_frame_none_input(self):
        """Test frame preprocessing with None input."""
        processed = preprocess_frame(None)
        assert processed is None

    def test_stack_frames_normal(self):
        """Test frame stacking with normal input."""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * i for i in range(4)]

        stacked = stack_frames(frames, max_frames=4)

        assert stacked is not None
        assert stacked.shape == (100, 100, 12)  # 4 frames * 3 channels

    def test_stack_frames_insufficient(self):
        """Test frame stacking with insufficient frames."""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * i for i in range(2)]

        stacked = stack_frames(frames, max_frames=4)

        assert stacked is not None
        assert stacked.shape == (100, 100, 12)  # Should pad with repeated frames

    def test_stack_frames_empty(self):
        """Test frame stacking with empty input."""
        stacked = stack_frames([])
        assert stacked is None

    def test_create_run_directory(self, temp_dir):
        """Test run directory creation."""
        experiment_name = "test_experiment"

        run_dir = create_run_directory(temp_dir, experiment_name)

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert experiment_name in run_dir.name

    def test_save_frame_sequence(self, temp_dir, sample_frame):
        """Test saving frame sequence."""
        frames = [sample_frame for _ in range(3)]

        save_frame_sequence(frames, temp_dir, prefix="test", format="png")

        # Check that files were created
        expected_files = [
            temp_dir / "test_000000.png",
            temp_dir / "test_000001.png",
            temp_dir / "test_000002.png",
        ]

        for file_path in expected_files:
            assert file_path.exists()

    def test_get_project_root(self):
        """Test project root detection."""
        root = get_project_root()

        assert isinstance(root, Path)
        assert root.exists()


class TestPerformanceTimer:
    """Test cases for PerformanceTimer class."""

    def test_timer_basic_usage(self):
        """Test basic timer usage."""
        timer = PerformanceTimer("test")

        timer.start()
        time.sleep(0.01)  # Small delay
        timer.stop()

        elapsed = timer.get_elapsed()
        assert elapsed > 0.0
        assert elapsed < 1.0  # Should be much less than 1 second

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with PerformanceTimer("test") as timer:
            time.sleep(0.01)

        elapsed = timer.get_elapsed()
        assert elapsed > 0.0
        assert elapsed < 1.0

    def test_timer_not_started(self):
        """Test timer error when not started."""
        timer = PerformanceTimer()

        with pytest.raises(RuntimeError):
            timer.stop()

    def test_timer_not_stopped(self):
        """Test timer error when not stopped."""
        timer = PerformanceTimer()
        timer.start()

        with pytest.raises(RuntimeError):
            timer.get_elapsed()


class TestMovingAverage:
    """Test cases for MovingAverage class."""

    def test_moving_average_basic(self):
        """Test basic moving average functionality."""
        ma = MovingAverage(window_size=3)

        # Add some values
        avg1 = ma.update(1.0)
        assert avg1 == 1.0

        avg2 = ma.update(2.0)
        assert avg2 == 1.5

        avg3 = ma.update(3.0)
        assert avg3 == 2.0

        # This should push out the first value
        avg4 = ma.update(4.0)
        assert avg4 == 3.0  # (2 + 3 + 4) / 3

    def test_moving_average_get_current(self):
        """Test getting current average."""
        ma = MovingAverage(window_size=2)

        ma.update(1.0)
        ma.update(3.0)

        assert ma.get_average() == 2.0

    def test_moving_average_empty(self):
        """Test moving average when empty."""
        ma = MovingAverage()

        assert ma.get_average() == 0.0

    def test_moving_average_reset(self):
        """Test resetting moving average."""
        ma = MovingAverage()

        ma.update(1.0)
        ma.update(2.0)

        ma.reset()
        assert ma.get_average() == 0.0


class TestFrameBuffer:
    """Test cases for FrameBuffer class."""

    def test_frame_buffer_basic(self):
        """Test basic frame buffer functionality."""
        buffer = FrameBuffer(max_size=3)

        frame1 = np.ones((10, 10, 3))
        frame2 = np.ones((10, 10, 3)) * 2
        frame3 = np.ones((10, 10, 3)) * 3

        buffer.add_frame(frame1)
        buffer.add_frame(frame2)
        buffer.add_frame(frame3)

        recent = buffer.get_recent_frames(2)
        assert len(recent) == 2
        assert np.array_equal(recent[0], frame2)
        assert np.array_equal(recent[1], frame3)

    def test_frame_buffer_overflow(self):
        """Test frame buffer with overflow."""
        buffer = FrameBuffer(max_size=2)

        frame1 = np.ones((5, 5)) * 1
        frame2 = np.ones((5, 5)) * 2
        frame3 = np.ones((5, 5)) * 3

        buffer.add_frame(frame1)
        buffer.add_frame(frame2)
        buffer.add_frame(frame3)  # Should overflow

        all_frames = buffer.get_all_frames()
        assert len(all_frames) == 2
        # Should contain frame2 and frame3, frame1 should be dropped
        assert np.array_equal(all_frames[0], frame2)
        assert np.array_equal(all_frames[1], frame3)

    def test_frame_buffer_recent_more_than_available(self):
        """Test getting more recent frames than available."""
        buffer = FrameBuffer(max_size=5)

        frame1 = np.ones((5, 5))
        frame2 = np.ones((5, 5)) * 2

        buffer.add_frame(frame1)
        buffer.add_frame(frame2)

        recent = buffer.get_recent_frames(5)  # More than available
        assert len(recent) == 2

    def test_frame_buffer_empty(self):
        """Test frame buffer when empty."""
        buffer = FrameBuffer()

        recent = buffer.get_recent_frames(3)
        assert len(recent) == 0

        all_frames = buffer.get_all_frames()
        assert len(all_frames) == 0

    def test_frame_buffer_clear(self):
        """Test clearing frame buffer."""
        buffer = FrameBuffer()

        frame = np.ones((5, 5))
        buffer.add_frame(frame)

        buffer.clear()

        assert len(buffer.get_all_frames()) == 0
        assert not buffer.is_full
        assert buffer.current_index == 0

    def test_frame_buffer_timestamps(self):
        """Test frame buffer with timestamps."""
        buffer = FrameBuffer(max_size=3)

        frame = np.ones((5, 5))
        test_time = time.time()

        buffer.add_frame(frame, timestamp=test_time)

        assert len(buffer.timestamps) == 1
        assert buffer.timestamps[0] == test_time
