"""Utility functions and classes for the mining assist project.

This module provides common utilities for logging, configuration,
file operations, and data processing.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def setup_logging(
    level: str = "INFO", log_file: str | None = None, format_string: str | None = None
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, handlers=[]
    )

    logger = logging.getLogger("mining_assist")
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str | Path) -> DictConfig | ListConfig | None:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration as DictConfig
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str | Path):
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        save_path: Path to save configuration to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path to ensure

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], filepath: str | Path):
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Path to save JSON file to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath) as f:
        return json.load(f)


def preprocess_frame(
    frame: np.ndarray | None,
    target_size: tuple[int, int] | None = None,
    grayscale: bool = False,
    normalize: bool = False,
) -> np.ndarray | None:
    """Preprocess a frame for model input.

    Args:
        frame: Input frame as numpy array
        target_size: Resize to (width, height)
        grayscale: Convert to grayscale
        normalize: Normalize to [0, 1] range

    Returns:
        Preprocessed frame
    """
    if frame is None:
        return None

    processed = frame.copy()

    # Resize if specified
    if target_size:
        processed = cv2.resize(processed, target_size)

    # Convert to grayscale if specified
    if grayscale and len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = np.expand_dims(processed, axis=-1)

    # Normalize if specified
    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def stack_frames(
    frames: list[np.ndarray], max_frames: int = 4, fill_value: float = 0.0
) -> np.ndarray | None:
    """Stack frames for temporal context.

    Args:
        frames: List of frames to stack
        max_frames: Maximum number of frames to stack
        fill_value: Value to use for padding if not enough frames

    Returns:
        Stacked frames as (H, W, channels * max_frames)
    """
    if not frames:
        return None

    # Take the most recent frames
    recent_frames = frames[-max_frames:]

    # Pad with copies of the first frame if not enough frames
    while len(recent_frames) < max_frames:
        if len(recent_frames) > 0:
            # Use the earliest available frame for padding
            recent_frames.insert(0, recent_frames[0])
        else:
            # Create zero frame if no frames available
            h, w = frames[0].shape[:2]
            channels = frames[0].shape[2] if len(frames[0].shape) == 3 else 1
            zero_frame = np.full((h, w, channels), fill_value, dtype=frames[0].dtype)
            recent_frames.append(zero_frame)

    # Stack along channel dimension
    stacked = np.concatenate(recent_frames, axis=-1)
    return stacked


class PerformanceTimer:
    """Simple performance timing utility."""

    def __init__(self, name: str = "Timer"):
        """Initialize the performance timer.

        Args:
            name: Name of the timer for logging purposes
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        """Enter context manager and start timing."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and stop timing."""
        self.stop()

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and calculate elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            raise RuntimeError("Timer not stopped")
        return self.elapsed

    def print_elapsed(self):
        """Print elapsed time."""
        if self.elapsed is None:
            raise RuntimeError("Timer not stopped")
        print(f"{self.name}: {self.elapsed:.4f} seconds")


class MovingAverage:
    """Moving average calculator."""

    def __init__(self, window_size: int = 10):
        """Initialize the moving average calculator.

        Args:
            window_size: Number of values to keep in the window
        """
        self.window_size = window_size
        self.values = []

    def update(self, value: float) -> float:
        """Update with new value and return current average.

        Args:
            value: New value to add

        Returns:
            Current moving average
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        return sum(self.values) / len(self.values)

    def get_average(self) -> float:
        """Get current average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset the moving average."""
        self.values.clear()


class FrameBuffer:
    """Circular buffer for storing frames."""

    def __init__(self, max_size: int = 100):
        """Initialize the frame buffer.

        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.frames = []
        self.timestamps = []
        self.current_index = 0
        self.is_full = False

    def add_frame(self, frame: np.ndarray, timestamp: float | None = None):
        """Add a frame to the buffer.

        Args:
            frame: Frame to add
            timestamp: Optional timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        if len(self.frames) < self.max_size:
            self.frames.append(frame.copy())
            self.timestamps.append(timestamp)
        else:
            self.frames[self.current_index] = frame.copy()
            self.timestamps[self.current_index] = timestamp
            self.is_full = True

        self.current_index = (self.current_index + 1) % self.max_size

    def get_recent_frames(self, count: int) -> list[np.ndarray]:
        """Get the most recent frames.

        Args:
            count: Number of recent frames to get

        Returns:
            List of recent frames (oldest to newest)
        """
        if not self.frames:
            return []

        if not self.is_full:
            # Buffer not full yet, return last `count` frames
            return self.frames[-count:] if count <= len(self.frames) else self.frames[:]

        # Buffer is full, need to handle circular indexing
        result = []
        for i in range(count):
            idx = (self.current_index - count + i) % self.max_size
            if idx >= 0:
                result.append(self.frames[idx])

        return result

    def get_all_frames(self) -> list[np.ndarray]:
        """Get all frames in chronological order."""
        if not self.is_full:
            return self.frames[:]

        # Reorder to get chronological sequence
        return self.frames[self.current_index :] + self.frames[: self.current_index]

    def clear(self):
        """Clear the buffer."""
        self.frames.clear()
        self.timestamps.clear()
        self.current_index = 0
        self.is_full = False


def create_run_directory(base_dir: str | Path, experiment_name: str) -> Path:
    """Create a unique run directory with timestamp.

    Args:
        base_dir: Base directory for runs
        experiment_name: Name of the experiment

    Returns:
        Path to created run directory
    """
    base_dir = Path(base_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_frame_sequence(
    frames: list[np.ndarray],
    output_dir: str | Path,
    prefix: str = "frame",
    format: str = "png",
):
    """Save a sequence of frames to disk.

    Args:
        frames: List of frames to save
        output_dir: Directory to save frames to
        prefix: Filename prefix
        format: Image format (png, jpg, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        filename = f"{prefix}_{i:06d}.{format}"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate up to find the root (where pyproject.toml is)
    for parent in current_file.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current directory
    return Path.cwd()
