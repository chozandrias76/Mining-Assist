"""Mining Assist - AI Agent for 3D Game Automation.

A Python package for training and running AI agents that can play 3D games
using computer vision and reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "Colin Swenson-Healey"
__email__ = "chozandrias76@gmail.com"

from .grabber import ScreenGrabber
from .utils import (
    FrameBuffer,
    MovingAverage,
    PerformanceTimer,
    load_config,
    preprocess_frame,
    save_config,
    setup_logging,
    stack_frames,
)

__all__ = [
    "ScreenGrabber",
    "setup_logging",
    "load_config",
    "save_config",
    "preprocess_frame",
    "stack_frames",
    "PerformanceTimer",
    "MovingAverage",
    "FrameBuffer",
]
