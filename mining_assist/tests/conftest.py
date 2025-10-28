"""Test configuration for mining assist package."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "test_model",
        "input_size": [128, 72],
        "modes": ["MENU", "FLIGHT", "LOADING"],
        "train": {"batch_size": 32, "epochs": 5, "lr": 0.001},
    }
