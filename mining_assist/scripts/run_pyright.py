#!/usr/bin/env python3
"""Script to run pyright with proper local package installation."""

import subprocess
import sys
from pathlib import Path


def main():
    """Install package in editable mode and run pyright."""
    # Get the project root (where pyproject.toml is)
    project_root = Path(__file__).parent

    # Run pyright using npx to ensure we have the latest version
    try:
        result = subprocess.run(
            ["npx", "pyright@1.1.344"], cwd=project_root, check=False
        )
        return result.returncode
    except FileNotFoundError:
        print("npx not found. Please install Node.js and npm.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
