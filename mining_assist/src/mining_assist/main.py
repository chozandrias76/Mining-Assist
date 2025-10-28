#!/usr/bin/env python3
"""Main entry point for Mining Assist.

This module provides the command-line interface for the Mining Assist project,
allowing users to access different functionality like teleoperation recording.
"""

import argparse
import sys

from mining_assist.scripts.teleop_record import main as teleop_main


def main():
    """Main entry point for Mining Assist CLI."""
    parser = argparse.ArgumentParser(
        description="Mining Assist - AI agent for 3D game automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  teleop-record  Record teleoperation data for training

Examples:
  mining-assist teleop-record --game-window "Game Window" --fps 10 --out ./data
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Teleoperation recording command
    teleop_parser = subparsers.add_parser(
        "teleop-record",
        help="Record teleoperation data for training",
        description="Record frames, keyboard/mouse inputs, and mode labels for training data collection.",
    )
    teleop_parser.add_argument(
        "--game-window", required=True, help="Target game window name"
    )
    teleop_parser.add_argument("--fps", type=float, default=10.0, help="Recording FPS")
    teleop_parser.add_argument("--out", required=True, help="Output directory")
    teleop_parser.add_argument(
        "--frame-size", nargs=2, type=int, help="Frame size (width height)"
    )
    teleop_parser.add_argument("--log-level", default="INFO", help="Logging level")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nError: No command specified. Use --help for more information.")
        sys.exit(1)

    # Execute the appropriate command
    if args.command == "teleop-record":
        # Set up sys.argv to match what teleop_record expects
        sys.argv = [
            "teleop-record",
            "--game-window",
            args.game_window,
            "--fps",
            str(args.fps),
            "--out",
            args.out,
        ]

        if args.frame_size:
            sys.argv.extend(["--frame-size"] + [str(x) for x in args.frame_size])

        if args.log_level:
            sys.argv.extend(["--log-level", args.log_level])

        # Call the teleop_record main function
        teleop_main()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
