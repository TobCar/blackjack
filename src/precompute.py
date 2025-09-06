#!/usr/bin/env python3
"""
Wrapper script for precomputation CLI that can be run from any directory.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

# Import and run the CLI
from precomputing.cli import cli_entry_point

if __name__ == "__main__":
    cli_entry_point()