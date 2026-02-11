#!/usr/bin/env python3
"""Convenience script to run full pipeline from project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import main

if __name__ == "__main__":
    main()
