"""
Prime ChainSentinel — Main Entry Point
Reaper Eagle Technologies | Fabian Sabogal

Run from project root:

    # Smoke test (CPU, fast)
    python main.py --quick

    # Full CPU benchmark
    python main.py --mode full

    # GPU smoke test (verify ROCm before burning credits)
    python main.py --gpu --quick

    # Full MI300X session: real CPU hash baselines + GPU synthetic baseline
    python main.py --gpu --mode full --out exports_mi300x
"""

from __future__ import annotations

import os
import sys

# Allow running from project root without installing as a package.
sys.path.insert(0, os.path.dirname(__file__))

from benchmarks.runner import main


if __name__ == "__main__":
    main()
