#!/usr/bin/env python3
"""Run the small CPU suite without pytest's optional interactive readline import."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Some Anaconda/macOS readline builds crash when imported without a terminal.
# Readline is unrelated to the models and is not needed by this batch runner.
sys.modules.setdefault('readline', None)
import pytest

if __name__ == '__main__':
    raise SystemExit(pytest.main(['-q', str(ROOT / 'tests'), *sys.argv[1:]]))
