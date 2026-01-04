"""
Pytest configuration for ML-Lab test suite.

This file sets up paths and fixtures for all tests.
"""

import sys
from pathlib import Path

# Add src directory to Python path so imports like 'from utils' work
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
