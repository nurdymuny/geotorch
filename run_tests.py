#!/usr/bin/env python3
"""Test runner for DavisTensor Phase 1."""

import sys
import pytest

if __name__ == "__main__":
    # Run tests with verbose output
    # Use --noconftest to avoid loading torch-dependent conftest.py
    exit_code = pytest.main([
        "tests/test_davistensor.py",
        "-v",
        "--tb=short",
        "--color=yes",
        "--noconftest"
    ])
    
    sys.exit(exit_code)
