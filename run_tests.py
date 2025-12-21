#!/usr/bin/env python3
"""Test runner script for DavisTensor."""

import sys
import os

# Add the repository root to the Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

if __name__ == '__main__':
    import pytest
    
    # Run DavisTensor tests
    args = ['tests/test_davistensor.py', '-v', '--tb=short']
    
    # Add any command-line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    sys.exit(pytest.main(args))
