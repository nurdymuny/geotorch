#!/usr/bin/env python3
"""Test runner script for DavisTensor."""

import sys
import pytest

if __name__ == '__main__':
    # Run DavisTensor tests
    args = ['tests/test_davistensor.py', '-v', '--tb=short']
    
    # Add any command-line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    sys.exit(pytest.main(args))
