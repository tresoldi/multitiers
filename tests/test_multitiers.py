#!/usr/bin/env python3

"""
test_multititers
================

Tests for the `multitiers` package.
"""

# Import Python libraries
import unittest

# Import the library itself
import multitiers


class TestMultiTiers(unittest.TestCase):
    def test_multitiers(self):
        mt = multitiers.MultiTiers()
        assert mt.test() == 13
