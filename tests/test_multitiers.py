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
        mt.add_tier("test", [0,1,2,3])

        with self.assertRaises(ValueError):
            mt.add_tier("test", [0,1,2,3])

        with self.assertRaises(ValueError):
            mt.add_tier("test_a", [0,1,2])
