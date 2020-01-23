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

TEST_DATA = [
    {"ID": "1", "DOCULECT": "la", "COGID": "A", "ALIGNMENT": "p - a"},
    {"ID": "2", "DOCULECT": "lb", "COGID": "A", "ALIGNMENT": "b - e"},
    {"ID": "3", "DOCULECT": "lc", "COGID": "A", "ALIGNMENT": "p j a"},
    {"ID": "4", "DOCULECT": "lc", "COGID": "B", "ALIGNMENT": "a r t"},
    {"ID": "5", "DOCULECT": "lb", "COGID": "B", "ALIGNMENT": "a r t"},
]


class TestMultiTiers(unittest.TestCase):
    def test_multitiers(self):
        mt = multitiers.MultiTiers(TEST_DATA)
