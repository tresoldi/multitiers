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
    {"ID": "4", "DOCULECT": "lb", "COGID": "B", "ALIGNMENT": "a r t"},
    {"ID": "5", "DOCULECT": "lc", "COGID": "B", "ALIGNMENT": "a r t"},
]


class TestMultiTiers(unittest.TestCase):
    def test_multitiers(self):
        mt = multitiers.MultiTiers(TEST_DATA)

    def test_multitiers_colnames(self):
        mt = multitiers.MultiTiers(
            TEST_DATA, doculect="DOCULECT", cogid="COGID", alignment="ALIGNMENT"
        )

    def test_multitiers_shifted(self):
        mt = multitiers.MultiTiers(TEST_DATA, left=2, right=2)

    def test_multitiers_str(self):
        mt = multitiers.MultiTiers(TEST_DATA)
        str(mt)

    def test_multitiers_repr(self):
        mt = multitiers.MultiTiers(TEST_DATA)
        repr(mt)

    def test_multitiers_hash(self):
        mt1 = multitiers.MultiTiers(TEST_DATA)
        mt2 = multitiers.MultiTiers(TEST_DATA)
        mt3 = multitiers.MultiTiers(TEST_DATA[:-1])

        assert hash(mt1) == hash(mt2)
        assert hash(mt1) != hash(mt3)


if __name__ == "__main__":
    # Explicitly creating and running a test suite allows to profile
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiTiers)
    unittest.TextTestRunner(verbosity=2).run(suite)
