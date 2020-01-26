#!/usr/bin/env python3

"""
__main__.py
"""

# Import Python standard libraries
import argparse

# Import our library
from multitiers import MultiTiers
from multitiers import clts_object, read_wordlist_data


def parse_arguments():
    """
    Parse arguments and return a namespace.
    """

    # TODO: add clts repos
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to the datafile.")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Whether to assume comma-delimited fields in datafile (default: false)",
    )

    args = parser.parse_args()

    return args


def main():
    """
    Main function for multitier operation from the command line.
    """

    # Parse command line arguments
    args = parse_arguments()

    # Read data file and build an MT object from it
    data = read_wordlist_data(args.filename, comma=args.csv)
    #    data = data[:10]
    mt = MultiTiers(data, left=2, right=1, models=["cv"])

    print(str(mt))

    study = [
        # initial position
        {
            "tier_name": "index",
            "includes": [1],
            "excludes": None,
            "unknown": False,
        },
        # All Proto-Germanic /s/
        {
            "tier_name": "Proto-Germanic",
            "includes": ["s"],
            "excludes": None,
            "unknown": False,
        },
        # No German r /s/
        {
            "tier_name": "German",
            "includes": None,
            "excludes": ["r"],
            "unknown": False,
        },
        # Proto-Germanic CV to the left
        {
            "tier_name": "Proto-Germanic_cv_L1",
            "includes": None,
            "excludes": None,
            "unknown": True,
        },
        # Proto-Germanic CV to the right
        {
            "tier_name": "Proto-Germanic_cv_R1",
            "includes": None,
            "excludes": None,
            "unknown": True,
        },
    ]

    data = mt.filter(study)
    print(mt.study(study))


if __name__ == "__main__":
    main()
