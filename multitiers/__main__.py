#!/usr/bin/env python3

"""
__main__.py
"""

# Import Python standard libraries
import argparse

# Import our library
from multitiers import MultiTiers
from multitiers import clts_object, wordlist2mt


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
    data = wordlist2mt(args.filename, comma=args.csv)
    clts = clts_object()
    mt = MultiTiers(data, clts)

    print(str(mt))
    print(len(mt.tiers["index"]))


if __name__ == "__main__":
    main()
