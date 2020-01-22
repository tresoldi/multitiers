"""
Utility functions for the `multitiers` library.
"""

# Import Python standard libraries
import csv
from pathlib import Path

# Import MPI-SHH libraries
from pyclts import CLTS

# Set the resource directory; this is safe as we already added
# `zip_safe=False` to setup.py
DEFAULT_CLTS = Path(__file__).parent.parent / "clts-master"


def reduce_alignment(alm):
    return [tok for tok in alm if tok not in ["(", ")"]]


# Initialize a clts object, returinign it, from the provided path
# or from default
def clts_object(repos=None):
    if not repos:
        repos = DEFAULT_CLTS.as_posix()

    clts = CLTS(repos)

    return clts


# TODO: receive column names as a dictionary, or kwargs
# TODO: rename to wordlist2data or read_wordlist_data
def wordlist2mt(
    filepath,
    cogid="COGID",
    alignment="ALIGNMENT",
    doculect="DOCULECT",
    comma=False,
):
    """
    Reads a wordlist in lingpy format and returns an equivalent MT object.

    Parameters
    ----------
    filepath : string
        Path to the wordlist.
    cogid : string
        Name of the cognate id column, used for grouping alignments
        (default: `COGID`).
    alignment : string
        Name of the alignment column (default: `ALIGNMENT`).
    comma : bool
        Whether to use commas instead of tabulations as field separator
        (default: False)
    """

    # Get the right field delimiter
    delimiter = "\t,"[comma]

    # Collect information in a single data structure, for later processing
    with open(filepath) as handler:
        reader = csv.DictReader(handler, delimiter=delimiter)
        rows = [row for row in reader]

    # Collect the cogids as an ordered set
    # TODO: needs to be sorted?
    cogids = sorted(set([row[cogid] for row in rows]))

    # Collect alignments per cogid
    # TODO: should we just distribute in slots first, so not to filter
    # everything?
    # TODO: function for getting alignment, allowing cleaning
    # TODO: what if there are more two alignments of the same language? synonyms
    data = {}
    for cid in cogids:
        subset = [
            {
                "doculect": row[doculect],
                "alignment": reduce_alignment(row[alignment].split()),
            }
            for row in rows
            if row[cogid] == cid
        ]

        data[cid] = subset

    return data
