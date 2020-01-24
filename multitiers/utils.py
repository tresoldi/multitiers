"""
Utility functions for the `multitiers` library.
"""

# Import Python standard libraries
import csv
import re
from pathlib import Path

# Import MPI-SHH libraries
from pyclts import CLTS

# Set the resource directory; this is safe as we already added
# `zip_safe=False` to setup.py.
DEFAULT_CLTS = Path(__file__).parent.parent / "clts-master"


def reduce_alignment(alm):
    return [tok for tok in alm if tok not in ["(", ")"]]


# TODO: decide on gap
def shift_tier(tier, vector, left_orders, right_orders):
    _GAP = None

    # Compute the requested left and right shifts, if any. Lengths
    # of zero are not allowed as, given Python's slicing, they
    # would return totally unexpected vectors. The [:len(x)] and
    # [-len(x):] slices are added so that we have exactly the
    # number of tokens in the shifted tier even in cases where
    # the left or right order are larger than the length of the
    # alignment (e.g., an alignment with 3 tokens and left order
    # of 5 would yield '0 0 0 0 0' and not '0 0 0')
    # for both left and right shifting, to make it easier we first strip
    # the eventual morpheme marks, adding the back later.
    new_tiers = {}
    for left_order in left_orders:
        shifted_vector = [_GAP] * left_order + vector[:-left_order]
        shifted_name = "%s_L%i" % (tier, left_order)

        new_tiers[shifted_name] = shifted_vector

    for right_order in right_orders:
        shifted_vector = vector[right_order:] + [_GAP] * right_order
        shifted_name = "%s_R%i" % (tier, right_order)

        new_tiers[shifted_name] = shifted_vector

    return new_tiers


# TODO: decide how to deal with Nones
# TODO: check status of https://github.com/cldf-clts/pyclts/issues/7
def sc_mapper(vector, mapper):
    sc_vector = [token if token else "-" for token in vector]
    sc_vector = mapper(" ".join(sc_vector))

    return sc_vector


# Initialize a clts object, returinign it, from the provided path
# or from default
def clts_object(repos=None):
    if not repos:
        repos = DEFAULT_CLTS.as_posix()

    clts = CLTS(repos)

    return clts


def get_orders(value):

    # Dictionary used for mapping string descriptions of window size to
    # actual Python ranges; by mapping to `range()` here in advance
    # (and consuming such range into a list), computations is a bit
    # faster and, in particular, it is clearer. Note that we always start
    # from 1, so no zero-length is included in the lists (the zero distance
    # is the actual alignment site itself).
    _ORDER_MAP = {
        "bigram": list(range(1, 2)),
        "trigram": list(range(1, 3)),
        "fourgram": list(range(1, 4)),
    }

    # get mapping
    if isinstance(value, int):
        orders = range(1, value + 1)
    elif isinstance(value, str):
        orders = _ORDER_MAP[value]
    else:
        orders = []

    return orders


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

    return rows

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
