"""
Utility functions for the `multitiers` library.
"""

# Import Python standard libraries
from collections import Counter
import csv
from pathlib import Path
import re

# Import MPI-SHH libraries
from pyclts import CLTS

# TODO: this is a simple&stupid function written quickly to provide a
# small language for demonstration, write a proper parser
def parse_correspondence_study(text):
    # initial dictionaries
    known = {}
    unknown = {}

    # Split in lines and replace multiple sapces
    lines = [re.sub("\s+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Parse each line
    for line in lines:
        # include and exclude for this rule
        restrict = {"include": [], "exclude": []}

        # split constituents
        fields = line.split()
        if len(fields) == 4:
            if fields[2] == "INCLUDE":
                restrict["include"] = fields[3].split(",")
            elif fields[2] == "EXCLUDE":
                restrict["exclude"] = fields[3].split(",")

        # For indexes, convert to integers
        if "index" in fields[1]:
            restrict["include"] = [int(v) for v in restrict["include"]]
            restrict["exclude"] = [int(v) for v in restrict["exclude"]]

        # Remove items if empty
        if not restrict["include"]:
            restrict.pop("include")
        if not restrict["exclude"]:
            restrict.pop("exclude")

        # add to the right dictionary
        if fields[0] == "KNOWN":
            known[fields[1]] = restrict
        elif fields[0] == "UNKNOWN":
            unknown[fields[1]] = restrict

    return known, unknown


# TODO: allow to strip
def parse_alignment(alignment):
    """
    Parses an alignment string.

    Alignment strings are composed of space separated graphemes, with
    optional parentheses for suppressed or non-mandatory material. The
    material between parentheses is kept.

    Parameters
    ----------
    alignment : str
        The alignment string to be parsed.

    Returns
    -------
    seq : list
        A list of tokens.
    """

    return [
        token for token in alignment.strip().split() if token not in ["(", ")"]
    ]


# TODO: allow to pass strings/lists/integers for orders
# TODO: rename tier_name to base_name
def shift_tier(vector, tier_name, left_orders, right_orders, oob="∅"):
    """
    Compute shifted versions of vectors.

    Parameters
    ----------

    vector : list
        Vector to be shifted.
    tier_name : str
        Base name of the tier.
    left_orders: list of int
        List of left contexts to include in the shifts.
    right_orders: list of int
        List of right contexts to include in the shifts.
    oob : str
        Value to use for out-of-bounds tokens (default: "∅").
    """

    # Compute the requested left and right shifts, if any. Lengths
    # of zero are not allowed because, given Python's slicing, they
    # would return totally unexpected vectors. The [:len(x)] and
    # [-len(x):] slices are added so that we have exactly the
    # number of tokens in the shifted tier, even in cases where
    # the left or right order are larger than the length of the
    # alignment (e.g., an alignment with 3 tokens and left order
    # of 5 would yield '0 0 0 0 0' and not '0 0 0').
    # For both left and right shifting, to make it easier we first strip
    # the eventual morpheme marks, adding them back later.
    new_tiers = {}
    for left_order in left_orders:
        shifted_vector = [oob] * left_order + vector[:-left_order]
        shifted_name = f"{tier_name}_L{left_order}"

        new_tiers[shifted_name] = shifted_vector

    for right_order in right_orders:
        shifted_vector = vector[right_order:] + [oob] * right_order
        shifted_name = f"{tier_name}_R{right_order}"

        new_tiers[shifted_name] = shifted_vector

    return new_tiers


# TODO: check status of https://github.com/cldf-clts/pyclts/issues/7
# TODO: decide on gap token (should we keep `-`?)
# TODO: perform with numpy?
# TODO: have an unified mapper signature? not bound to pyclts?
# TODO: rename `alignment` to `vector` whereever appropriate
def sc_mapper(alignment, mapper, oob="∅", gap="-"):
    """
    Maps an alignment vector to sound classes.

    Parameters
    ----------
    alignment : list
        The grapheme alignment to map.
    mapper : xxx
    oob : str
        Out-of-bounds grapheme (default `∅`).
    gap : str
        Gap grapheme (default `-`).

    Returns
    -------
    sc_alignment : list
        A vector with the sound classes corresponding to the provided
        alignment.
    """

    # Prepare the vector in the format expected by CLTS mapper
    # NOTE: tokens are mapped to "-" if they are None (`if token`)
    #       or if they are out-of-bonds/gaps (`in [oob, gap]`)
    sc_vector_str = " ".join(
        [
            token if token and token not in [oob, gap] else "-"
            for token in alignment
        ]
    )
    sc_vector = mapper(sc_vector_str)

    return sc_vector


def clts_object(repos=None):
    """
    Initialize and return a CLTS object.

    Parameters
    ----------
    repos : str
        Path to the root of the CLTS data repository (defaults to the
        copy distributed with the library).

    Returns
    -------
    clts : obj
        An initialized CLTS object.
    """

    if not repos:
        repos = Path(__file__).parent.parent / "clts-master"
        repos = repos.as_posix()

    clts = CLTS(repos)

    return clts


def get_orders(value):
    """
    Return a list of orders for context tiers.

    Parameters
    ----------
    value : int or string
        The maximum context length or a string in the set "bigram" (for
        context 1, and 2), "trigram" (for context 1, 2, and 3), or
        "fourgram" (for contexts 1, 2, 3, and 4).
    """

    # Dictionary used for mapping string descriptions of window size to
    # actual Python ranges; by mapping to `range()` here in advance
    # (and consuming such range into a list), computations is a bit
    # faster and, in particular, it is clearer. Note that we always start
    # from 1, so no zero-length is included in the lists (the zero distance
    # is the actual alignment site itself).
    order_map = {
        "bigram": list(range(1, 2)),
        "trigram": list(range(1, 3)),
        "fourgram": list(range(1, 4)),
    }

    # get mapping
    if isinstance(value, int):
        orders = list(range(1, value + 1))
    elif isinstance(value, str):
        orders = order_map[value]
    else:
        orders = []

    return orders


def read_wordlist_data(filepath, comma=False):
    """
    Reads a wordlist in lingpy format and returns an equivalent MT object.

    Parameters
    ----------
    filepath : string
        Path to the wordlist.
    comma : bool
        Whether to use commas instead of tabulations as field separator
        (default: False)
    """

    # Get the right field delimiter
    delimiter = "\t,"[comma]

    # Collect information in a single data structure, for later processing
    with open(filepath) as handler:
        rows = [row for row in csv.DictReader(handler, delimiter=delimiter)]

    return rows


def check_data(data, fields, id_field="id"):
    """
    Auxiliary function for data validation.

    The function will first check if all mandatory `fields` are found
    in all `data` rows, and then check if the `id_field` is unique.
    An exception is thrown if one of the checks fails.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries with the data.
    fields : dict
        A dictionary with the mandatory fields as keys and their corresponding
        keys in `data`. Intended to be the `.field` property of a
        MultiTiers object.
    id_field : str
        The name of the id_field (default: "id").
    """

    # First check if all the fields (including ID) are found in all entries
    for row in data:
        row_cols = list(row.keys())
        missing_fields = [
            field
            for field, col_name in fields.items()
            if col_name not in row_cols
        ]
        if missing_fields:
            raise ValueError(f"One or more mandatory fields missing in {row}")

    # Collect all ids and make sure they are unique
    row_ids = {row[fields[id_field]] for row in data}
    if len(row_ids) != len(data):
        raise ValueError("Data has non-unique IDs.")


def check_synonyms(data, cogid_field, doculect_field):
    """
    Auxiliary function for detecting synonyms.

    Synonyms are internally defined as pairs of (cogid, doculect)
    with more than one entry. The function will throw an exception if
    at least one synonym is found (listing all the affected pairs)
    and pass silently otherwise.

    Parameters
    ----------
    data : list of dict
        A list of dictionaries containing the data.
    cogid_field : str
        The key for cogids in `data`.
    doculect_field : str
        The key for doculects in `data`.
    """

    cogid_doculect = Counter(
        [(row[cogid_field], row[doculect_field]) for row in data]
    )
    synonyms = sorted(
        [pair for pair, count in cogid_doculect.items() if count > 1]
    )
    if synonyms:
        raise ValueError("Synonym pairs were found: %s." % str(synonyms))
