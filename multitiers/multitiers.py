from collections import defaultdict
import random

from tabulate import tabulate

# TODO: try to remove entirely
def get_orders(left, right):
    """
    Maps a pair of `left` and `right` arguments to the ranges they specify.
    """

    # Dictionary used for mapping string descriptions of window size to
    # actual Python ranges; by mapping to `range()` here in advance
    # (and consuming such range into a list), computations is a bit
    # faster and, in particular, is clearer. Note that we always start
    # from 1, so no zero-length is included in the lists (the zero distance
    # is the actual alignment site itself).
    # TODO: if kept, should be moved out of the function
    _ORDER_MAP = {
        "bigram": list(range(1, 2)),
        "trigram": list(range(1, 3)),
        "fourgram": list(range(1, 4)),
    }

    # get left mapping
    if isinstance(left, int):
        left = range(1, left + 1)
    elif isinstance(left, str):
        left = _ORDER_MAP[left]
    else:
        left = []

    # get right mapping
    if isinstance(right, int):
        right = range(1, right + 1)
    elif isinstance(right, str):
        right = _ORDER_MAP[right]
    else:
        right = []

    return left, right


# TODO: decide on tier names
def shift_tier(tier, vector, left, right):
    _GAP = None

    left_orders, right_orders = get_orders(left, right)
    # TODO: should pad the string with the number with zeros, for alignment;
    # users are not supposed to change in the middle

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


# TODO: should add cogid tier? maybe with a name which by default is
# not considered in the analyses (underscore?)
class MultiTiers:
    """
    Class for representing a single multitier object.

    Tiers are essentially dictionaries of sequences (lists, arrays, etc.)
    """

    # Reserved tier names
    _reserved = ["index", "rindex"]

    def __init__(self, data, clts=None):
        """
        Initialize a MultiTiers object.
        """

        # TODO: if no clts, get default from utils.

        # Collect the doculects as an ordered set
        # TODO: cannot have reserved names, as `index` and `r_index`
        # doculects = sorted(set([row['doculect'] for row in rows]))
        doculects = [
            set([entry["doculect"] for entry in cogid_values])
            for cogid_values in data.values()
        ]
        doculects = sorted(
            set([doculect for cogid in doculects for doculect in cogid])
        )
        for tier_name in self._reserved:
            if tier_name in doculects:
                raise ValueError(
                    "Reserved name `%s` used a doculect id." % tier_name
                )

        # Initialize tier collection with doculect alignments and reserved tiers
        # TODO: move to defaultdict(list)?
        # self.tiers = {doculect: [] for doculect in doculects + self._reserved}
        self.tiers = defaultdict(list)

        # cache soundclass translator
        sca = clts.soundclass("sca")

        # Add entries in sorted order
        count = 0
        for cogid in sorted(data):
            # Get data
            entries = data[cogid]

            # Collect a doculect/alignment length dictionary, so we can obtain
            # the alignment length (for the positional tiers), check if all
            # alignments have the same length as required, and identify
            # missing doculects.
            # TODO: what if we have synonyms?
            alignments = {
                entry["doculect"]: entry["alignment"] for entry in entries
            }
            alm_lens = set([len(alm) for alm in alignments.values()])
            if len(alm_lens) > 1:
                raise ValueError(
                    "Cogid `%s` has alignments of different sizes." % cid
                )
            alm_len = list(alm_lens)[0]

            # TODO: decide on using None
            for doculect in doculects:
                # Get the alignment for the current doculect, building an
                # empty one if missing
                alignment = alignments.get(doculect, None)
                if not alignment:
                    alignment = [None] * alm_len

                # Extend the doculect tier
                self.tiers[doculect] += alignment

                # Obtain the shifted tiers
                shifted = shift_tier(doculect, alignment, left=2, right=1)
                for shifted_name, shifted_vector in shifted.items():
                    self.tiers[shifted_name] += shifted_vector

                # map to all requested sound classes
                # TODO: check status of https://github.com/cldf-clts/pyclts/issues/7
                # TODO: deal with Nones
                sc_vector = [token if token else "-" for token in alignment]
                sc_vector = sca(" ".join(sc_vector))
                self.tiers["%s_SCA" % doculect] += sc_vector

            # Extend the positional tiers
            # TODO: deal with markers, etc.
            self.tiers["index"] += [idx + 1 for idx in range(alm_len)]
            self.tiers["rindex"] += [idx for idx in range(alm_len, 0, -1)]

            count += 1
            if count == 2:
                break

    # TODO: add some limit, maybe having __repr__ as limitless (with a general tabulate)
    # TODO: estimate/compute number of doculects?
    def __str__(self):
        # set RNG for randomly selecting the same number of columns (if necessary)
        #        random.seed("calc")

        tiers = sorted(self.tiers.keys())
        tiers.remove("index")
        tiers.remove("rindex")
        tiers = ["index", "rindex"] + sorted(
            random.sample(tiers, min(len(tiers), 7))
        )

        # Build the data
        data = []
        vector_length = len(self.tiers["index"])
        for pos in range(vector_length):
            row = [self.tiers[tier_name][pos] for tier_name in tiers]
            data.append(row)

        # Build the representation and return
        str_tiers = "MultiTiers with %i tiers (length %i)\n\n" % (
            len(self.tiers),
            vector_length,
        )
        str_tiers += tabulate(data, headers=tiers, tablefmt="simple")

        return str_tiers
