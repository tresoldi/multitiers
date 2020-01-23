from collections import defaultdict
import hashlib
import random

from tabulate import tabulate

# TODO: order
from .utils import clts_object, reduce_alignment, sc_mapper, get_orders

# TODO: presence/absence vector
# TODO: phonological feature vectors

# TODO: decide on tier names
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


# TODO: should add cogid/id tier? maybe with a name which by default is
# not considered in the analyses (underscore?)
# TODO: load/save
class MultiTiers:
    """
    Class for representing a single multitier object.

    Tiers are essentially dictionaries of sequences (lists, arrays, etc.)
    """

    # Reserved tier names
    _reserved = ["index", "rindex"]

    def __init__(self, data, **kwargs):
        """
        Initialize a MultiTiers object.
        """

        # Store data column names
        self.col = {
            "doculect": kwargs.get("doculect", "DOCULECT"),
            "cogid": kwargs.get("cogid", "COGID"),
            "alignment": kwargs.get("alignment", "ALIGNMENT"),
        }

        # Store default left and right orders
        self.left = get_orders(kwargs.get("left", 0))
        self.right = get_orders(kwargs.get("right", 0))

        # If no `clts` mapper was provided, store the default one (with the
        # distributed copy of clts-data)
        if "clts" not in kwargs:
            self.clts = clts_object()
        else:
            self.clts = kwargs["clts"]

        # Store sound class translators
        sc_models = kwargs.get("models", ["sca"])
        self.sc_translators = {
            model: self.clts.soundclass(model) for model in sc_models
        }

        # Initialize tier collection
        self.tiers = defaultdict(list)

        # Collect all doculects as an ordered set, checking that no
        # reserved name is used
        self.doculects = sorted(
            set([entry[self.col["doculect"]] for entry in data])
        )
        if any([tier_name in self.doculects for tier_name in self._reserved]):
            raise ValueError("Reserved tier name used as a doculect id.")

        # Actually add data
        self.add_data(data)

    # TODO: Allow extension, etc.
    # TODO: note in documentation that it can be extended, but doculects
    # must match, etc.
    # TODO: check if all entries have all mandatory fields, and decide what
    # to do if not, and unique IDs
    def add_data(self, data):
        # Split the data into a dictionary of `cogids`. This makes the
        # later iteration easier, and we only need to go through the
        # entire data once, and we can already convert the alignments
        # in the same loop.
        # Note that this essentially performs a copy operation which,
        # while more expansive, guarantees we don't change `data` itself.
        cogid_data = defaultdict(list)
        for row in data:
            cogid = row[self.col["cogid"]]
            entry = {
                "doculect": row[self.col["doculect"]],
                "alignment": reduce_alignment(
                    row[self.col["alignment"]].split()
                ),
            }

            cogid_data[cogid].append(entry)

        # Add entries (in sorted order for reproducibility)
        # TODO: what if we have synonyms?
        for cogid, entries in sorted(cogid_data.items()):
            # Collect a doculect/alignment dictionary, so we can also obtain
            # the alignment length (for the positional tiers), check if all
            # alignments have the same length as required, and identify
            # missing doculects.
            alignments = {
                entry["doculect"]: entry["alignment"] for entry in entries
            }

            # Get alignment length and check consistency
            alm_lens = set([len(alm) for alm in alignments.values()])
            if len(alm_lens) > 1:
                raise ValueError(
                    "Cogid `%s` has alignments of different sizes." % cogid
                )
            alm_len = list(alm_lens)[0]

            # Extend the positional tiers
            # TODO: deal with markers, etc.
            self.tiers["index"] += [idx + 1 for idx in range(alm_len)]
            self.tiers["rindex"] += [idx for idx in range(alm_len, 0, -1)]

            # Extend vectors with alignment information
            for doculect in self.doculects:
                # Get the doculect alignment, defaulting to an empty one
                alignment = alignments.get(doculect, None)
                if not alignment:
                    alignment = [None] * alm_len

                # Extend the doculect tier (and the shifted ones, if any)
                self._extend_vector(alignment, doculect)

                # Extend sound class mappings (and shifted), if any
                # TODO: deal with Nones
                for model, translator in self.sc_translators.items():
                    sc_vector = sc_mapper(alignment, translator)
                    sc_name = "%s_%s" % (doculect, model)
                    self._extend_vector(sc_vector, sc_name)

    # TODO: flag not to run shifted
    def _extend_vector(self, vector, tier_name):
        # Extend the provived vector
        self.tiers[tier_name] += vector

        # Extend the shifted vectors, if any
        self._extend_shifted_vector(vector, tier_name)

    def _extend_shifted_vector(self, vector, base_name):
        # Obtain the shifted tiers
        # TODO: change order to vector/name
        shifted_vectors = shift_tier(
            base_name, vector, left_orders=self.left, right_orders=self.right
        )

        for shifted_name, shifted_vector in shifted_vectors.items():
            self.tiers[shifted_name] += shifted_vector

    def tier_names(self):
        """
        Return a properly sorted list of the tiers in the current object.
        """

        tiers = list(self.tiers.keys())
        tiers.remove("index")
        tiers.remove("rindex")

        return ["index", "rindex"] + sorted(tiers)

    def __repr__(self):
        v = [
            [tier_name] + self.tiers[tier_name]
            for tier_name in self.tier_names()
        ]

        return str(v)

    # TODO: add some limit, maybe having __repr__ as limitless (with a general tabulate)
    # TODO: estimate/compute number of doculects?
    def __str__(self):
        # set RNG for randomly selecting the same number of columns (if necessary)
        #        random.seed("calc")

        tiers = self.tier_names()
        tiers = tiers[:2] + sorted(
            random.sample(tiers[2:], min(len(tiers[2:]), 7))
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
        str_tiers += tabulate(data[:10], headers=tiers, tablefmt="simple")

        return str_tiers

    def __hash__(self):
        # We build a representation similar to that of __repr__, but as
        # tuples
        # to be used for compariosn
        value = self.__repr__().encode("utf-8")

        return int(hashlib.sha1(value).hexdigest(), 16) % 10 ** 8
