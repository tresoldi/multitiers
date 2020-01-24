"""
Module with the implementation of the MultiTiers object.
"""

# Import Python standard libraries
from collections import defaultdict
import hashlib
import json

# Import 3rd party libraries
from tabulate import tabulate

# Import utility functions
from .utils import check_data
from .utils import check_synonyms
from .utils import clts_object
from .utils import get_orders
from .utils import parse_alignment
from .utils import sc_mapper
from .utils import shift_tier

# List of sound class models allowed
SOUND_CLASS_MODELS = ["cv", "dolgo", "asjp", "sca"]


class MultiTiers:
    """
    Class for representing a single multitier object.

    MultiTiers must be initialized with either `data`, a list of dictionaries
    including the mandatory fields, or `filename`, with the unmodified
    output of the `.save()` method. Loading from other data formats, such as
    from LingPy's wordlists or from CLDF datasets, is performed by first
    building a Python list of dictionaries following the intended structure
    and then feeding a MultiTiers object with it.

    Internally, tiers are stored in the `.tiers` dictionary. Tier names are
    mostly free, but a number are reserved (particulatly those for indexing),
    as determined by the internal `._reserved` list. Tiers whose name (i.e.,
    dictionary key) begin with an underscore carry complementary information
    which should not be used in computations (in particular, unique ids as
    provided in the raw data).

    Iteration over tier names should always follow the sorted list of keys
    returned by the `.tier_names()` method, which guarantees reproducibility
    and readibility. Likewise, the doculects should always be iterated
    following the order in the `self.doculects` property.
    """

    # Reserved tier names
    _reserved = ["index", "rindex"]

    # Maximum number of rows and columns for the .__str__() method
    num_rows = 20
    num_cols = 10

    def __init__(self, data=None, filename=None, **kwargs):
        """
        Initialize a MultiTiers object.

        Either `data` (a list of dictionaries in the expected format) or
        `filename` (the output of the `.save()` method) must be provided.

        Parameters
        ----------
        data : list of dictionaries
            List of dictionaries with the data for initialization.
        filename : str
            Path to the output of a MultiTiers.save() call.
        col_id : str
            Column name (i.e. key) containing the unique row ids in `data`
            (default: `ID`). The value is disregarded if the object is
            initialized with a `filename`.
        col_doculect : str
            Column name (i.e. key) containing the row doculects in `data`
            (default: `DOCULECT`). The value is disregarded if the object is
            initialized with a `filename`.
        col_cogid : str
            Column name (i.e. key) containing the row cogids in `data`
            (default: `COGID`). The value is disregarded if the object is
            initialized with a `filename`.
        col_alignment : str
            Column name (i.e. key) containing the row alignments in `data`
            (default: `ALIGNMENT`). The value is disregarded if the object is
            initialized with a `filename`.
        left :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the left to be consider
            (default : 0). The value is disregarded if the object is
            initialized with a `filename`.
        right :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the right to be consider
            (default : 0). The value is disregarded if the object is
            initialized with a `filename`.
        models : list
            A list of one or more sound class models to compute in terms
            of alignments, from the available models "cv", "dolgo",
            "asjp", and "sca" (default: empty list). The value is disregarded
            if the object is initialized with a `filename`.
        """

        # Initialize tier collection
        self.tiers = defaultdict(list)

        # Store the default CLTS mapper. In the future we might allow
        # different mappers, for example following Chomsky & Halle
        # classes, or any other user-provided feature system.
        self.clts = clts_object()

        # Make sure either `data` or `filename` was provided
        if data and filename:
            raise ValueError(
                "MultiTiers initialized with both `data` and `filename`."
            )
        if not data and not filename:
            raise ValueError(
                "MultiTiers initialized with neither `data` or `filename`."
            )

        # The default intended operation is to initialize new MultiTiers
        # from data; as such, we check for the "exception" of an
        # initialization from a save file, skipping over the rest of
        # the tasks in this .__init__() method.
        if filename:
            self._init_from_file(filename)

        # Store fields names and corresponding columns in data, either
        # user provided or defaults
        self.field = {
            "id": kwargs.get("col_id", "ID"),
            "doculect": kwargs.get("col_doculect", "DOCULECT"),
            "cogid": kwargs.get("col_cogid", "COGID"),
            "alignment": kwargs.get("col_alignment", "ALIGNMENT"),
        }

        # Store default left and right orders
        self.left = get_orders(kwargs.get("left", 0))
        self.right = get_orders(kwargs.get("right", 0))

        # Store sound class translators
        self.sc_translators = {}
        sc_models = set(kwargs.get("models", []))
        for model in sc_models:
            if model not in SOUND_CLASS_MODELS:
                raise ValueError(
                    "Invalid sound class model `%s` requested." % model
                )
            self.sc_translators[model] = self.clts.soundclass(model)

        # Collect all doculects as an ordered set, checking that no
        # reserved name is used
        self.doculects = sorted(
            {entry[self.field["doculect"]] for entry in data}
        )
        if any([tier_name in self.doculects for tier_name in self._reserved]):
            raise ValueError("Reserved tier name used as a doculect id.")

        # Check if all entries have (at least) the mandatory fields and
        # unique ids
        check_data(data, self.field)

        # Check for synonyms by counting pairs of (cogid, doculects),
        # raising an error if any pair is found
        check_synonyms(data, self.field["cogid"], self.field["doculect"])

        # Actually add data
        self._add_data(data)

    def _init_from_file(self, filename):
        raise ValueError("File initialization not implemented yet.")

    def _add_data(self, data):
        """
        Internal function for adding data during initialization.
        """

        # Distribute the data into a dictionary of `cogids`. This makes the
        # later iterations easier, and we only need to go through the
        # entire data once, also performing the normalization of alignments
        # (with potential of being computationally expansive) in the same
        # loop.
        # Note that, while it is also a bit more expansive, we copy the
        # provided data, guaranteeing that the structure provided by the
        # user is not modified.
        cogid_data = defaultdict(list)
        for row in data:
            entry = {
                "id": row[self.field["id"]],
                "doculect": row[self.field["doculect"]],
                "alignment": parse_alignment(row[self.field["alignment"]]),
            }

            cogid_data[row[self.field["cogid"]]].append(entry)

        # Add entries (in sorted order for reproducibility)
        for cogid, entries in sorted(cogid_data.items()):
            # Collect a doculect/alignment dictionary, so we can also obtain
            # the alignment length (for the positional tiers), check if all
            # alignments have the same length as required, and identify
            # missing doculects.
            rows = {entry["doculect"]: entry for entry in entries}

            # Get alignment lengths and check consistency
            alm_lens = {len(row["alignment"]) for row in rows.values()}
            if len(alm_lens) > 1:
                raise ValueError(
                    "Cogid `%s` has alignments of different sizes." % cogid
                )
            alm_len = list(alm_lens)[0]

            # Extend the positional tiers
            self.tiers["index"] += [idx + 1 for idx in range(alm_len)]
            self.tiers["rindex"] += list(range(alm_len, 0, -1))

            # Extend doculect vectors with alignment and id information
            for doculect in self.doculects:
                if doculect in rows:
                    alm_vector = rows[doculect]["alignment"]
                    id_vector = [rows[doculect]["id"]] * alm_len
                else:
                    alm_vector = [None] * alm_len
                    id_vector = [None] * alm_len

                # Extend the doculect tier (and the shifted ones, if any)
                # and the id ones
                self._extend_vector(alm_vector, doculect)
                self._extend_vector(id_vector, "_%s_id" % doculect, shift=False)

                # Extend sound class mappings (and shifted), if any
                for model, translator in self.sc_translators.items():
                    sc_vector = sc_mapper(alm_vector, translator)
                    self._extend_vector(sc_vector, "%s_%s" % (doculect, model))

    def _extend_vector(self, vector, tier_name, shift=True):
        """
        Internal function for extending a vector with optional shifting.
        """

        # Extend the provived vector
        self.tiers[tier_name] += vector

        # Extend the shifted vectors, if any and requested
        if shift:
            # Obtain the shifted tiers
            shifted_vectors = shift_tier(
                vector,
                tier_name,
                left_orders=self.left,
                right_orders=self.right,
            )

            # If we had a `vector` of Nones (such as for missing data),
            # the vectors in `shifter_vectors` will at this point include
            # both out-of-bounds symbols and Nones, but it makes more
            # sense for them to be full Nones. We could do the list
            # comprehension manually, but given that we have the new
            # shifted tier names in `shifted_vectors`, we can just
            # reuse it.
            if all(value is None for value in vector):
                shifted_vectors = {
                    tier_name: vector for tier_name in shifted_vectors
                }

            # Extend the shifted vectors
            for shifted_name, shifted_vector in shifted_vectors.items():
                self.tiers[shifted_name] += shifted_vector

    def tier_names(self):
        """
        Return a properly sorted list of the tiers in the current object.
        """

        tiers = list(self.tiers.keys())
        # first reserved
        tiers.remove("index")
        tiers.remove("rindex")
        # then doculects (alignments)
        for doculect in self.doculects:
            tiers.remove(doculect)

        return ["index", "rindex"] + self.doculects + sorted(tiers)

    def as_list(self):
        """
        Return a representation of the object's data as a list of lists.
        """

        list_repr = [
            [tier_name] + self.tiers[tier_name]
            for tier_name in self.tier_names()
        ]

        return list_repr

    def save(self, filename):
        """
        Save a textual representation of the object to disk.

        Parameters
        ----------
        filename : str
            Path to the output file.
        """

        # Build output data
        output = {
            "fields": self.field,
            "doculects": self.doculects,
            "data": self.tiers,
            "left": self.left,
            "right": self.right,
            "models": list(self.sc_translators),
        }

        # Write as JSON
        with open(filename, "w") as handler:
            json.dump(output, handler, indent=2)

    def __repr__(self):
        return str(self.as_list())

    def __str__(self):
        # Obtain the list representation (as in __repr__), select only the
        # first NUM_ROWS and NUM_COLS, and build a tabulate representation
        # (which needs to tranpose the data).
        data = [
            tier[: self.num_rows] for tier in self.as_list()[: self.num_cols]
        ]
        data = list(zip(*data))  # transposition

        # Collect basic statistics
        str_tiers = (
            "MultiTiers object (%i tiers, %i doculects, %i rows)\n\n"
            % (
                len(self.tier_names()),
                len(self.doculects),
                len(self.tiers["index"]),
            )
        )

        str_tiers += tabulate(data, tablefmt="simple")

        return str_tiers

    def __hash__(self):
        value = self.__repr__().encode("utf-8")

        return int(hashlib.sha1(value).hexdigest(), 16) % 10 ** 8
