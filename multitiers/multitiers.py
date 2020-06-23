"""
Module with the implementation of the MultiTiers object.
"""

# Import Python standard libraries
from collections import defaultdict, Counter
import itertools
import hashlib
import json

# Import 3rd party libraries
import numpy as np
import pandas as pd
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

# TODO: extend class, use attributes
# TODO: should store with pandas internally?
# TODO: make sure tiers starting with underscore are never used, as per docs
# TODO: do we need a default left/right? shouldn't it be computed at each
#       necessary call? same thing for `models`
# TODO: explain, perhaps in docs, that tiers with doculect names
#       are supposed to carry segments
# TODO: make sure known/unknown and similar don't share tiers
# TODO: validate tier name, and remove `reserved`
class MultiTiers:
    """
    Class for representing a single multitier object.

    MultiTiers must be initialized with a mandatory `data` structure in the
    expected format. Loading from other data formats, such as
    from LingPy's wordlists or from CLDF datasets, is performed by first
    building a Python list of dictionaries following the intended structure
    and then feeding a MultiTiers object with it.

    Internally, tiers are stored in the `.tiers` dictionary. Tier names are
    mostly free, but a number are reserved (particularly those for indexing),
    as determined by the internal `._reserved` list. Tiers whose names (i.e.,
    dictionary key) begin with an underscore carry complementary information
    which should not be used in computations (in particular, unique ids as
    provided in the raw data).

    Iteration over tier names should always follow the sorted list of keys
    returned by the `.tier_names()` method, which guarantees reproducibility
    and readability. Likewise, the doculects should always be iterated
    following the order in the `self.doculects` property.
    """

    # Reserved tier names
    _reserved = ["index", "rindex"]

    # Maximum number of rows and columns for the .__str__() method
    num_rows = 20
    num_cols = 10

    def __init__(self, data, **kwargs):
        """
        Initialize a MultiTiers object.

        Either `data` (a list of dictionaries in the expected format) or
        `filename` (the output of the `.save()` method) must be provided.

        Parameters
        ----------
        data : list of dictionaries
            List of dictionaries with the data for initialization.
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
        alm_left :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the left to be consider for
            alignment (default : 0). Note that this does *not* affect
            context computation for extended tiers (such as those for sound
            classes).
        alm_right :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the right to be consider for
            alignment (default : 0). Note that this does *not* affect
            context computation for extended tiers (such as those for sound
            classes).
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
        # TODO: implement a dummy model, as for the comment above
        self.clts = clts_object()

        # Store fields names and corresponding columns in data, either
        # user provided or defaults
        # NOTE: data loaded with `._init_from_file()` will override this
        self.field = {
            "id": kwargs.get("col_id", "ID"),
            "doculect": kwargs.get("col_doculect", "DOCULECT"),
            "cogid": kwargs.get("col_cogid", "COGID"),
            "alignment": kwargs.get("col_alignment", "ALIGNMENT"),
        }

        # Store sound class translators
        self.sc_translators = {}

        # Collect all doculects as an ordered set, checking that no
        # reserved name is used
        self.doculects = sorted({entry[self.field["doculect"]] for entry in data})
        if any([tier_name in self.doculects for tier_name in self._reserved]):
            raise ValueError("Reserved tier name used as a doculect id.")

        # Check if entries have the mandatory fields and ids are unique
        check_data(data, self.field)

        # Check for synonyms by counting pairs of (cogid, doculects),
        # raising an error if any pair is found
        check_synonyms(data, self.field["cogid"], self.field["doculect"])

        # Actually add data / tiers
        self._add_data(data, kwargs.get("alm_left", 0), kwargs.get("alm_right", 0))

    def _add_data(self, data, left, right):
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
                raise ValueError(f"Cogid '{cogid}' has alignments of different sizes.")
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
                self._extend_vector(doculect, alm_vector, left, right)
                self._extend_vector(f"_{doculect}_id", id_vector, left=0, right=0)

    def _add_missing_tiers(self, tier_list):
        # add tiers which are needed
        # TODO: better way of finding names, share with other functions
        # TODO: correct for languages with underscore
        # TODO: currently computing for all doculects
        new_tier_l = defaultdict(list)
        new_tier_r = defaultdict(list)
        for tier_name in tier_list:
            if tier_name not in self.tiers:
                # is there a context?
                if Counter(tier_name)["_"] == 2:
                    doculect, model, context = tier_name.split("_")
                    context_dir, context_idx = context[0], context[1]

                    # store model name and requested contexts, so later in
                    # a single call we get the highest value
                    if context_dir == "L":
                        new_tier_l[model].append(int(context_idx))
                    elif context_dir == "R":
                        new_tier_r[model].append(int(context_idx))
                else:
                    doculect, model = tier_name.split("_")
                    new_tier_l[model].append(0)
                    new_tier_r[model].append(0)

        # Add tiers
        for model in set(list(new_tier_l) + list(new_tier_r)):
            self._add_tiers(
                model, max(new_tier_l.get(model, [0])), max(new_tier_r.get(model, [0]))
            )

    # TODO: requires doculects and index tiers already inserted, test
    # TODO: allow to exclude some doculects? or whitelist?
    # TODO: don't add tiers already there
    def _add_tiers(self, model, left=0, right=0):
        # Load and cache sc_translator, if not available
        if model not in self.sc_translators:
            if model not in SOUND_CLASS_MODELS:
                raise ValueError(f"Invalid sound class model `{model}`.")
            self.sc_translators[model] = self.clts.soundclass(model)

        # Obtain doculect alignment tier, compute sound class tier,
        # and add it (extending if necessary)
        for doculect in self.doculects:
            sc_vector = sc_mapper(self.tiers[doculect], self.sc_translators[model])
            self._extend_vector(
                f"{doculect}_{model}", sc_vector, left=left, right=right
            )

    # TODO: left and right mandatory?
    def _extend_vector(self, tier_name, vector, left=0, right=0):
        """
        Internal function for extending a vector with optional shifting.
        """

        # Extend the provived vector
        self.tiers[tier_name] += vector

        # Extend the shifted vectors, if any and if requested
        if any([left, right]):
            # Obtain the shifted tiers
            shifted_vectors = shift_tier(
                vector,
                tier_name,
                left_orders=get_orders(left),
                right_orders=get_orders(right),
            )

            # If we had a `vector` of Nones (such as for missing data),
            # the vectors in `shifter_vectors` will at this point include
            # both out-of-bounds symbols and Nones, but it makes more
            # sense for them to be full Nones. We could do the list
            # comprehension manually, but given that we have the new
            # shifted tier names in `shifted_vectors`, we can just
            # reuse it.
            if all(value is None for value in vector):
                shifted_vectors = {tier_name: vector for tier_name in shifted_vectors}

            # Extend the shifted vectors
            for shifted_name, shifted_vector in shifted_vectors.items():
                self.tiers[shifted_name] += shifted_vector

    # TODO: decide what to do with None, currently just skipping
    #       (could use OneHotEncoder handle_unknown
    # TODO: explain multiple y, collected as a tuple
    def filter_Xy(self, X_tiers, y_tiers, use_dummies=True):
        # add tiers which are needed
        self._add_missing_tiers(list(X_tiers) + list(y_tiers))

        X = []
        y = []

        all_tiers = X_tiers.copy()
        all_tiers.update(y_tiers)
        for idx in range(len(self.tiers["index"])):
            filtered = True

            for tier, tier_info in all_tiers.items():
                if self.tiers[tier][idx] is None:
                    filtered = False
                    break

                if "include" in tier_info:
                    if self.tiers[tier][idx] not in tier_info["include"]:
                        filtered = False
                        break
                if "exclude" in tier_info:
                    if self.tiers[tier][idx] in tier_info["exclude"]:
                        filtered = False
                        break

            if filtered:
                X.append([self.tiers[tier][idx] for tier in X_tiers.keys()])
                y.append([self.tiers[tier][idx] for tier in y_tiers.keys()])

        # make pandas dfs
        X = pd.DataFrame(X, columns=X_tiers)
        y = pd.DataFrame(y, columns=y_tiers)

        # if we have more than one y_tier, join as a single, tuple one
        # TODO: how to do it better in pandas?
        # TODO: do our own mapping for > 1 tiers
        if len(y_tiers) > 1:
            y["/".join(y_tiers)] = [
                "/".join(row) for row in zip(*[list(y[column]) for column in y_tiers])
            ]
            y = y.drop(columns=y_tiers)

        # TODO: check `drop_first` for colinearity
        # TODO: deal with numeric/boolean columns, can be complex
        if use_dummies:
            X = pd.get_dummies(X, prefix_sep="_", drop_first=True)

        y = np.array(y).reshape(-1)

        return X, y

    def correspondence_study(self, study_known, study_unknown):
        # add tiers which are needed
        self._add_missing_tiers(list(study_known) + list(study_unknown))

        # collect filtered data
        # TODO: to it in a better, less expansive way
        data = [
            {
                tier: self.tiers[tier][idx]
                for tier in list(study_known) + list(study_unknown)
            }
            for idx in range(len(self.tiers["index"]))
        ]

        for group in [study_known, study_unknown]:
            for tier, value in group.items():
                # filter inclusion
                if "include" in value:
                    data = [row for row in data if row[tier] in value["include"]]

                # filter exclusion
                if "exclude" in value:
                    rows = [row for row in data if row[tier] not in value["exclude"]]

        # collect tuples
        entries = []
        for entry in data:
            known = tuple([entry[tier] for tier in study_known])
            unknown = tuple([entry[tier] for tier in study_unknown])
            entries.append((known, unknown))
        c = Counter(entries)

        # tabulate in a proper way, as a dictionary of known/unknown
        results = defaultdict(dict)
        for (known, unknown), count in c.items():
            results[known][unknown] = count

        return dict(results)

    # TODO: compute at each change and cache
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

    def get_correlation(self):
        import catcoocc

        # combinations/permutatios
        for tier_pair in itertools.combinations(self.tier_names(), 2):
            #print (self.tiers[tier_pair[0]], self.tiers[tier_pair[0]])
            cv = catcoocc.correlation.cramers_v(self.tiers[tier_pair[0]], self.tiers[tier_pair[1]])
            print(tier_pair, cv)

    def as_list(self):
        """
        Return a representation of the object's data as a list of lists.
        """

        list_repr = [
            [tier_name] + self.tiers[tier_name] for tier_name in self.tier_names()
        ]

        return list_repr

    def as_dataframe(self):
        return pd.DataFrame.from_dict(self.tiers)

    def __repr__(self):
        return str(self.as_list())

    def __str__(self):
        # Obtain the list representation (as in __repr__), select only the
        # first NUM_ROWS and NUM_COLS, and build a tabulate representation
        # (which needs to tranpose the data).
        data = [tier[: self.num_rows] for tier in self.as_list()[: self.num_cols]]
        data = list(zip(*data))  # transposition

        # Collect basic statistics
        str_tiers = "MultiTiers object (%i tiers, %i doculects, %i rows)\n\n" % (
            len(self.tier_names()),
            len(self.doculects),
            len(self.tiers["index"]),
        )

        str_tiers += tabulate(data, tablefmt="simple")

        return str_tiers

    # TODO: make tuples and just hash tuples
    def __hash__(self):
        value = self.__repr__().encode("utf-8")

        return int(hashlib.sha1(value).hexdigest(), 16) % 10 ** 8
