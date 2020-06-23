"""
Module with the implementation of the MultiTiers object.
"""

# Import Python standard libraries
from collections import defaultdict
import itertools

# Import 3rd party libraries
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
import catcoocc

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

    MultiTiers must be initialized with a mandatory `data` structure in the
    expected format of a list of dictionaries. Loading from other data formats,
    such as from LingPy's wordlists or from CLDF datasets, is performed by
    first building a Python list of dictionaries following the intended
    structure and then feeding a MultiTiers object with it.

    Internally, tiers are stored in a pandas' dataframe.
    """

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
        models : list
            A list of one or more sound class models to compute in terms
            of alignments, from the available models "cv", "dolgo",
            "asjp", and "sca" (default: empty list). The value is disregarded
            if the object is initialized with a `filename`.
        left :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the left to be consider for
            alignment (default : 0). Note that this does *not* affect
            context computation for extended tiers (such as those for sound
            classes).
        right :  int or str
            An integer or a string (either "bigram", "trigram", or "fourgram")
            indicating the highest context to the right to be consider for
            alignment (default : 0). Note that this does *not* affect
            context computation for extended tiers (such as those for sound
            classes).
        """

        # Store the default CLTS mapper. In the future we might allow
        # different mappers, for example following Chomsky & Halle
        # classes, or any other user-provided feature system.
        # TODO: implement a dummy model, as for the comment above
        # TODO: replace with an actual dictionary of with a function,
        #       for pandas mapping? Decide in terms of the `translators`
        self.clts = clts_object()
        self.sc_translators = {}

        # Store fields names and corresponding columns in data, either
        # user provided or defaults
        self.field = {
            "id": kwargs.get("col_id", "ID"),
            "doculect": kwargs.get("col_doculect", "DOCULECT"),
            "cogid": kwargs.get("col_cogid", "COGID"),
            "alignment": kwargs.get("col_alignment", "ALIGNMENT"),
        }

        # Get models, left and right order
        models = kwargs.get("models", [])
        if models is None:
            models = []
        left_orders = get_orders(kwargs.get("left", 0))
        right_orders = get_orders(kwargs.get("right", 0))

        # Collect all doculects as an ordered set
        self.doculects = sorted({entry[self.field["doculect"]] for entry in data})

        # Check if entries have the mandatory fields and if ids are unique
        check_data(data, self.field)

        # Check for synonyms by counting pairs of (cogid, doculects),
        # raising an error if any pair is found
        # TODO: allow to override exception, taking the first or a random
        #       entry?
        check_synonyms(data, self.field["cogid"], self.field["doculect"])

        # Instantiate dataframe and add the data
        vector = defaultdict(list)

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
            # the alignment length (for the positional tiers)
            rows = {entry["doculect"]: entry for entry in entries}

            # Get alignment lengths and check for consistency
            alm_lens = {len(row["alignment"]) for row in rows.values()}
            if len(alm_lens) > 1:
                raise ValueError(f"Cogid '{cogid}' has alignments of different sizes.")
            alm_len = list(alm_lens)[0]

            # Extend the positional tiers
            vector["index"] += [idx + 1 for idx in range(alm_len)]
            vector["rindex"] += list(range(alm_len, 0, -1))

            # Extend doculect vectors with alignment and id information
            for doculect in self.doculects:
                if doculect in rows:
                    alm_vector = rows[doculect]["alignment"]
                    id_vector = [rows[doculect]["id"]] * alm_len
                else:
                    alm_vector = [np.nan] * alm_len
                    id_vector = [np.nan] * alm_len

                # Extend the doculect id tier and the segments ones (along
                # with the shifted tiers, if any)
                vector[f"id_{doculect}"] += id_vector

                vector[f"segment_{doculect}"] += alm_vector
                shifted = shift_tier(
                    alm_vector, f"segment_{doculect}", left_orders, right_orders
                )
                for shift_name, shift_vector in shifted.items():
                    vector[shift_name] += shift_vector

                for model in models:
                    vector[f"{model}_{doculect}"] += sc_mapper(
                        alm_vector, self.clts.soundclass(model)
                    )
                    shifted = shift_tier(
                        vector[f"{model}_{doculect}"],
                        f"{model}_{doculect}",
                        left_orders,
                        right_orders,
                    )
                    for shift_name, shift_vector in shifted.items():
                        vector[shift_name] = shift_vector

        # Build the data frame
        self.df = pd.DataFrame(vector)

    # TODO: decide what to do with None, currently just skipping
    #       (could use OneHotEncoder handle_unknown
    # TODO: explain multiple y, collected as a tuple
    def filter_Xy(self, X_tiers, y_tiers, use_dummies=True):
        # Make a single description of all tiers in the study
        all_tiers = X_tiers.copy()
        all_tiers.update(y_tiers)

        # First, apply filters in a copy of data, and drop NAs
        filtered = self.df.loc[:, list(all_tiers)]
        for tier, tier_info in all_tiers.items():
            if "include" in tier_info:  # TODO: remove this
                filtered = filtered.loc[filtered[tier].isin(tier_info["include"])]
            if "exclude" in tier_info:  # TODO: remove this
                filtered = filtered.loc[~filtered[tier].isin(tier_info["exclude"])]
        filtered = filtered.dropna()

        # Split in X and y
        X = filtered.loc[:, list(X_tiers)]
        y = filtered.loc[:, list(y_tiers)]

        # if we have more than one y_tier, join as a single, tuple one
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

        # Make sure `y` is a series
        y = y[y.columns[0]]

        return X, y

    def correspondence_study(self, study_known, study_unknown):
        # TODO: combine this repeated code from filter_Xy
        # Make a single description of all tiers in the study
        all_tiers = study_known.copy()
        all_tiers.update(study_unknown)

        # First, apply filters in a copy of data, and drop NAs
        filtered = self.df.loc[:, list(all_tiers)]
        for tier, tier_info in all_tiers.items():
            if "include" in tier_info:  # TODO: remove this
                filtered = filtered.loc[filtered[tier].isin(tier_info["include"])]
            if "exclude" in tier_info:  # TODO: remove this
                filtered = filtered.loc[~filtered[tier].isin(tier_info["exclude"])]
        filtered = filtered.dropna()

        # collect tuples of occurrence
        # qumin approch thanks to Sacha
        # TODO: make faster, with a pure groupby?
        known = filtered[study_known].apply(tuple, axis=1)
        unknown = filtered[study_unknown].apply(tuple, axis=1)
        counter = dict(unknown.groupby(known).value_counts())

        # tabulate in the expected way, as a dictionary of known/unknown
        # TODO: return as a normal counter
        results = defaultdict(dict)
        for (known, unknown), count in counter.items():
            results[known][unknown] = count

        return dict(results)

    # TODO: write using idiomatic pandas code
    def get_correlation(self):
        # combinations/permutatios
        for tier_pair in itertools.combinations(self.df.columns, 2):
            _tmp = self.df.loc[:, tier_pair]
            _tmp = _tmp.dropna()

            # print (self.tiers[tier_pair[0]], self.tiers[tier_pair[0]])
            cv = catcoocc.correlation.cramers_v(_tmp[tier_pair[0]], _tmp[tier_pair[1]])
            print(tier_pair, cv)

    def __hash__(self):
        return hash(tuple(hash_pandas_object(self.df)))
