from collections import defaultdict
from pathlib import Path
import re

from sklearn import preprocessing
from sklearn import tree
import graphviz

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

import multitiers


class Classifier:
    def __init__(self, data, left=2, right=2):
        # build internal multitiers object
        self.mt = multitiers.MultiTiers(data, left=left, right=right)

    def train(self, X_tiers, y_tiers, model="decision_tree"):
        # get X and y vectors
        self.X, y = self.mt.filter_Xy(X_tiers, y_tiers)
        self.y_encoder = LabelEncoder()
        self.y_encoder.fit(y)
        y_le = self.y_encoder.transform(y)

        # train classifier
        if model == "decision_tree":
            max_depth = 5

            self.clf = tree.DecisionTreeClassifier(max_depth=max_depth)
            self.clf = self.clf.fit(self.X, y_le)  # TODO: need attribution?

        else:
            raise ValueError("not implemented")

    # TODO: accept data instead of X (make pipeline)
    def show_pred(self, X=None):
        if not X:
            X = self.X

        y_pred = clf.predict(X)
        for o, p in zip(y, self.y_encoder.inverse_transform(y_pred)):
            print(o, p)

    # TODO: accept data instead of X (make pipeline)
    def show_pred(self, X=None):
        if not X:
            X = self.X

        y_pred = clf.predict_proba(X)
        for o, p in zip(y, y_pred):
            print(o, p)

    def to_dot(self, proportion=False):
        # Build dot source
        dot_data = tree.export_graphviz(
            self.clf,
            out_file=None,
            feature_names=self.X.columns,
            class_names=self.y_encoder.classes_,
            filled=True,
            rounded=True,
            proportion=proportion,
            special_characters=True,
        )

        # improve notation in dot_data
        dot_data = re.sub(
            r"label=<(?P<label>.+)_(?P<value>.+) &le; 0\.5",
            "label=<\g<label> is \g<value>",
            dot_data,
        )
        class_label = "/".join(y_tiers)
        dot_data = dot_data.replace("<br/>class =", f"<br/>{class_label} =")
        dot_data = dot_data.replace('headlabel="True"', "$$$HOLDER$$$")
        dot_data = dot_data.replace('headlabel="False"', 'headlabel="True"')
        dot_data = dot_data.replace("$$$HOLDER$$$", 'headlabel="False"')

        return dot_data


# Read data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)

# Build classifier
clf = Classifier(data)

# Study
X_tiers = {
    "index": {"includes": [1]},
    "Proto-Germanic": {"includes": ["s"]},
    "Proto-Germanic_sca_L1": {},
    "Proto-Germanic_sca_R1": {},
    #        "Dutch":{},
}
y_tiers = {"German": {}, "English": {}}  # {"excludes":["r"]},

clf.train(X_tiers, y_tiers)

dot_data = clf.to_dot()

graph = graphviz.Source(dot_data)
graph.render("germanic", cleanup=False)
