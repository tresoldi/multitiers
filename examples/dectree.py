from pathlib import Path
import re

from sklearn import preprocessing
from sklearn import tree
import graphviz

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

import multitiers

source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=0)
mt = multitiers.MultiTiers(data, left=2, right=1, models=["sca"])

# obtain X/y
X_tiers = {
#        "index":{"includes" : [1]},
        "Proto-Germanic":{"includes":["s", "p"]},
        "Proto-Germanic_sca_L1":{},
        "Proto-Germanic_sca_R1":{},
#        "English":{},
#        "Dutch":{},
    }
y_tiers = {
        "German":{},#{"excludes":["r"]},
    #    "English":{},
    }

# Obtain X/y
X, y = mt.filter_Xy(X_tiers, y_tiers)

# `y` is mapped to an appropriate np.array, so we can use the Encoder
y_encoder = LabelEncoder()
y_encoder.fit(y)
y_le = y_encoder.transform(y)

# fit decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y_le)

# show predictions
#y_pred = clf.predict(X)
#for o, p in zip(y, y_encoder.inverse_transform(y_pred)):
#    print(o, p)

# show prediction probabilities
y_pred = clf.predict_proba(X)

# visualize
proportion = False
dot_data = tree.export_graphviz(clf, out_file=None,
                      feature_names=X.columns,
                      class_names=y_encoder.classes_,
                      filled=True, rounded=True, proportion=proportion,
                      special_characters=True)

# improve boolean notation in dot_data
dot_data = re.sub(
    r"label=<(?P<label>.+)_(?P<value>.+) &le; 0\.5",
    "label=<\g<label> is not \g<value>",
    dot_data)
class_label = "segment"
dot_data = dot_data.replace("<br/>class =", f"<br/>{class_label} =")

#
#RE_LABEL = re.compile()
#matches = RE_LABEL.findall(dot_data)
#for m in matches:
#    print(m)

graph = graphviz.Source(dot_data)
graph.render("germanic", cleanup=False)
