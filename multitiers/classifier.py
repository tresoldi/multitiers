import re

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import graphviz

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import multitiers

# TODO: make sure it is resuable
class Classifier:
    def __init__(self, data, left=2, right=2):
        # build internal multitiers object
        self.mt = multitiers.MultiTiers(data, left=left, right=right)

    def train(
        self,
        X_tiers,
        y_tiers,
        model="decision_tree",
        max_depth=5,
        min_impurity_decrease=0.0,
    ):
        self.X_tiers = X_tiers
        self.y_tiers = y_tiers

        # get X and y vectors
        self.X, self.y = self.mt.filter_Xy(self.X_tiers, self.y_tiers)
        self.y_encoder = LabelEncoder()
        self.y_encoder.fit(self.y)
        self.y_le = self.y_encoder.transform(self.y)

        # train classifier
        if model == "decision_tree":
            self.clf = tree.DecisionTreeClassifier(
                max_depth=max_depth, min_impurity_decrease=min_impurity_decrease
            )
            self.clf = self.clf.fit(self.X, self.y_le)  # TODO: need attribution?

        else:
            raise ValueError("not implemented")

    # TODO: accept data instead of X (make pipeline)
    def show_pred(self, X=None, max_lines=None):
        if not X:
            X = self.X

        y_pred = self.clf.predict(X)
        for idx, (orig, pred) in enumerate(
            zip(self.y, self.y_encoder.inverse_transform(y_pred))
        ):
            print(f"#{idx}: {orig}/{pred} -- {orig == pred}")
            if max_lines and max_lines == idx:
                break

    # TODO: accept data instead of X (make pipeline)
    def show_pred_prob(self, X=None, max_lines=None):
        if not X:
            X = self.X

        print()

        y_pred = self.clf.predict_proba(X)
        for idx, (orig, pred) in enumerate(zip(self.y, y_pred)):
            # zip list of predictions/classes, excluding those with zero,
            # sorting
            predicts = list(zip(pred, self.y_encoder.classes_))
            predicts = [p for p in predicts if p[0] > 0.0]
            predicts = sorted(predicts, key=lambda p: p[0], reverse=True)[:3]
            predicts = ",".join(["%s|%.3f" % (y, p) for p, y in predicts])

            print(f"#{idx}: {orig}/({predicts})")
            if max_lines and max_lines == idx:
                break

    # TODO: Rename to feature importance?
    def feature_extraction(self, model, **kwargs):
        if model == "tree":
            num_feats = kwargs.get("num_feats")

            etclf = ExtraTreesClassifier(n_estimators=50)
            etclf = etclf.fit(self.X, self.y_le)

            features = list(zip(self.X, etclf.feature_importances_))
            features = sorted(features, key=lambda f: f[1], reverse=True)
            return features[:num_feats]
        elif model == "lsvc":
            C = kwargs.get("C", 0.01)

            lsvc = LinearSVC(C=C, penalty="l1", dual=False).fit(self.X, self.y_le)
            model = SelectFromModel(lsvc, prefit=True)

            features = list(zip(self.X, model.get_support()))
            features = [feature for feature in features if feature[1]]
            return features
        else:
            raise ValueError("Model not implemented")

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

        # replace value array with readable one
        def _array2str(matchobj):
            # grab array, remove any internal "<br/>", split into ints
            val = matchobj.group("array").replace("<br/>", ",")
            val = [int(v.strip()) for v in val.split(",")]

            # zip into class/count tuples, remove zero counts, sort, map
            # to strings, insert a "<br/>" at every interval (so node is not
            # too large)
            val = list(zip(val, self.y_encoder.classes_))
            val = [item for item in val if item[0] > 0]
            val = sorted(val, key=lambda i: i[0], reverse=True)
            val = [f"{y}={c}," for c, y in val]
            interval = 5
            if len(val) > interval:
                val = [
                    x
                    for l in [
                        val[i : i + interval] + ["<br/>"]
                        for i in range(len(val) // interval)
                    ]
                    for x in l
                ]
                if val[-1] == "<br/>":
                    val = val[:-1]

            # return (with [:-1] for the final comma)
            val = " ".join(val)
            return f">values = [{val[:-1]}]<br/>"

        dot_data = re.sub(r">value = \[(?P<array>.+)\]<br\/>", _array2str, dot_data)

        # class labels and arrow labels
        class_label = "/".join(self.y_tiers)
        dot_data = dot_data.replace("<br/>class =", f"<br/>{class_label} =")
        dot_data = dot_data.replace('headlabel="True"', "$$$HOLDER$$$")
        dot_data = dot_data.replace('headlabel="False"', 'headlabel="True"')
        dot_data = dot_data.replace("$$$HOLDER$$$", 'headlabel="False"')

        return dot_data

    def to_graphviz(self, filename, format="png"):
        graph = graphviz.Source(self.to_dot(), format=format)
        graph.render(filename, cleanup=True)
