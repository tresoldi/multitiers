from pathlib import Path
import multitiers

# Read data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)

# Build classifier
print("============ STUDY 1")
clf = multitiers.Classifier(data, models=["cv"], left=1, right=1)

# Study
X_tiers = {
    "index": {"include": [1]},  # first position in word...
    "segment_Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "cv_Proto-Germanic_R1": {},  # any following class
}
y_tiers = {"segment_German": {"exclude": ["r"]}}  # and G doesn't have /r/

clf.train(X_tiers, y_tiers)
clf.to_graphviz("docs/germanic")

#############

print("============ STUDY 2")

# Build classifier
clf2 = multitiers.Classifier(data, models=["sca"], left=1, right=1)

study = """
X_tier index INCLUDE 1
X_tier segment_Proto-Germanic INCLUDE s
X_tier sca_Proto-Germanic_L1
X_tier sca_Proto-Germanic_R1
y_tier segment_German EXCLUDE r
y_tier segment_English
"""

X_tiers2, y_tiers2 = multitiers.utils.parse_study(study)
clf2.train(X_tiers2, y_tiers2, max_depth=3)
clf2.to_graphviz("docs/germanic2")

###########

print("============ STUDY 3")

# Build classifier
clf3 = multitiers.Classifier(data, models=["cv"])

study3 = """
X_tier segment_German
X_tier segment_English
X_tier cv_Dutch INCLUDE V
y_tier segment_Dutch
"""

X_tiers3, y_tiers3 = multitiers.utils.parse_study(study3)
clf3.train(X_tiers3, y_tiers3, min_impurity_decrease=0.0333)
clf3.to_graphviz("docs/dutch_pred")

###########

print("============ STUDY 4")

# Build classifier
clf4 = multitiers.Classifier(data, models=["cv", "sca"], left=1, right=1)

study4 = """
X_tier segment_German
X_tier segment_English
X_tier sca_German
X_tier sca_German_L1
X_tier sca_German_R1
X_tier sca_English
X_tier sca_English_L1
X_tier sca_English_R1
X_tier cv_Dutch INCLUDE C
y_tier segment_Dutch
"""

X_tiers4, y_tiers4 = multitiers.utils.parse_study(study4)
clf4.train(X_tiers4, y_tiers4, max_depth=15)
clf4.show_pred(max_lines=10)
clf4.show_pred_prob(max_lines=10)

##############################

print("============ STUDY 5")

# Read data
source = Path(__file__).parent.parent / "resources" / "latin2spanish.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)

# Build classifier
clf = multitiers.Classifier(data, models=["sca"], left=1, right=1)

study = """
X_tier segment_Latin INCLUDE t
X_tier sca_Latin_L1
X_tier sca_Latin_R1
y_tier segment_Spanish
"""

X_tiers, y_tiers = multitiers.utils.parse_study(study)
clf.train(X_tiers, y_tiers)
clf.to_graphviz("docs/latin_t")

print(clf.feature_extraction("tree", num_feats=5))
print(clf.feature_extraction("lsvc"))
