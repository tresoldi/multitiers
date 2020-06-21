from pathlib import Path
import multitiers

# Read data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)

# Build classifier
clf = multitiers.Classifier(data)

# Study
X_tiers = {
    "index": {"include": [1]},  # first position in word...
    "Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "Proto-Germanic_cv_R1": {},  # any following class
}
y_tiers = {"German": {"exclude": ["r"]}}  # and G doesn't have /r/

clf.train(X_tiers, y_tiers)
clf.to_graphviz("docs/germanic")

#############

# Build classifier
clf2 = multitiers.Classifier(data)

study = """
X_tier index INCLUDE 1
X_tier Proto-Germanic INCLUDE s
X_tier Proto-Germanic_sca_L1
X_tier Proto-Germanic_sca_R1
y_tier German EXCLUDE r
y_tier English
"""

X_tiers2, y_tiers2 = multitiers.utils.parse_study(study)
clf2.train(X_tiers2, y_tiers2, max_depth=3)
clf2.to_graphviz("docs/germanic2")

###########

# Build classifier
clf3 = multitiers.Classifier(data)

study3 = """
X_tier German
X_tier English
X_tier Dutch_cv INCLUDE V
y_tier Dutch
"""

X_tiers3, y_tiers3 = multitiers.utils.parse_study(study3)
clf3.train(X_tiers3, y_tiers3, min_impurity_decrease=0.0333)
clf3.to_graphviz("docs/dutch_pred")

###########

# Build classifier
clf4 = multitiers.Classifier(data)

study4 = """
X_tier German
X_tier English
X_tier German_sca
X_tier German_sca_L1
X_tier German_sca_R1
X_tier English_sca
X_tier English_sca_L1
X_tier English_sca_R1
X_tier Dutch_cv INCLUDE C
y_tier Dutch
"""

X_tiers4, y_tiers4 = multitiers.utils.parse_study(study4)
clf4.train(X_tiers4, y_tiers4, max_depth=15)
clf4.show_pred(max_lines=10)
clf4.show_pred_prob(max_lines=10)

##############################

# Read data
source = Path(__file__).parent.parent / "resources" / "latin2spanish.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)

# Build classifier
clf = multitiers.Classifier(data)

study = """
X_tier Latin INCLUDE t
X_tier Latin_sca_L1
X_tier Latin_sca_R1
y_tier Spanish
"""

X_tiers, y_tiers = multitiers.utils.parse_study(study)
clf.train(X_tiers, y_tiers)
clf.to_graphviz("docs/latin_t")
