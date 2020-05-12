from pathlib import Path
from pprint import pprint
import multitiers

# load data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=0)
mt = multitiers.MultiTiers(data, left=2, right=1, models=["cv"])

# run a correspondence study
known = {
    "index":          {"include":[1]},   # first position in word...
    "Proto-Germanic": {"include":["s"]}, # when PG has /s/
    "German":         {"exclude":["r"]}  # and G doesn't have /r/
}

unknown = {
    "Proto-Germanic_cv_L1":{},
    "Proto-Germanic_cv_R1":{}
}

study_result = mt.correspondence_study(known, unknown)

# print results
print(mt)
pprint(study_result)
