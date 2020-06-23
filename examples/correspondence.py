from pathlib import Path
from pprint import pprint
import multitiers

print("=========== STUDY 1")

# load data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)
mt = multitiers.MultiTiers(data, models=["cv"], left=1, right=1)

# run a correspondence study
known1 = {
    "index": {"include": [1]},  # first position in word...
    "segment_Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "segment_German": {"exclude": ["r"]},  # and G doesn't have /r/
}
unknown1 = {"cv_Proto-Germanic_R1": {}}
study_result1 = mt.correspondence_study(known1, unknown1)

print("=========== STUDY 2")

# TODO: rename KNOWN->INCLUDE/EXCLUDE UNKNWON->FREE
demo_study = """
KNOWN index INCLUDE 1
KNOWN segment_Proto-Germanic INCLUDE s
KNOWN segment_German EXCLUDE r
UNKNOWN segment_Proto-Germanic_R1
"""
known2, unknown2 = multitiers.utils.parse_study(demo_study)
study_result2 = mt.correspondence_study(known2, unknown2)

# print results
print(mt)
pprint(study_result1)
pprint(study_result2)

######################

print("=========== STUDY 3")

# Read data
source = Path(__file__).parent.parent / "resources" / "latin2spanish.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)
mt = multitiers.MultiTiers(data, models=["cv"], left=1, right=1)

study = """
KNOWN segment_Latin INCLUDE t
KNOWN segment_Spanish INCLUDE tʃ
UNKNOWN cv_Latin_L1
UNKNOWN cv_Latin_R1
"""

known, unknown = multitiers.utils.parse_study(study)
study_result = mt.correspondence_study(known, unknown)

# print results
print(mt)
pprint(study_result)

mt.get_correlation()
