from pathlib import Path
from pprint import pprint
import multitiers

# load data
source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)
mt = multitiers.MultiTiers(data)

# run a correspondence study
known1 = {
    "index": {"include": [1]},  # first position in word...
    "Proto-Germanic": {"include": ["s"]},  # when PG has /s/
    "German": {"exclude": ["r"]},  # and G doesn't have /r/
}
unknown1 = {"Proto-Germanic_cv_R1": {}}
study_result1 = mt.correspondence_study(known1, unknown1)

demo_study = """
KNOWN index INCLUDE 1
KNOWN Proto-Germanic INCLUDE s
KNOWN German EXCLUDE r
UNKNOWN Proto-Germanic_cv_R1
"""
known2, unknown2 = multitiers.utils.parse_study(demo_study)
study_result2 = mt.correspondence_study(known2, unknown2)

# print results
print(mt)
pprint(study_result1)
pprint(study_result2)

######################

# Read data
source = Path(__file__).parent.parent / "resources" / "latin2spanish.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=False)
mt = multitiers.MultiTiers(data)

study = """
KNOWN Latin INCLUDE t
KNOWN Spanish INCLUDE tʃ
UNKNOWN Latin_cv_L1
UNKNOWN Latin_cv_R1
"""

known, unknown = multitiers.utils.parse_study(study)
study_result = mt.correspondence_study(known, unknown)

# print results
print(mt)
pprint(study_result)
