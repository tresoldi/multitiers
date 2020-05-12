from pathlib import Path
import multitiers

source = Path(__file__).parent.parent / "resources" / "germanic.tsv"
data = multitiers.read_wordlist_data(source.as_posix(), comma=0)
mt = multitiers.MultiTiers(data, left=2, right=1, models=["cv"])

# TODO: have a small language for includes/excludes
study = [
        # initial position
        {
            "tier_name": "index",
            "includes": [1],
            "excludes": None,
            "unknown": False,
        },
        # All Proto-Germanic /s/
        {
            "tier_name": "Proto-Germanic",
            "includes": ["s"],
            "excludes": None,
            "unknown": False,
        },
        # No German r /s/
        {
            "tier_name": "German",
            "includes": None,
            "excludes": ["r"],
            "unknown": False,
        },
        # Proto-Germanic CV to the left
        {
            "tier_name": "Proto-Germanic_cv_L1",
            "includes": None,
            "excludes": None,
            "unknown": True,
        },
        # Proto-Germanic CV to the right
        {
            "tier_name": "Proto-Germanic_cv_R1",
            "includes": None,
            "excludes": None,
            "unknown": True,
        },
    ]

data = mt.filter(study)

study_result = mt.study(study)

from pprint import pprint
pprint(study_result)
