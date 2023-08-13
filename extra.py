"""
Extra functions for the project.
"""

from pathlib import Path
from typing import Union, Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
import unicodedata
import re

BASE_PATH = Path(__file__).parent
RESOURCES_PATH = BASE_PATH / "resources"


def normalize_grapheme(grapheme: str) -> str:
    return unicodedata.normalize("NFD", grapheme)


def create_sound_functions() -> (
    Tuple[
        Callable[[str], Union[int, float]],
        Callable[[str], Union[str, float]],
        Callable[[str], Union[str, float]],
    ]
):
    """
    Create functions to get prosody, sca, and dolgopolsky values for a phoneme.

    @return: Tuple of functions to get prosody, sca, and dolgopolsky values for a phoneme.
    @rtype: tuple
    """

    # Read the sounds.csv file
    sounds_df = pd.read_csv("sounds.csv")

    # Normalize the "GRAPHEME" column
    sounds_df["GRAPHEME"] = sounds_df["GRAPHEME"].apply(normalize_grapheme)

    # Create dictionaries for prosody, sca, and dolgopolsky values
    prosody_dict = {
        row["GRAPHEME"]: float(row["PROSODY"]) if row["PROSODY"] != "?" else np.nan for index, row in
        sounds_df.iterrows()
    }
    sca_dict = {row["GRAPHEME"]: row["SCA"] for index, row in sounds_df.iterrows()}
    dolgopolsky_dict = {
        row["GRAPHEME"]: row["DOLGOPOLSKY"] for index, row in sounds_df.iterrows()
    }

    # Define the function to get prosody
    def get_prosody(phoneme: str) -> Union[float, type(np.nan)]:
        return prosody_dict.get(phoneme, np.nan)

    # Define the function to get sca
    def get_sca(phoneme: str) -> Union[str, float]:
        return sca_dict.get(phoneme, np.nan)

    # Define the function to get dolgopolsky
    def get_dolgopolsky(phoneme: str) -> Union[str, float]:
        return dolgopolsky_dict.get(phoneme, np.nan)

    return get_prosody, get_sca, get_dolgopolsky


def read_data(
    file_path: Union[str, Path],
    concept_col: str = "CONCEPT",
    doculect_col: str = "DOCULECT",
    alignment_col: str = "ALIGNMENT",
    drop_parentheses: bool = True,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Read alignment data from a tabular file.

    The file should have the following columns:
        - CONCEPT: Name of the concept.
        - DOCULECT: Name of the doculect.
        - ALIGNMENT: Alignment of the concept in the doculect.

    The alignment should be a string of phonemes separated by spaces.

    @param file_path: Path to the source file, can be a string or pathlib.Path object.
    @type file_path: Union[str, Path]
    @param concept_col: Name of the column for concepts in the source file.
    @type concept_col: str
    @param doculect_col: Name of the column for doculects in the source file.
    @type doculect_col: str
    @param alignment_col: Name of the column for alignments in the source file.
    @type alignment_col: str
    @param drop_parentheses: Whether to drop or keep the content inside parentheses in the alignments. Default is True, meaning the content will be dropped.
    @type drop_parentheses: bool
    @return: Dictionary with keys as (CONCEPT, DOCULECT) and values as lists of phonemes.
    @rtype: Dict[Tuple[str, str], List[str]]
    @raise ValueError: If there are inconsistent alignment lengths for the same concept.
    """

    # Detect the separator
    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline()
        separator = ',' if ',' in first_line else (';' if ';' in first_line else '\t')

    # Read data
    df = pd.read_csv(file_path, sep=separator)

    # Initialize data dictionary
    data = {}

    # Check alignment lengths
    alignment_lengths = {}

    # Iterate over rows and build data dictionary
    for index, row in df.iterrows():
        concept: str = row[concept_col]
        doculect: str = row[doculect_col]
        alignment_str = row[alignment_col]

        # Handle parentheses
        if drop_parentheses:
            alignment_str = re.sub(r'\(.*?\)', '', alignment_str)
        else:
            alignment_str = alignment_str.replace('(', '').replace(')', '')

        alignment = alignment_str.strip().split(" ")
        alignment = [normalize_grapheme(phoneme) for phoneme in alignment]

        # Check if alignment length is consistent
        if concept in alignment_lengths:
            if alignment_lengths[concept] != len(alignment):
                raise ValueError(
                    f"Inconsistent alignment lengths for concept '{concept}'."
                )
        else:
            alignment_lengths[concept] = len(alignment)

        # Add to data
        data[(concept, doculect)] = alignment

    return data