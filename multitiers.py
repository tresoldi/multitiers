import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import extra


def build_multitiers(
    data: Dict[Tuple[str, str], List[str]],
    left: int = 0,
    right: int = 0,
    function_dict: Optional[Dict[str, Callable]] = None,
    context_tiers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a multitiered representation of extended alignments.

    The function creates a DataFrame where each alignment site becomes a row, and phonemes from the doculects
    as columns. Additional columns are added using the functions in function_dict, and left and right context
    are added as specified.

    @param data: Dictionary with (parameter, doculect) tuples as keys and lists of phonemes as values.
    @param left: The size of the left context to include.
    @param right: The size of the right context to include.
    @param function_dict: Optional dictionary with tier names as keys and functions as values to map phonemes.
    @param context_tiers: Optional list of tiers to be extended with left and right context.
    @return: DataFrame with alignments, additional columns mapped by functions, and left and right context.
    """

    # Set default values if None
    if context_tiers is None:
        context_tiers = []

    if function_dict is None:
        function_dict = {}

    # Extract unique parameters and doculects
    parameters: List[str] = sorted(set(parameter for parameter, _ in data.keys()))
    doculects: List[str] = sorted(set(doculect for _, doculect in data.keys()))

    # List to hold individual dataframes for each parameter
    dfs = []

    # Iterate through parameters and build individual dataframes
    for parameter in parameters:
        # Retrieve alignments for each doculect
        alignments = [
            data.get((parameter, doculect), [np.nan]) for doculect in doculects
        ]
        alm_length = max(len(align) for align in alignments)

        # Initialize DataFrame for this parameter
        df_parameter = pd.DataFrame(columns=["index", "rindex"] + doculects)

        # Iterate through alignment positions and fill DataFrame
        for i in range(alm_length):
            row_data = [i + 1, alm_length - i] + [
                align[i] if i < len(align) else np.nan for align in alignments
            ]
            row_df = pd.DataFrame([row_data], columns=["index", "rindex"] + doculects)
            df_parameter = pd.concat([df_parameter, row_df], ignore_index=True)

        # Apply functions to phoneme columns to create additional tiers
        for doculect in doculects:
            for tier, func in function_dict.items():
                df_parameter[f"{doculect}_{tier}"] = df_parameter[doculect].apply(func)

        # Add left and right contexts for specified tiers
        for tier in doculects + [
            f"{doculect}_{t}" for doculect in doculects for t in context_tiers
        ]:
            for l in range(1, left + 1):
                df_parameter[tier + f"_left_{l}"] = ["∅"] * l + df_parameter[
                    tier
                ].tolist()[:-l]
            for r in range(1, right + 1):
                df_parameter[tier + f"_right_{r}"] = (
                    df_parameter[tier].tolist()[r:] + ["∅"] * r
                )

        # Add ID column
        df_parameter["ID"] = [f"{parameter}_{i+1}" for i in range(alm_length)]

        # Add this subdataframe to the list
        dfs.append(df_parameter)

    # Concatenate the subdataframes into the final DataFrame
    df = pd.concat(dfs)

    return df

def analyze_multitiers(df, target_doculect, handle_nans='drop', feature_reduction=False):
    def encode_features(X):
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le
        return X, encoders

    def decode_features(X, encoders):
        for col, le in encoders.items():
            if col in X.columns:
                X[col] = le.inverse_transform(X[col])
        return X

    # Drop rows with NaN in target doculect
    df_filtered = df.dropna(subset=[target_doculect])

    # Separate target and features
    X = df_filtered.drop(columns=[target_doculect])
    y = df_filtered[target_doculect]

    # Save IDs for later use and drop from features
    ids = X['ID']
    X = X.drop(columns=['ID'])

    # Encode categorical features
    X, encoders = encode_features(X.copy())

    # Handle NaNs
    mask = X.index
    if handle_nans == 'drop':
        mask = X.dropna().index
        X = X.loc[mask]
        y = y.loc[mask]
    elif handle_nans == 'impute':
        imp = SimpleImputer(strategy='mean')
        X = imp.fit_transform(X)

    # Optional feature reduction
    if feature_reduction:
        selector = SelectKBest(k='all')  # Set appropriate parameters
        X = selector.fit_transform(X, y)

    # List of classifiers to be trained
    classifiers = [
        ('Decision Tree', DecisionTreeClassifier())
        # Add more classifiers here
    ]

    # Create a copy of the original dataframe for the result
    df_result = df.copy()

    # Perform training, prediction, and evaluation
    for name, clf in classifiers:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        # Decode predictions
        predictions_df = pd.DataFrame({'ID': ids[mask], 'prediction': predictions})
        predictions_df = decode_features(predictions_df, encoders)

        # Add predictions to the result dataframe using the ID field
        prediction_column_name = f"{target_doculect}_{name.replace(' ', '_').lower()}"
        for i, row in predictions_df.iterrows():
            df_result.loc[df_result['ID'] == row['ID'], prediction_column_name] = row['prediction']

    return df_result

def test():
    data = {
        ("ASH", "German"): ["a", "ʃ", "ɛ"],
        ("ASH", "English"): ["æ", "ʃ", "-"],
        ("ASH", "Dutch"): ["ɑ", "s", "-"],
        ("BITE", "German"): ["b", "ai", "s", "ə", "n"],
        ("BITE", "English"): ["b", "ai", "t", "-", "-"],
        ("BITE", "Dutch"): ["b", "ɛi", "t", "ə", "-"],
        ("BELLY", "German"): ["b", "au", "x"],
        ("BELLY", "Dutch"): ["b", "œi", "k"],
    }

    data = extra.read_data(
        "resources/germanic.tsv",
        concept_col="PARAMETER",
        doculect_col="DOCULECT",
        alignment_col="ALIGNMENT",
    )

    prosody_func, sca_func, dolgopolsky_func = extra.create_sound_functions()

    function_dict = {
        "prosody": prosody_func,
        "sca": sca_func,
        "dolgopolsky": dolgopolsky_func,
    }
    tiers = ["sca"]
    df = build_multitiers(
        data, left=2, right=2, function_dict=function_dict, context_tiers=tiers
    )
    print(df)

    result = analyze_multitiers(df, "English")

    # Write the result to a tabular file
    result.to_csv("temp.tsv", sep="\t", index=False)

if __name__ == "__main__":
    test()
