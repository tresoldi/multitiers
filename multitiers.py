from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Optional, Callable, Tuple
import joblib
import numpy as np
import pandas as pd
import os


import extra
import evaluate


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
        df_parameter = pd.DataFrame(
            columns=["index", "rindex"]
            + [f"{doc}.{tier}" for doc in doculects for tier in ["phoneme"]]
        )

        # Iterate through alignment positions and fill DataFrame
        for i in range(alm_length):
            row_data = [i + 1, alm_length - i] + [
                align[i] if i < len(align) else np.nan for align in alignments
            ]
            row_df = pd.DataFrame(
                [row_data],
                columns=["index", "rindex"]
                + [f"{doc}.{tier}" for doc in doculects for tier in ["phoneme"]],
            )
            df_parameter = pd.concat([df_parameter, row_df], ignore_index=True)

        # Apply functions to phoneme columns to create additional tiers
        for doculect in doculects:
            for tier, func in function_dict.items():
                df_parameter[f"{doculect}.{tier}"] = df_parameter[
                    f"{doculect}.phoneme"
                ].apply(func)

        # Add left and right contexts for specified tiers
        for tier in [
            f"{doc}.{t}" for doc in doculects for t in ["phoneme"] + context_tiers
        ]:
            for l in range(1, left + 1):
                df_parameter[f"{tier}_left_{l}"] = ["∅"] * l + df_parameter[
                    tier
                ].tolist()[:-l]
            for r in range(1, right + 1):
                df_parameter[f"{tier}_right_{r}"] = (
                    df_parameter[tier].tolist()[r:] + ["∅"] * r
                )

        # Add ID column
        df_parameter["ID"] = [f"{parameter}_{i+1}" for i in range(alm_length)]

        # Add this subdataframe to the list
        dfs.append(df_parameter)

    # Concatenate the subdataframes into the final DataFrame
    df = pd.concat(dfs)

    return df


def encode_features(X):
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    return X, encoders


def decode_features(X, encoders):
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.inverse_transform(X[col])
    return X


def train_classifiers(X, y, dataset_name, output_dir="trained_classifiers"):
    """
    Train different classifier pipelines and save them to disk.
    """
    # Define classifiers and their names
    classifiers = {
        "simpleimput_dt3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("classifier", DecisionTreeClassifier(max_depth=3)),
            ]
        ),
        "dropna_rf": Pipeline(
            [
                ("imputer", "passthrough"),
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier()),
            ]
        ),
        "simpleimput_lr": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression()),
            ]
        ),
    }

    # Create local copies of X and y
    X_local = X.copy().drop(columns=["ID"])
    y_local = y.copy()

    # Filter out rows where the target variable contains NaN values
    mask = ~y_local.isna()
    X_filtered = X_local[mask]
    y_filtered = y_local[mask]

    # Encode features
    X_encoded, encoders = encode_features(X_filtered.copy())

    # Train classifiers and save them to disk
    for name, clf in classifiers.items():
        clf.fit(X_encoded, y_filtered)
        filename = f"{dataset_name}.{name}.pkl"
        joblib.dump((clf, encoders), os.path.join(output_dir, filename))


def apply_classifiers(
    df, X, target_doculect, dataset_name, input_dir="trained_classifiers"
):
    """
    Load trained classifiers from disk and apply them to the data.
    """
    # List available classifiers
    classifier_names = [
        name
        for name in os.listdir(input_dir)
        if name.startswith(dataset_name) and name.endswith(".pkl")
    ]

    # Drop the "ID" column from X
    X_local = X.copy().drop(columns=["ID"])

    # Apply each classifier to the data
    for name in classifier_names:
        pipeline_name = name.split(".")[1].rsplit(".pkl", 1)[0]
        clf, encoders = joblib.load(os.path.join(input_dir, name))

        # Encode features
        X_encoded = encode_features(X_local.copy())[0]

        predictions = clf.predict(X_encoded)

        # Decode predictions
        predictions_df = pd.DataFrame({"ID": df["ID"], "prediction": predictions})
        predictions_df = decode_features(predictions_df, encoders)

        # Add predictions to the result dataframe using the ID field
        prediction_column_name = f"{target_doculect}.prediction.{pipeline_name}"
        df[prediction_column_name] = predictions_df["prediction"]

    return df


def process_and_train(data, dataset_name):
    # Create sound functions
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

    # Drop English related tiers for training
    X = df.drop(columns=[col for col in df.columns if "English." in col])
    y = df["English.phoneme"]

    # Train classifiers
    train_classifiers(X, y, dataset_name)

    # Apply classifiers
    df_predictions = apply_classifiers(df, X, "English", dataset_name)

    # Evaluate classifiers
    evaluate.evaluate_classifiers(df_predictions, "English", dataset_name)

    # Write the result to a tabular file
    df_predictions.to_csv(f"{dataset_name}_results.tsv", sep="\t", index=False)


def test_toy_dataset():
    # Toy dataset
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

    # Process and train classifiers
    process_and_train(data, "toy")


def test_germanic_dataset():
    # Read data from germanic.tsv
    data = extra.read_data(
        "resources/aligned_germanic.tsv",
        concept_col="PARAMETER",
        doculect_col="DOCULECT",
        alignment_col="ALIGNMENT",
    )

    # Process and train classifiers
    process_and_train(data, "germanic")


if __name__ == "__main__":
    test_toy_dataset()
    test_germanic_dataset()
