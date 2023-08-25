# Import Python modules
from typing import Dict, List, Callable, Optional, Tuple
import os
import logging

# Import third-party modules
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import local modules
import extra
import evaluate

# Constants
EMPTY_SYMBOL = "∅"

# Global default classifiers dictionary
DEFAULT_CLASSIFIERS = {
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
    # "simpleimput_lr": Pipeline(
    #     [
    #         ("imputer", SimpleImputer(strategy="mean")),
    #         ("scaler", StandardScaler()),
    #         ("classifier", LogisticRegression()),
    #     ]
    # ),
    # # Two additional classifier pipelines
    # "simpleimput_svm": Pipeline(
    #     [
    #         ("imputer", SimpleImputer(strategy="mean")),
    #         ("scaler", StandardScaler()),
    #         ("classifier", SVC()),
    #     ]
    # ),
    # "dropna_knn": Pipeline(
    #     [
    #         ("imputer", "passthrough"),
    #         ("scaler", StandardScaler()),
    #         ("classifier", KNeighborsClassifier()),
    #     ]
    # ),
}


def build_multitiers(
    data: Dict[Tuple[str, str], List[str]],
    left: int = 0,
    right: int = 0,
    function_dict: Dict[str, Callable] = {},
    context_tiers: List[str] = [],
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

        # Iterate through alignment positions and fill the DataFrame
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
                df_parameter[f"{tier}_left_{l}"] = [EMPTY_SYMBOL] * l + df_parameter[
                    tier
                ].tolist()[:-l]
            for r in range(1, right + 1):
                df_parameter[f"{tier}_right_{r}"] = (
                    df_parameter[tier].tolist()[r:] + [EMPTY_SYMBOL] * r
                )

        # Add ID column
        df_parameter["ID"] = [f"{parameter}_{i+1}" for i in range(alm_length)]

        # Add this subdataframe to the list
        dfs.append(df_parameter)

    # Concatenate the subdataframes into the final DataFrame
    df = pd.concat(dfs)

    # Reset the index to ensure unique indices for each row
    df.reset_index(drop=True, inplace=True)

    return df


def encode_features(X: DataFrame) -> Tuple[DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode the features in the DataFrame.

    @param X: DataFrame containing the feature data.

    @return: Tuple containing the encoded DataFrame and a dictionary of encoders.
    """
    encoders = {}
    X_encoded = X.copy()

    for col in X.columns:
        # If the column is of object type (string labels), encode it
        if X[col].dtype == "object":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        # Else, if the column has NaN values, impute them
        elif X[col].isnull().any():
            if X[col].dtype in ["int64", "float64"]:
                imputer = SimpleImputer(strategy="mean")
            else:
                imputer = SimpleImputer(strategy="most_frequent")
            X_encoded[col] = imputer.fit_transform(X[col].values.reshape(-1, 1))

    return X_encoded, encoders


def decode_features(X: DataFrame, encoders: Dict[str, LabelEncoder]) -> DataFrame:
    """
    Decode previously encoded features in the DataFrame using provided encoders.

    @param X: DataFrame with features to be decoded.
    @param encoders: Dictionary of encoders used for encoding.
    @return: DataFrame with decoded features.
    """
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.inverse_transform(X[col])
    return X


def train_classifiers(
    X: DataFrame,
    y: DataFrame,
    dataset_name: str,
    doculect: str,
    classifiers=DEFAULT_CLASSIFIERS,
    output_dir: Optional[str] = None,
) -> Dict[str, Tuple[Pipeline, Dict[str, LabelEncoder]]]:
    """
    Train different classifier pipelines and optionally save them to disk.

    @param X: DataFrame containing the feature data.
    @param y: DataFrame containing the target variable.
    @param dataset_name: Name of the dataset, used for naming saved classifiers.
    @param doculect: Name of the doculect, used for naming saved classifiers.
    @param classifiers: Dictionary of classifier pipelines.
    @param output_dir: Optional directory where trained classifiers will be saved.
                       If None, classifiers are not saved to disk.

    @return: Dictionary containing trained classifiers and their associated encoders.
    """

    # Create local copies of X and y
    X_local = X.copy().drop(columns=["ID"])
    y_local = y.copy()

    # Filter out rows where the target variable contains NaN values
    mask = ~y_local.isna()
    X_filtered = X_local[mask]
    y_filtered = y_local[mask]

    # Encode features
    X_encoded, encoders = encode_features(X_filtered.copy())

    trained_classifiers = {}

    # Train classifiers and optionally save them to disk
    for name, clf in classifiers.items():
        clf.fit(X_encoded, y_filtered)
        trained_classifiers[name] = (clf, encoders)

        if output_dir:
            filename = f"{dataset_name}.{doculect}.{name}.pkl"
            joblib.dump((clf, encoders), os.path.join(output_dir, filename))

    return trained_classifiers


def apply_classifiers(
    df, X, target_doculect, dataset_name, input_dir=None, classifiers_dict=None
):
    """
    Load trained classifiers from disk or from a dictionary and apply them to the data.

    @param df: DataFrame containing the data.
    @param X: DataFrame containing the feature data.
    @param target_doculect: Target doculect for which predictions are made.
    @param dataset_name: Name of the dataset.
    @param input_dir: Directory from where to read serialized classifiers.
    @param classifiers_dict: Dictionary of classifiers.

    @return: DataFrame with predictions.
    """

    # Create a copy of df to avoid SettingWithCopyWarning
    df = df.copy()

    # Ensure either input_dir or classifiers_dict is provided
    if not input_dir and not classifiers_dict:
        raise ValueError("Either input_dir or classifiers_dict must be provided.")

    # Drop the "ID" column from X
    X_local = X.copy().drop(columns=["ID"])

    if input_dir:
        # List available classifiers
        classifier_names = [
            name
            for name in os.listdir(input_dir)
            if name.startswith(f"{dataset_name}.{target_doculect}")
            and name.endswith(".pkl")
        ]

        # Apply each classifier to the data
        for name in classifier_names:
            # Extract the classifier name from the filename
            pipeline_name = name.split(".")[2]  # Adjusted this line

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

    elif classifiers_dict:
        # Apply each classifier from the dictionary to the data
        for pipeline_name, (clf, encoders) in classifiers_dict.items():
            # Encode features
            X_encoded = encode_features(X_local.copy())[0]

            predictions = clf.predict(X_encoded)

            # Decode predictions
            predictions_df = pd.DataFrame({"ID": df["ID"], "prediction": predictions})
            predictions_df = decode_features(predictions_df, encoders)

            # Add predictions to the result dataframe using the ID field
            prediction_column_name = f"{target_doculect}.prediction.{pipeline_name}"
            df.loc[:, prediction_column_name] = predictions_df["prediction"].values

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

    # Extract unique doculects from the dataframe
    doculects = sorted(
        set(doculect.split(".")[0] for doculect in df.columns if "." in doculect)
    )

    for doculect in doculects:
        # Drop the current doculect related tiers for training
        X = df.drop(columns=[col for col in df.columns if f"{doculect}." in col])
        y = df[f"{doculect}.phoneme"]

        # Train classifiers
        train_classifiers(
            X, y, dataset_name, doculect, output_dir="trained_classifiers"
        )

        # Apply classifiers
        df_predictions = apply_classifiers(
            df, X, doculect, dataset_name, input_dir="trained_classifiers"
        )

        # Evaluate classifiers
        evaluate.evaluate_classifiers(df_predictions, doculect, dataset_name)

    # Write the result to a tabular file
    df_predictions.to_csv(f"{dataset_name}_results.tsv", sep="\t", index=False)


def leave_one_out_evaluation(data, sound_class_dict, distance_dict):
    """
    Perform a leave-one-out evaluation on the data.

    @param data: Data in the format of `process_and_train()`.
    @param sound_class_dict: Dictionary mapping phonemes to sound classes.
    @param distance_dict: Dictionary containing distances between phonemes.

    @return: Aggregated scores DataFrame by doculect and pipeline.
    """

    logging.info("Starting leave-one-out evaluation...")

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

    # Extract unique doculects from the dataframe
    doculects = sorted(
        set(doculect.split(".")[0] for doculect in df.columns if "." in doculect)
    )

    # Extract unique words from the "ID" column
    unique_words = df["ID"].str.extract(r"(.+)_(\d+)")[0].unique()

    # Placeholder for collecting detailed scores by doculect and pipeline
    doculect_pipeline_scores = []

    for doculect in doculects:
        logging.info(f"Processing doculect: {doculect}")
        for word in tqdm(
            unique_words, desc="Processing words for {doculect}", leave=False
        ):
            logging.debug(f"Evaluating word: {word}")

            # Drop rows corresponding to the current word for the leave-one-out evaluation
            mask = df["ID"].str.startswith(word)
            df_dropped = df[mask]
            df_remaining = df[~mask]

            # Prepare data for training
            X = df_remaining.drop(
                columns=[col for col in df_remaining.columns if f"{doculect}." in col]
            )
            y = df_remaining[f"{doculect}.phoneme"]

            # Train classifiers
            classifiers_dict = train_classifiers(X, y, "leave_one_out", doculect)

            for pipeline_name in classifiers_dict:
                logging.debug(f"Applying pipeline: {pipeline_name}")

                # Prepare data for predictions on dropped rows
                X_dropped = df_dropped.drop(
                    columns=[col for col in df_dropped.columns if f"{doculect}." in col]
                )

                # Apply classifiers only on dropped rows
                df_predictions = apply_classifiers(
                    df_dropped,
                    X_dropped,
                    doculect,
                    "leave_one_out",
                    classifiers_dict={pipeline_name: classifiers_dict[pipeline_name]},
                )

                # Collect scores for the current word
                (
                    simple_score,
                    sound_class_score,
                    distance_score,
                ) = evaluate.calculate_word_scores_single_word(
                    df_predictions,
                    doculect,
                    pipeline_name,
                    sound_class_dict=sound_class_dict,
                    distance_dict=distance_dict,
                )

                doculect_pipeline_scores.append(
                    {
                        "Doculect": doculect,
                        "Pipeline": pipeline_name,
                        "Word_ID": word,
                        "Simple_Score": simple_score,
                        "Sound_Class_Score": sound_class_score,
                        "Distance_Score": distance_score,
                    }
                )

    # Convert doculect_pipeline_scores to a DataFrame
    scores_df = pd.DataFrame(doculect_pipeline_scores)

    # Aggregate the scores by doculect and pipeline
    aggregated_scores_df = (
        scores_df.groupby(["Doculect", "Pipeline"])
        .agg(
            {
                "Word_ID": "count",
                "Simple_Score": ["mean", "min", "max", "std"],
                "Sound_Class_Score": ["mean", "min", "max", "std"],
                "Distance_Score": ["mean", "min", "max", "std"],
            }
        )
        .reset_index()
    )

    # Flatten the multi-level column names
    aggregated_scores_df.columns = [
        "_".join(col).strip() for col in aggregated_scores_df.columns.values
    ]

    logging.info("Leave-one-out evaluation completed.")

    return aggregated_scores_df


def test_toy_dataset():
    # Toy dataset
    data = {
        ("ASH", "German"): ["a", "ʃ", "ɛ"],
        ("ASH", "English"): ["æ", "ʃ", "-"],
        ("ASH", "Dutch"): ["ɑ", "s", "-"],
        ("BITE", "German"): ["b", "a", "i", "s", "ə", "n"],
        ("BITE", "English"): ["b", "a", "i", "t", "-", "-"],
        ("BITE", "Dutch"): ["b", "ɛ", "i", "t", "ə", "-"],
        ("BELLY", "German"): ["b", "a", "u", "x"],
        ("BELLY", "Dutch"): ["b", "œ", "i", "k"],
    }

    # Process and train classifiers
    process_and_train(data, "toy")

    # Run experiments with leave-one-out evaluation
    sound_class_dict = evaluate.load_sound_class_dictionary()
    distance_dict = evaluate.load_distance_dictionary()
    leave_one_out_evaluation(data, sound_class_dict, distance_dict)


def test_germanic_dataset():
    # Read data from germanic.tsv
    data = extra.read_data(
        "resources/aligned_germanic.tsv",
        concept_col="PARAMETER",
        doculect_col="DOCULECT",
        alignment_col="ALIGNMENT",
    )

    # Filter data to only include rows where the doculect is "English", "German", or "Dutch"
    filtered_data = {
        key: value
        for key, value in data.items()
        if key[1] in ["English", "German", "Dutch"]
    }

    # Process and train classifiers
    process_and_train(filtered_data, "germanic")

    # Run experiments with leave-one-out evaluation
    sound_class_dict = evaluate.load_sound_class_dictionary()
    distance_dict = evaluate.load_distance_dictionary()
    oneout = leave_one_out_evaluation(data, sound_class_dict, distance_dict)
    oneout.to_csv(
        f"germanic.oneout.csv",
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_toy_dataset()
    # test_germanic_dataset()
