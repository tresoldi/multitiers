import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def format_percentage_label(val):
    """
    Format the percentage label for the heatmap.
    """
    if val == 1:
        return "00"
    elif val == 0:
        return "0"
    else:
        return str(int(val * 100))


def plot_confusion_matrices(
    y_true_filtered, y_pred_filtered, labels, col, dataset_name, output_dir
):
    abbreviated_labels = [label[:2] for label in labels]
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
    cm_percentage = cm / cm.sum(axis=1, keepdims=True)

    # Plot absolute confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=abbreviated_labels,
        yticklabels=abbreviated_labels,
        annot_kws={"size": 10},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Absolute Confusion Matrix for {col}")
    plt.savefig(f"{output_dir}/{dataset_name}.{col}.confusion_matrix_absolute.png")
    plt.close()

    # Plot percentage confusion matrix
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        cm_percentage,
        annot=True,
        fmt="",
        cmap="Blues",
        xticklabels=abbreviated_labels,
        yticklabels=abbreviated_labels,
        annot_kws={"size": 10},
        cbar=False,
    )
    for t in ax.texts:
        t.set_text(format_percentage_label(float(t.get_text())))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Percentage Confusion Matrix for {col}")
    plt.savefig(f"{output_dir}/{dataset_name}.{col}.confusion_matrix_percentage.png")
    plt.close()


def generate_classification_report(
    y_true_filtered, y_pred_filtered, labels, col, dataset_name, output_dir
):
    report = classification_report(
        y_true_filtered,
        y_pred_filtered,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Extracting top 3 predictions for each phoneme
    df_predictions = pd.DataFrame(
        {"True": y_true_filtered, "Predicted": y_pred_filtered}
    )
    top_predictions = {}
    for phoneme in labels:
        phoneme_total = len(df_predictions[df_predictions["True"] == phoneme])
        top_3 = (
            df_predictions[df_predictions["True"] == phoneme]["Predicted"]
            .value_counts()
            .head(3)
        )
        top_predictions[phoneme] = [
            (pred, count, count / phoneme_total)
            for pred, count in zip(top_3.index, top_3)
        ]

    # Building the table
    table_data = []
    for phoneme in labels:
        row = [phoneme]
        metrics = report[phoneme]
        for metric, value in metrics.items():
            if metric != "support":  # Exclude 'support' metric
                row.append(f"{value:.2f}")
        top_3_str = ", ".join(
            [
                f"{pred[0]} ({pred[1]:d}, {pred[2]:.2f})"
                for pred in top_predictions[phoneme]
            ]
        )
        row.append(top_3_str)
        table_data.append(row)

    headers = ["Phoneme", "Precision", "Recall", "F1-Score", "Top 3 Predictions"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")

    with open(
        f"{output_dir}/{dataset_name}.{col}.classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"Classification Report for {col}:\n")
        f.write("\n")
        f.write(table)


def evaluate_classifiers(
    df, target_doculect, dataset_name, output_dir="evaluation_results"
):
    """
    Evaluate the performance of the classifiers.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract true values
    y_true = df[f"{target_doculect}.phoneme"]

    # Extract predicted values
    classifier_columns = [
        col
        for col in df.columns
        if col.startswith(
            f"{target_doculect}.prediction."
        )  # Only include classifiers for phonemes
    ]

    for col in classifier_columns:
        y_pred = df[col]

        # Filter out rows where y_true has NaNs
        mask = ~y_true.isna()
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]

        # Convert y_true_filtered and y_pred_filtered to string type
        y_true_filtered = y_true_filtered.astype(str)
        y_pred_filtered = y_pred_filtered.astype(str)

        # Convert labels to string type
        labels = sorted(
            [
                str(label)
                for label in list(
                    set(y_true_filtered.unique()) | set(y_pred_filtered.unique())
                )
            ]
        )

        # Call the functions to plot confusion matrices and generate classification report
        plot_confusion_matrices(
            y_true_filtered, y_pred_filtered, labels, col, dataset_name, output_dir
        )
        generate_classification_report(
            y_true_filtered, y_pred_filtered, labels, col, dataset_name, output_dir
        )


def simple_alignment_score(L1, L2):
    """Return the ratio of matching phonemes between two lists."""
    if len(L1) != len(L2):
        raise ValueError("Both lists must have the same length.")

    matches = sum(1 for a, b in zip(L1, L2) if a == b)
    return matches / len(L1)


def sound_class_alignment_score(L1, L2, sound_class_dict):
    """Map phonemes to sound classes and return the ratio of matching sound classes."""
    if len(L1) != len(L2):
        raise ValueError("Both lists must have the same length.")

    L1_classes = [sound_class_dict.get(phoneme, phoneme) for phoneme in L1]
    L2_classes = [sound_class_dict.get(phoneme, phoneme) for phoneme in L2]

    matches = sum(1 for a, b in zip(L1_classes, L2_classes) if a == b)
    return matches / len(L1)


def distance_based_alignment_score(L1, L2, distance_dict=None, penalty=1.5):
    """Return the alignment score based on a distance dictionary."""
    if len(L1) != len(L2):
        raise ValueError("Both lists must have the same length.")

    # Default distance function
    if distance_dict is None:

        def distance(a, b):
            if a == b:
                return 0
            elif set(a).issubset(set(b)) or set(b).issubset(set(a)):
                return 0.5
            else:
                return 1

    else:

        def distance(a, b):
            return distance_dict.get((a, b), 1)

    total_distance = sum(distance(a, b) for a, b in zip(L1, L2))
    raw_score = 1 - (total_distance / len(L1))

    # Apply penalty
    penalized_score = raw_score**penalty

    return penalized_score


import pandas as pd


def load_sound_class_dictionary(
    filename="sounds.csv", grapheme_col="GRAPHEME", sound_class_col="SCA"
):
    """
    Load a dictionary mapping graphemes to sound classes from a CSV file.

    Parameters:
    - filename: Name of the CSV file to read.
    - grapheme_col: Name of the column containing graphemes.
    - sound_class_col: Name of the column containing sound classes.

    Returns:
    - sound_class_dict: Dictionary mapping graphemes to sound classes.
    """
    df = pd.read_csv(filename, encoding="utf-8")
    sound_class_dict = dict(zip(df[grapheme_col], df[sound_class_col]))
    return sound_class_dict


def load_distance_dictionary(filename="distance_dictionary.txt"):
    """Load the distance dictionary from a file."""
    distance_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            g1, g2, dist = line.strip().split(",")
            distance_dict[(g1, g2)] = float(dist)
    return distance_dict


if __name__ == "__main__":
    # Perfect match
    L3 = ["a", "b", "c"]
    L4 = ["a", "b", "c"]

    # Partial match
    L5 = ["a", "b", "c"]
    L6 = ["a", "d", "e"]

    # No match
    L7 = ["a", "b", "c"]
    L8 = ["x", "y", "z"]

    sound_class_dict = load_sound_class_dictionary()
    distance_dict = load_distance_dictionary()

    print("Perfect Match:")
    print("Simple:", simple_alignment_score(L3, L4))
    print("Sound class:", sound_class_alignment_score(L3, L4, sound_class_dict))
    print("Default distance:", distance_based_alignment_score(L3, L4))
    print("CLTS distance:", distance_based_alignment_score(L3, L4, distance_dict))
    print()

    print("Partial Match:")
    print("Simple:", simple_alignment_score(L5, L6))
    print("Sound class:", sound_class_alignment_score(L5, L6, sound_class_dict))
    print("Default distance:", distance_based_alignment_score(L5, L6))
    print("CLTS distance:", distance_based_alignment_score(L5, L6, distance_dict))
    print()

    print("No Match:")
    print("Simple:", simple_alignment_score(L7, L8))
    print("Sound class:", sound_class_alignment_score(L7, L8, sound_class_dict))
    print("Default distance:", distance_based_alignment_score(L7, L8))
    print("CLTS distance:", distance_based_alignment_score(L7, L8, distance_dict))
