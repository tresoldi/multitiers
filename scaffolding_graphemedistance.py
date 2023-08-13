import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import product
from scipy.spatial import distance


def build_distance_dictionary(df):
    # 1. Read data from the sounds.csv file
    graphemes = df["GRAPHEME"].values
    descriptions = df["DESCRIPTION"].values

    # Convert descriptions to binary feature vectors
    all_features = set()
    for desc in descriptions:
        all_features.update(desc.split())

    feature_vectors = []
    for desc in descriptions:
        features = desc.split()
        vector = [1 if feature in features else 0 for feature in all_features]
        feature_vectors.append(vector)

    # 2. Perform feature reduction using PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Read the "SCA" column
    sca_values = df["SCA"].values

    # Convert unique SCA values to distinct colors
    unique_sca_values = np.unique(sca_values)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_sca_values)))
    color_map = dict(zip(unique_sca_values, colors))

    # Map each SCA value to its corresponding color
    sca_colors = [color_map[val] for val in sca_values]

    # 3. Plot the PCA
    plt.figure(figsize=(20, 18))  # Increase the figure size
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=sca_colors)

    # Set the font to 'DejaVu Sans' to support a wide range of Unicode characters
    plt.rcParams["font.family"] = "DejaVu Sans"

    for i, grapheme in enumerate(graphemes):
        plt.annotate(grapheme, (pca_result[i, 0], pca_result[i, 1]))

    # Create a legend for the SCA values
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=sca,
            markersize=10,
            markerfacecolor=color_map[sca],
        )
        for sca in unique_sca_values
    ]
    plt.legend(handles=handles, title="SCA")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of Phonemes")
    plt.savefig("pca_phonemes.png", bbox_inches="tight")
    plt.close()

    # 4. Calculate the Euclidean distance between all permutations of graphemes
    distance_dict = {}
    for g1, g2 in product(
        graphemes, repeat=2
    ):  # Use product to get all combinations including (g1, g1)
        idx1 = np.where(graphemes == g1)[0][0]
        idx2 = np.where(graphemes == g2)[0][0]
        dist = distance.euclidean(pca_result[idx1], pca_result[idx2])
        distance_dict[(g1, g2)] = dist

    # Normalize the distances relative to the maximum distance
    max_distance = max(distance_dict.values())
    for key, value in distance_dict.items():
        distance_dict[key] = value / max_distance

    # Save the distance dictionary to disk
    with open("distance_dictionary.txt", "w", encoding="utf-8") as f:
        for (g1, g2), dist in sorted(distance_dict.items()):  # Sort by graphemes
            f.write(
                f"{g1},{g2},{dist:.6f}\n"
            )  # Format the distance with six digits after the separator

    return distance_dict


if __name__ == "__main__":
    df = pd.read_csv("sounds.csv")
    build_distance_dictionary(df)
