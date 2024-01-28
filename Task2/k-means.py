import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
import umap
import os

# Doc vectors of bert on source.
BERT_SOURCE_A_VECTORS = "./data/bert_on_source/bert_source_A_docvec.xlsx"
BERT_SOURCE_B_VECTORS = "./data/bert_on_source/bert_source_B_docvec.xlsx"
BERT_SOURCE_C_VECTORS = "./data/bert_on_source/bert_source_C_docvec.xlsx"
BERT_SOURCE_OUTPUT_FOLDER = "./output/K-means/Bert_On_Source_Groups/"

# Doc vectors of d2v on source.
D2V_SOURCE_A_VECTORS = "./data/d2v_on_source/d2v_source_A_docvec.xlsx"
D2V_SOURCE_B_VECTORS = "./data/d2v_on_source/d2v_source_B_docvec.xlsx"
D2V_SOURCE_C_VECTORS = "./data/d2v_on_source/d2v_source_C_docvec.xlsx"
D2V_SOURCE_OUTPUT_FOLDER = "./output/K-means/D2V_On_Source_Groups/"

# Doc vectors of tfidf on lemots.
TFIDF_LEMOTS_A_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"
TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/K-means/TFIDF_On_Lemots_Groups/"

# Doc vectors of tfidf on words.
TFIDF_WORDS_A_VECTORS = "./data/tfidf_on_words/tfidf_words_A_docvec.xlsx"
TFIDF_WORDS_B_VECTORS = "./data/tfidf_on_words/tfidf_words_B_docvec.xlsx"
TFIDF_WORDS_C_VECTORS = "./data/tfidf_on_words/tfidf_words_C_docvec.xlsx"
TFIDF_WORDS_OUTPUT_FOLDER = "./output/K-means/TFIDF_On_Words_Groups/"

# Doc vectors of w2v on lemots.
W2V_LEMOTS_A_VECTORS = "./data/w2v_on_lemots/w2v_lemots_A_docvec.xlsx"
W2V_LEMOTS_B_VECTORS = "./data/w2v_on_lemots/w2v_lemots_B_docvec.xlsx"
W2V_LEMOTS_C_VECTORS = "./data/w2v_on_lemots/w2v_lemots_C_docvec.xlsx"
W2V_LEMOTS_OUTPUT_FOLDER = "./output/K-means/W2V_On_Lemots_Groups/"

# Doc vectors of w2v on words.
W2V_WORDS_A_VECTORS = "./data/w2v_on_words/w2v_words_A_docvec.xlsx"
W2V_WORDS_B_VECTORS = "./data/w2v_on_words/w2v_words_B_docvec.xlsx"
W2V_WORDS_C_VECTORS = "./data/w2v_on_words/w2v_words_C_docvec.xlsx"
W2V_WORDS_OUTPUT_FOLDER = "./output/K-means/W2V_On_Words_Groups/"


def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]


def k_means_clustering(vectors1, vectors2):
    all_vectors = np.vstack((vectors1, vectors2))
    kmeans = KMeans(n_clusters=2, random_state=42).fit(all_vectors)
    labels1 = kmeans.predict(vectors1)
    labels2 = kmeans.predict(vectors2)
    return labels1, labels2


def evaluate_clustering(label1_true, label2_true, labels_pred_cluster1, labels_pred_cluster2):
    """
    Evaluate clustering performance.

    Parameters:
    - label1_true: True labels for cluster 1
    - label2_true: True labels for cluster 2
    - labels_pred_cluster1: Predicted labels for cluster 1
    - labels_pred_cluster2: Predicted labels for cluster 2

    Returns:
    - total_precision: Overall precision
    - total_recall: Overall recall
    - total_f1: Overall F1 score
    - total_accuracy: Overall accuracy
    """

    # Combine true labels from both clusters
    labels_true = np.concatenate((label1_true, label2_true))

    # Combine predicted labels from both clusters
    labels_pred = np.concatenate((labels_pred_cluster1, labels_pred_cluster2))

    # Evaluate overall performance
    total_precision = precision_score(labels_true, labels_pred)
    total_recall = recall_score(labels_true, labels_pred)
    total_f1 = f1_score(labels_true, labels_pred)
    total_accuracy = accuracy_score(labels_true, labels_pred)

    return total_precision, total_recall, total_f1, total_accuracy


def plot_TSNE_graph(vectors1, vectors2, labels1, labels2, label1_name, label2_name, title, save_path):
    """
    Visualize clusters using SNE-T.

    Parameters:
    - vectors1: List of vectors with label 1.
    - vectors2: List of vectors with label 2.
    - labels1: List of predictions on vectors1, 0 - predict Label 1, 1 - predict Label 2.
    - labels2: List of predictions on vectors2, 0 - predict Label 1, 1 - predict Label 2.
    - label1_name: A, B, or C.
    - label2_name: A, B, or C.
    - title: Title for the plot.
    - save_path: Path to save the plot.

    Output:
    Plot of two clusters (blue & red).
    """

    # Combine vectors and labels
    all_vectors = np.vstack((vectors1, vectors2))

    # Apply t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=42)
    transformed_vectors = tsne.fit_transform(all_vectors)

    # Separate transformed vectors back to vectors1 and vectors2
    transformed_vectors1 = transformed_vectors[:len(vectors1)]
    transformed_vectors2 = transformed_vectors[len(vectors1):]

    # Plot scatter plot
    plt.figure(figsize=(8, 6))

    # Plot vectors with label 1 in blue
    plt.scatter(transformed_vectors1[labels1 == 0, 0], transformed_vectors1[labels1 == 0, 1],
                color='blue', label=f'Label {label1_name} (Predicted)')

    # Plot vectors with label 2 in red
    plt.scatter(transformed_vectors2[labels2 == 1, 0], transformed_vectors2[labels2 == 1, 1],
                color='red', label=f'Label {label2_name} (Predicted)')

    plt.title(title)
    plt.legend()
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path)
    print(f"t-SNE plot saved in: {save_path}")


def create_output_for_groups(group1_path, group2_path, group3_path, output_folder):
    print("Clustering Matrices using K-MEANS...\n")

    # Get vectors from data
    print("Retrieving vectors from data...")
    vectors_A = get_vectors_from(group1_path)
    print("Group A Retrieved.")
    vectors_B = get_vectors_from(group2_path)
    print("Group B Retrieved.")
    vectors_C = get_vectors_from(group3_path)
    print("Group C Retrieved.\n")

    # Create clusters and evaluate for (A, B), (A, C), (B, C) pairs
    pairs = [(vectors_A, vectors_B), (vectors_A, vectors_C), (vectors_B, vectors_C)]
    pair_describe = ""

    for i, (vectors1, vectors2) in enumerate(pairs):
        if i == 0:
            pair_describe = "A&B"
        elif i == 1:
            pair_describe = "A&C"
        elif i == 2:
            pair_describe = "B&C"

        print(f"Working on pairs: {pair_describe} ...")

        # Create clusters using K-Means
        vectors1, vectors2 = np.array(vectors1), np.array(vectors2)  # Convert to numpy array
        labels1, labels2 = k_means_clustering(vectors1, vectors2)
        print("K-means clustering finish.")

        # Evaluate clusters
        print("Evaluate results:")
        precision, recall, f1, accuracy = evaluate_clustering(np.zeros(len(vectors1)), np.ones(len(vectors2)), labels1,
                                                              labels2)

        # Create output for JSON
        result = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        # Save the result into json file.
        json_save_path = os.path.join(output_folder, f'{pair_describe}.json')
        with open(json_save_path, 'w') as json_file:
            json.dump(result, json_file)
        print(f"Evaluate result saved in: {json_save_path}")

        # Plot clusters using t-SNE
        plot_TSNE_graph(vectors1, vectors2, labels1, labels2, pair_describe[0], pair_describe[2],
                        f"t-SNE {pair_describe}",
                        os.path.join(output_folder, f'tsne_{pair_describe}.png'))

        print("\n")


def main():
    # K-means for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(BERT_SOURCE_A_VECTORS, BERT_SOURCE_B_VECTORS, BERT_SOURCE_C_VECTORS,
                             output_folder=BERT_SOURCE_OUTPUT_FOLDER)

    # K-means for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(D2V_SOURCE_A_VECTORS, D2V_SOURCE_B_VECTORS, D2V_SOURCE_C_VECTORS,
                             output_folder=D2V_SOURCE_OUTPUT_FOLDER)

    # K-means for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(TFIDF_LEMOTS_A_VECTORS, TFIDF_LEMOTS_B_VECTORS, TFIDF_LEMOTS_C_VECTORS,
                             output_folder=TFIDF_LEMOTS_OUTPUT_FOLDER)

    # K-means for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(TFIDF_WORDS_A_VECTORS, TFIDF_WORDS_B_VECTORS, TFIDF_WORDS_C_VECTORS,
                             output_folder=TFIDF_WORDS_OUTPUT_FOLDER)

    # K-means for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(W2V_LEMOTS_A_VECTORS, W2V_LEMOTS_B_VECTORS, W2V_LEMOTS_C_VECTORS,
                             output_folder=W2V_LEMOTS_OUTPUT_FOLDER)

    # K-means for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(W2V_WORDS_A_VECTORS, W2V_WORDS_B_VECTORS, W2V_WORDS_C_VECTORS,
                             output_folder=W2V_WORDS_OUTPUT_FOLDER)



if __name__ == "__main__":
    main()
