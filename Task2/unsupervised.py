from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import json
import os

ALG = {"K-means": 1, "DBSCAN": 2, "Mixture of Gaussian": 3, "ANN": 4}

# Data Files

# Doc vectors of bert on source.
BERT_SOURCE_A_VECTORS = "./data/bert_on_source/bert_source_A_docvec.xlsx"
BERT_SOURCE_B_VECTORS = "./data/bert_on_source/bert_source_B_docvec.xlsx"
BERT_SOURCE_C_VECTORS = "./data/bert_on_source/bert_source_C_docvec.xlsx"

# Doc vectors of d2v on source.
D2V_SOURCE_A_VECTORS = "./data/d2v_on_source/d2v_source_A_docvec.xlsx"
D2V_SOURCE_B_VECTORS = "./data/d2v_on_source/d2v_source_B_docvec.xlsx"
D2V_SOURCE_C_VECTORS = "./data/d2v_on_source/d2v_source_C_docvec.xlsx"

# Doc vectors of tfidf on lemots.
TFIDF_LEMOTS_A_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"

# Doc vectors of tfidf on words.
TFIDF_WORDS_A_VECTORS = "./data/tfidf_on_words/tfidf_words_A_docvec.xlsx"
TFIDF_WORDS_B_VECTORS = "./data/tfidf_on_words/tfidf_words_B_docvec.xlsx"
TFIDF_WORDS_C_VECTORS = "./data/tfidf_on_words/tfidf_words_C_docvec.xlsx"

# Doc vectors of w2v on lemots.
W2V_LEMOTS_A_VECTORS = "./data/w2v_on_lemots/w2v_lemots_A_docvec.xlsx"
W2V_LEMOTS_B_VECTORS = "./data/w2v_on_lemots/w2v_lemots_B_docvec.xlsx"
W2V_LEMOTS_C_VECTORS = "./data/w2v_on_lemots/w2v_lemots_C_docvec.xlsx"

# Doc vectors of w2v on words.
W2V_WORDS_A_VECTORS = "./data/w2v_on_words/w2v_words_A_docvec.xlsx"
W2V_WORDS_B_VECTORS = "./data/w2v_on_words/w2v_words_B_docvec.xlsx"
W2V_WORDS_C_VECTORS = "./data/w2v_on_words/w2v_words_C_docvec.xlsx"

# DBSCAN Parameters for eps.
DBSCAN_EPS = "./input/dbscan-eps.json"

# Output Folders

# K-means output
KMEANS_BERT_SOURCE_OUTPUT_FOLDER = "./output/K-means/Bert_On_Source_Groups/"
KMEANS_D2V_SOURCE_OUTPUT_FOLDER = "./output/K-means/D2V_On_Source_Groups/"
KMEANS_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/K-means/TFIDF_On_Lemots_Groups/"
KMEANS_TFIDF_WORDS_OUTPUT_FOLDER = "./output/K-means/TFIDF_On_Words_Groups/"
KMEANS_W2V_LEMOTS_OUTPUT_FOLDER = "./output/K-means/W2V_On_Lemots_Groups/"
KMEANS_W2V_WORDS_OUTPUT_FOLDER = "./output/K-means/W2V_On_Words_Groups/"

# DBSCAN output
DBSCAN_BERT_SOURCE_OUTPUT_FOLDER = "./output/DBSCAN/Bert_On_Source_Groups/"
DBSCAN_D2V_SOURCE_OUTPUT_FOLDER = "./output/DBSCAN/D2V_On_Source_Groups/"
DBSCAN_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/DBSCAN/TFIDF_On_Lemots_Groups/"
DBSCAN_TFIDF_WORDS_OUTPUT_FOLDER = "./output/DBSCAN/TFIDF_On_Words_Groups/"
DBSCAN_W2V_LEMOTS_OUTPUT_FOLDER = "./output/DBSCAN/W2V_On_Lemots_Groups/"
DBSCAN_W2V_WORDS_OUTPUT_FOLDER = "./output/DBSCAN/W2V_On_Words_Groups/"

# Mixture of Gaussian output
MixOG_BERT_SOURCE_OUTPUT_FOLDER = "./output/MixOG/Bert_On_Source_Groups/"
MixOG_D2V_SOURCE_OUTPUT_FOLDER = "./output/MixOG/D2V_On_Source_Groups/"
MixOG_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/MixOG/TFIDF_On_Lemots_Groups/"
MixOG_TFIDF_WORDS_OUTPUT_FOLDER = "./output/MixOG/TFIDF_On_Words_Groups/"
MixOG_W2V_LEMOTS_OUTPUT_FOLDER = "./output/MixOG/W2V_On_Lemots_Groups/"
MixOG_W2V_WORDS_OUTPUT_FOLDER = "./output/MixOG/W2V_On_Words_Groups/"


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]


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


def visualize_with_tsne(vectors1, vectors2, labels1, labels2, label1_name, label2_name, title, save_path):
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


def visualize_with_umap(vectors, labels, title, save_path):
    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42).fit_transform(vectors)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(title)
    plt.savefig(save_path)
    print(f"UMAP plot saved in: {save_path}")


def plot_k_distance_graph(vectors_path_group1, vectors_path_group2, save_folder, groups_name):
    # Load vectors
    vectors_group1 = get_vectors_from(vectors_path_group1)
    vectors_group2 = get_vectors_from(vectors_path_group2)

    # Concatenate vectors from both groups
    all_vectors = np.vstack((vectors_group1, vectors_group2))

    # Calculate distances to the k-nearest neighbor
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(all_vectors)
    distances, indices = nbrs.kneighbors(all_vectors)
    distances = np.sort(distances[:, 1], axis=0)

    # Plot the K-Distance Graph
    plt.plot(range(1, len(distances) + 1), distances)
    plt.title('K-Distance Graph for Choosing eps')
    plt.xlabel('Number of Points')
    plt.ylabel('k-Distance')
    plt.savefig(f"{save_folder}/k_distance_graph_{groups_name}")


def k_means_clustering(vectors1, vectors2):
    all_vectors = np.vstack((vectors1, vectors2))
    kmeans = KMeans(n_clusters=2, random_state=42).fit(all_vectors)
    labels1 = kmeans.predict(vectors1)
    labels2 = kmeans.predict(vectors2)
    return labels1, labels2


def dbscan_clustering(vectors1, vectors2, eps, min_samples):
    all_vectors = np.vstack((vectors1, vectors2))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Adjust parameters as needed
    labels_all = dbscan.fit_predict(all_vectors)
    labels1 = labels_all[:len(vectors1)]
    labels2 = labels_all[len(vectors1):]
    # Mark second group with 1 instead of -1, first group is 0. (like K-Means).
    labels1 = [-1 * i for i in labels1]
    labels2 = [-1 * i for i in labels2]
    return labels1, labels2


def mixture_of_gaussian_clustering(vectors1, vectors2):
    all_vectors = np.vstack((vectors1, vectors2))
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(all_vectors)
    labels1 = gmm.predict(vectors1)
    labels2 = gmm.predict(vectors2)
    return labels1, labels2


def create_output_for_groups(group1_path, group2_path, group3_path, output_folder, alg):
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

        # Create clusters
        if alg[0] == ALG["K-means"]:
            print("Clustering algorithm: K-means.")
            labels1, labels2 = k_means_clustering(vectors1, vectors2)
        elif alg[0] == ALG["DBSCAN"]:
            print("Clustering algorithm: DBSCAN.")
            labels1, labels2 = dbscan_clustering(vectors1, vectors2, alg[1], alg[2])
        elif alg[0] == ALG["Mixture of Gaussian"]:
            print("Clustering algorithm: Mixture of Gaussian.")
            labels1, labels2 = mixture_of_gaussian_clustering(vectors1, vectors2)

        else:
            raise Exception("Unknown clustering algorithm.")
        print("clustering finish.")

        # Evaluate clusters
        print("Evaluate results:")
        precision_1, recall_1, f1_1, accuracy_1 = evaluate_clustering(np.zeros(len(vectors1)), np.ones(len(vectors2)),
                                                                      labels1,
                                                                      labels2)
        precision_2, recall_2, f1_2, accuracy_2 = evaluate_clustering(np.ones(len(vectors2)), np.zeros(len(vectors1)),
                                                                      labels1,
                                                                      labels2)

        # Create output for JSON
        result_1 = {
            'precision': precision_1,
            'recall': recall_1,
            'f1': f1_1,
            'accuracy': accuracy_1
        }
        result_2 = {
            'precision': precision_2,
            'recall': recall_2,
            'f1': f1_2,
            'accuracy': accuracy_2
        }

        # Save the result into json file.
        json_save_path = os.path.join(output_folder, f'{pair_describe}_1.json')
        with open(json_save_path, 'w') as json_file:
            json.dump(result_1, json_file)
        print(f"Evaluate result saved in: {json_save_path}")
        json_save_path = os.path.join(output_folder, f'{pair_describe}_2.json')
        with open(json_save_path, 'w') as json_file:
            json.dump(result_2, json_file)
        print(f"Evaluate result saved in: {json_save_path}")

        # Plot clusters
        if alg[0] == ALG["K-means"] or alg[0] == ALG["Mixture of Gaussian"]:
            visualize_with_tsne(vectors1, vectors2, labels1, labels2, pair_describe[0], pair_describe[2],
                            f"t-SNE {pair_describe}",
                                os.path.join(output_folder, f'tsne_{pair_describe}.png'))
        elif alg[0] == ALG["DBSCAN"]:
            visualize_with_umap(vectors1 + vectors2, labels1 + labels2, f"UMAP {pair_describe}",
                                os.path.join(output_folder, f'umap_{pair_describe}.png'))

        print("\n")


def kmeans_output():
    # K-means
    print("Clustering Matrices using K-MEANS...\n")

    # K-means for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
                             group3_path=BERT_SOURCE_C_VECTORS,
                             output_folder=KMEANS_BERT_SOURCE_OUTPUT_FOLDER, alg=[ALG["K-means"]])

    # K-means for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
                             group3_path=D2V_SOURCE_C_VECTORS,
                             output_folder=KMEANS_D2V_SOURCE_OUTPUT_FOLDER, alg=[ALG["K-means"]])

    # K-means for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=KMEANS_TFIDF_LEMOTS_OUTPUT_FOLDER, alg=[ALG["K-means"]])

    # K-means for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
                             group3_path=TFIDF_WORDS_C_VECTORS,
                             output_folder=KMEANS_TFIDF_WORDS_OUTPUT_FOLDER, alg=[ALG["K-means"]])

    # K-means for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
                             group3_path=W2V_LEMOTS_C_VECTORS,
                             output_folder=KMEANS_W2V_LEMOTS_OUTPUT_FOLDER, alg=[ALG["K-means"]])

    # K-means for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
                             group3_path=W2V_WORDS_C_VECTORS,
                             output_folder=KMEANS_W2V_WORDS_OUTPUT_FOLDER, alg=[ALG["K-means"]])


def k_distance_graphs_output():
    # Bert On Source
    plot_k_distance_graph(BERT_SOURCE_A_VECTORS, BERT_SOURCE_B_VECTORS, DBSCAN_BERT_SOURCE_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(BERT_SOURCE_A_VECTORS, BERT_SOURCE_C_VECTORS, DBSCAN_BERT_SOURCE_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(BERT_SOURCE_B_VECTORS, BERT_SOURCE_C_VECTORS, DBSCAN_BERT_SOURCE_OUTPUT_FOLDER, "BC")

    # D2V On Source
    plot_k_distance_graph(D2V_SOURCE_A_VECTORS, D2V_SOURCE_B_VECTORS, DBSCAN_D2V_SOURCE_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(D2V_SOURCE_A_VECTORS, D2V_SOURCE_C_VECTORS, DBSCAN_D2V_SOURCE_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(D2V_SOURCE_B_VECTORS, D2V_SOURCE_C_VECTORS, DBSCAN_D2V_SOURCE_OUTPUT_FOLDER, "BC")

    # TFIDF On Lemots
    plot_k_distance_graph(TFIDF_LEMOTS_A_VECTORS, TFIDF_LEMOTS_B_VECTORS, DBSCAN_TFIDF_LEMOTS_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(TFIDF_LEMOTS_A_VECTORS, TFIDF_LEMOTS_C_VECTORS, DBSCAN_TFIDF_LEMOTS_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(TFIDF_LEMOTS_B_VECTORS, TFIDF_LEMOTS_C_VECTORS, DBSCAN_TFIDF_LEMOTS_OUTPUT_FOLDER, "BC")

    # TFIDF On Words
    plot_k_distance_graph(TFIDF_WORDS_A_VECTORS, TFIDF_WORDS_B_VECTORS, DBSCAN_TFIDF_WORDS_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(TFIDF_WORDS_A_VECTORS, TFIDF_WORDS_B_VECTORS, DBSCAN_TFIDF_WORDS_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(TFIDF_WORDS_A_VECTORS, TFIDF_WORDS_B_VECTORS, DBSCAN_TFIDF_WORDS_OUTPUT_FOLDER, "BC")

    # W2V On Lemots
    plot_k_distance_graph(W2V_LEMOTS_A_VECTORS, W2V_LEMOTS_B_VECTORS, DBSCAN_W2V_LEMOTS_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(W2V_LEMOTS_A_VECTORS, W2V_LEMOTS_C_VECTORS, DBSCAN_W2V_LEMOTS_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(W2V_LEMOTS_B_VECTORS, W2V_LEMOTS_C_VECTORS, DBSCAN_W2V_LEMOTS_OUTPUT_FOLDER, "BC")

    # W2V On Words
    plot_k_distance_graph(W2V_WORDS_A_VECTORS, W2V_WORDS_B_VECTORS, DBSCAN_W2V_WORDS_OUTPUT_FOLDER, "AB")
    plot_k_distance_graph(W2V_WORDS_A_VECTORS, W2V_WORDS_C_VECTORS, DBSCAN_W2V_WORDS_OUTPUT_FOLDER, "AC")
    plot_k_distance_graph(W2V_WORDS_B_VECTORS, W2V_WORDS_C_VECTORS, DBSCAN_W2V_WORDS_OUTPUT_FOLDER, "BC")


def dbscan_output():
    # DBSCAN
    print("Clustering Matrices using DBSCAN...\n")

    eps = read_json_file(DBSCAN_EPS)

    # DBSCAN for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
                             group3_path=BERT_SOURCE_C_VECTORS,
                             output_folder=DBSCAN_BERT_SOURCE_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps["Bert_eps"], 2500])

    # DBSCAN for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
                             group3_path=D2V_SOURCE_C_VECTORS,
                             output_folder=DBSCAN_D2V_SOURCE_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps['D2V_eps'], 1500])

    # DBSCAN for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=DBSCAN_TFIDF_LEMOTS_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps['TFIDF_eps'], 2500])

    # DBSCAN for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
                             group3_path=TFIDF_WORDS_C_VECTORS,
                             output_folder=DBSCAN_TFIDF_WORDS_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps['TFIDF_eps'], 2500])

    # DBSCAN for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
                             group3_path=W2V_LEMOTS_C_VECTORS,
                             output_folder=DBSCAN_W2V_LEMOTS_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps['W2V_eps'], 2500])

    # DBSCAN for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
                             group3_path=W2V_WORDS_C_VECTORS,
                             output_folder=DBSCAN_W2V_WORDS_OUTPUT_FOLDER,
                             alg=[ALG["DBSCAN"], eps['W2V_eps'], 2500])


def mixture_of_gaussian_output():
    # Mixture of Gaussian
    print("Clustering Matrices using Mixture of Gaussian...\n")

    # Mixture of Gaussian for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
                             group3_path=BERT_SOURCE_C_VECTORS,
                             output_folder=MixOG_BERT_SOURCE_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])

    # Mixture of Gaussian for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
                             group3_path=D2V_SOURCE_C_VECTORS,
                             output_folder=MixOG_D2V_SOURCE_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])

    # Mixture of Gaussian for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=MixOG_TFIDF_LEMOTS_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])

    # Mixture of Gaussian for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
                             group3_path=TFIDF_WORDS_C_VECTORS,
                             output_folder=MixOG_TFIDF_WORDS_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])

    # Mixture of Gaussian for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
                             group3_path=W2V_LEMOTS_C_VECTORS,
                             output_folder=MixOG_W2V_LEMOTS_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])

    # Mixture of Gaussian for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
                             group3_path=W2V_WORDS_C_VECTORS,
                             output_folder=MixOG_W2V_WORDS_OUTPUT_FOLDER, alg=[ALG["Mixture of Gaussian"]])


def main():
    # K-means Clustering
    kmeans_output()

    # DBSCAN Clustering
    dbscan_output()

    # Mixture of Gaussian Clustering
    mixture_of_gaussian_output()


if __name__ == "__main__":
    main()
