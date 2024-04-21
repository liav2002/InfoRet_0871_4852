import os
import json
import numpy as np
from Utills import *
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import relu, sigmoid

ALG = {"ANN": 1, "NB": 2, "LogisticRegression": 3}

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

# Output Folders

# ANN output
ANN_BERT_SOURCE_OUTPUT_FOLDER = "./output/ANN/Bert_On_Source_Groups/"
ANN_D2V_SOURCE_OUTPUT_FOLDER = "./output/ANN/D2V_On_Source_Groups/"
ANN_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/ANN/TFIDF_On_Lemots_Groups/"
ANN_TFIDF_WORDS_OUTPUT_FOLDER = "./output/ANN/TFIDF_On_Words_Groups/"
ANN_W2V_LEMOTS_OUTPUT_FOLDER = "./output/ANN/W2V_On_Lemots_Groups/"
ANN_W2V_WORDS_OUTPUT_FOLDER = "./output/ANN/W2V_On_Words_Groups/"

# NB output
NB_BERT_SOURCE_OUTPUT_FOLDER = "./output/NB/Bert_On_Source_Groups/"
NB_D2V_SOURCE_OUTPUT_FOLDER = "./output/NB/D2V_On_Source_Groups/"
NB_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/NB/TFIDF_On_Lemots_Groups/"
NB_TFIDF_WORDS_OUTPUT_FOLDER = "./output/NB/TFIDF_On_Words_Groups/"
NB_W2V_LEMOTS_OUTPUT_FOLDER = "./output/NB/W2V_On_Lemots_Groups/"
NB_W2V_WORDS_OUTPUT_FOLDER = "./output/NB/W2V_On_Words_Groups/"

# LogisticRegression output
LogisticRegression_BERT_SOURCE_OUTPUT_FOLDER = "./output/LogisticRegression/Bert_On_Source_Groups/"
LogisticRegression_D2V_SOURCE_OUTPUT_FOLDER = "./output/LogisticRegression/D2V_On_Source_Groups/"
LogisticRegression_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/LogisticRegression/TFIDF_On_Lemots_Groups/"
LogisticRegression_TFIDF_WORDS_OUTPUT_FOLDER = "./output/LogisticRegression/TFIDF_On_Words_Groups/"
LogisticRegression_W2V_LEMOTS_OUTPUT_FOLDER = "./output/LogisticRegression/W2V_On_Lemots_Groups/"
LogisticRegression_W2V_WORDS_OUTPUT_FOLDER = "./output/LogisticRegression/W2V_On_Words_Groups/"


class ANN_C:
    def __init__(self, output_folder, topology):
        self.EPOCHS = 15
        self.BATCH_SIZE = 128
        self.output_folder = output_folder
        self.topology = topology  # [(10, relu), (10, relu), (7, relu)] || [(10, gelu), (10, gelu), (7, gelu)]

    def train_neural_network(self, train_data, train_labels, val_data, val_labels, pairs_name):
        # Create a sequential model
        model = Sequential()

        # Add an Embedding layer with input_shape matching the shape of each vector in train_data
        model.add(Dense(units=len(train_data), input_shape=(train_data.shape[1],)))

        # Flatten the output of the Embedding layer
        model.add(Flatten())

        # Adding layers based on the provided topology
        for layer_size, activation_func in self.topology:
            model.add(Dense(units=layer_size, activation=activation_func))

        # Output layer with sigmoid activation
        model.add(Dense(1, activation=sigmoid))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
        checkpoint_path = os.path.join(self.output_folder, 'best_model' + '_' + pairs_name + '.h5')
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

        # Train the model
        print("Training model...")
        history = model.fit(
            train_data, train_labels,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            validation_data=(val_data, val_labels),
            callbacks=[early_stopping, model_checkpoint]
        )
        print("Model training finished.")

        # Save training history to a file
        save_path = os.path.join(self.output_folder, 'training_history' + '_' + pairs_name + '.json')
        with open(save_path, 'w') as json_file:
            json.dump(history.history, json_file)
        print(f"Model result saved in {save_path}.")

        return model

    def evaluate_and_save_results(self, model, test_data, test_labels, pair_name):
        print("Evaluate results:")
        predictions = model.predict(test_data)
        predictions_binary = np.round(predictions).flatten().astype(int)

        accuracy = accuracy_score(test_labels, predictions_binary)
        precision = precision_score(test_labels, predictions_binary)
        recall = recall_score(test_labels, predictions_binary)
        f1 = f1_score(test_labels, predictions_binary)

        # Save results to a JSON file
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        result_path = os.path.join(self.output_folder, f'result_{pair_name}.json')
        with open(result_path, 'w') as json_file:
            json.dump(result, json_file)
        print(f"Evaluate result saved in: {result_path}")

        # Visualize in UMAP
        umap = TSNE(n_components=2, random_state=42)
        umap_vectors = umap.fit_transform(test_data)

        plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1], c=test_labels, cmap='viridis', s=5)
        plt.title(f'UMAP Visualization for {pair_name} Pair')
        save_path = os.path.join(self.output_folder, f'umap_{pair_name}.png')
        plt.savefig(save_path)
        print(f"UMAP plot saved in: {save_path}\n\n")

    def run_experiment(self, vectors1, vectors2, pair_name):
        print(f"Working on pair: {pair_name} ...")

        # Combine vectors and create labels
        all_vectors = np.vstack((vectors1, vectors2))
        all_labels = np.hstack((np.zeros(len(vectors1)), np.ones(len(vectors2))))

        # Shuffle and split data
        all_vectors, all_labels = shuffle(all_vectors, all_labels, random_state=42)
        train_vectors, test_vectors, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                                  test_size=0.2, random_state=42)
        train_vectors, val_vectors, train_labels, val_labels = train_test_split(train_vectors, train_labels,
                                                                                test_size=0.1, random_state=42)

        # Train the neural network
        model = self.train_neural_network(train_vectors, train_labels, val_vectors, val_labels, pair_name)

        # Evaluate and save results
        self.evaluate_and_save_results(model, test_vectors, test_labels, pair_name)


class NB_C:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def evaluate_and_save_results(self, model, test_data, test_labels, pair_name):
        print("Evaluate results:")
        predictions = model.predict(test_data)

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        # Save results to a JSON file
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        result_path = os.path.join(self.output_folder, f'result_{pair_name}.json')
        with open(result_path, 'w') as json_file:
            json.dump(result, json_file)
        print(f"Evaluate result saved in: {result_path}")

        # Visualize in UMAP
        umap = TSNE(n_components=2, random_state=42)
        umap_vectors = umap.fit_transform(test_data)

        plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1], c=test_labels, cmap='viridis', s=5)
        plt.title(f'UMAP Visualization for {pair_name} Pair')
        save_path = os.path.join(self.output_folder, f'umap_{pair_name}.png')
        plt.savefig(save_path)
        print(f"UMAP plot saved in: {save_path}\n\n")

    def run_experiment(self, vectors1, vectors2, pair_name):
        print(f"Working on pair: {pair_name} ...")

        # Combine vectors and create labels
        all_vectors = np.vstack((vectors1, vectors2))
        all_labels = np.hstack((np.zeros(len(vectors1)), np.ones(len(vectors2))))

        # Shuffle and split data
        all_vectors, all_labels = shuffle(all_vectors, all_labels, random_state=42)
        train_vectors, test_vectors, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                                  test_size=0.2, random_state=42)
        train_vectors, val_vectors, train_labels, val_labels = train_test_split(train_vectors, train_labels,
                                                                                test_size=0.1, random_state=42)

        # Create the Naive Bayes model
        model = GaussianNB()

        # Fit the Naive Bayes model
        print("Training model...")
        model.fit(train_vectors, train_labels)
        print("Model training finished.")

        # Evaluate and save results
        self.evaluate_and_save_results(model, test_vectors, test_labels, pair_name)


class LogisticRegression_C:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.selected_features = []

    def evaluate_and_save_results(self, model, test_data, test_labels, pair_name):
        print("Evaluate results:")
        predictions = model.predict(test_data)

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        # Save results to a JSON file
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        result_path = os.path.join(self.output_folder, f'result_{pair_name}.json')
        with open(result_path, 'w') as json_file:
            json.dump(result, json_file)
        print(f"Evaluate result saved in: {result_path}")

        # Save selected features to JSON file
        features_path = os.path.join(self.output_folder, f'features_{pair_name}.json')
        with open(features_path, 'w') as json_file:
            json.dump(self.selected_features, json_file)
        print(f"Selected features saved in: {features_path}")

        # Visualize in UMAP
        umap = TSNE(n_components=2, random_state=42)
        umap_vectors = umap.fit_transform(test_data)

        plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1], c=test_labels, cmap='viridis', s=5)
        plt.title(f'UMAP Visualization for {pair_name} Pair')
        save_path = os.path.join(self.output_folder, f'umap_{pair_name}.png')
        plt.savefig(save_path)
        print(f"UMAP plot saved in: {save_path}\n\n")

    def run_experiment(self, vectors1, vectors2, pair_name):
        print(f"Working on pair: {pair_name} ...")

        # Combine vectors and create labels
        all_vectors = np.vstack((vectors1, vectors2))
        all_labels = np.hstack((np.zeros(len(vectors1)), np.ones(len(vectors2))))

        # Shuffle and split data
        all_vectors, all_labels = shuffle(all_vectors, all_labels, random_state=42)
        train_vectors, test_vectors, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                                  test_size=0.2, random_state=42)
        train_vectors, val_vectors, train_labels, val_labels = train_test_split(train_vectors, train_labels,
                                                                                test_size=0.1, random_state=42)

        # Create the Logistic Regression model
        model = LogisticRegression(random_state=42)

        # Fit the Logistic Regression model
        print("Training model...")
        model.fit(train_vectors, train_labels)
        print("Model training finished.")

        # Store selected features
        self.selected_features = model.coef_[0].tolist()

        # Evaluate and save results
        self.evaluate_and_save_results(model, test_vectors, test_labels, pair_name)


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

        if alg[0] == ALG["ANN"]:
            ann = ANN_C(output_folder=output_folder, topology=alg[1])
            ann.run_experiment(vectors1, vectors2, pair_describe)
        elif alg[0] == ALG["NB"]:
            nb = NB_C(output_folder=output_folder)
            nb.run_experiment(vectors1, vectors2, pair_describe)
        elif alg[0] == ALG["LogisticRegression"]:
            lr = LogisticRegression_C(output_folder=output_folder)
            lr.run_experiment(vectors1, vectors2, pair_describe)
            print(f"Selected features for pair {pair_describe}: {lr.selected_features}")

def ann_output():
    # ANN
    print("Clustering Matrices using ANN...\n")

    # ANN for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
                             group3_path=BERT_SOURCE_C_VECTORS,
                             output_folder=ANN_BERT_SOURCE_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])

    # ANN for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
                             group3_path=D2V_SOURCE_C_VECTORS,
                             output_folder=ANN_D2V_SOURCE_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])

    # ANN for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=ANN_TFIDF_LEMOTS_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])

    # ANN for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
                             group3_path=TFIDF_WORDS_C_VECTORS,
                             output_folder=ANN_TFIDF_WORDS_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])

    # ANN for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
                             group3_path=W2V_LEMOTS_C_VECTORS,
                             output_folder=ANN_W2V_LEMOTS_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])

    # ANN for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
                             group3_path=W2V_WORDS_C_VECTORS,
                             output_folder=ANN_W2V_WORDS_OUTPUT_FOLDER,
                             alg=[ALG["ANN"], [(10, relu), (10, relu), (7, relu)]])


def naive_bayes_output():
    # NB
    print("Clustering Matrices using Naive Bayes...\n")

    # NB for bert_on_source vectors
    print("Create output for 'Bert_On_Source' doc vectors.")
    create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
                             group3_path=BERT_SOURCE_C_VECTORS,
                             output_folder=NB_BERT_SOURCE_OUTPUT_FOLDER, alg=[ALG["NB"]])

    # NB for d2v_on_source vectors
    print("Create output for 'D2V_On_Source' doc vectors.")
    create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
                             group3_path=D2V_SOURCE_C_VECTORS,
                             output_folder=NB_D2V_SOURCE_OUTPUT_FOLDER, alg=[ALG["NB"]])

    # NB for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=NB_TFIDF_LEMOTS_OUTPUT_FOLDER, alg=[ALG["NB"]])

    # NB for tfidf_on_words vectors
    print("Create output for 'TFIDF_On_Words' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
                             group3_path=TFIDF_WORDS_C_VECTORS,
                             output_folder=NB_TFIDF_WORDS_OUTPUT_FOLDER, alg=[ALG["NB"]])

    # NB for w2v_on_lemots vectors
    print("Create output for 'W2V_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
                             group3_path=W2V_LEMOTS_C_VECTORS,
                             output_folder=NB_W2V_LEMOTS_OUTPUT_FOLDER, alg=[ALG["NB"]])

    # NB for w2v_on_words vectors
    print("Create output for 'W2V_On_Words' doc vectors.")
    create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
                             group3_path=W2V_WORDS_C_VECTORS,
                             output_folder=NB_W2V_WORDS_OUTPUT_FOLDER, alg=[ALG["NB"]])


def logistic_regression_output():
    # # Logistic Regression
    # print("Clustering Matrices using Logistic Regression...\n")

    # # Logistic Regression for bert_on_source vectors
    # print("Create output for 'Bert_On_Source' doc vectors.")
    # create_output_for_groups(group1_path=BERT_SOURCE_A_VECTORS, group2_path=BERT_SOURCE_B_VECTORS,
    #                          group3_path=BERT_SOURCE_C_VECTORS,
    #                          output_folder=LogisticRegression_BERT_SOURCE_OUTPUT_FOLDER,
    #                          alg=[ALG["LogisticRegression"]])

    # # Logistic Regression for d2v_on_source vectors
    # print("Create output for 'D2V_On_Source' doc vectors.")
    # create_output_for_groups(group1_path=D2V_SOURCE_A_VECTORS, group2_path=D2V_SOURCE_B_VECTORS,
    #                          group3_path=D2V_SOURCE_C_VECTORS,
    #                          output_folder=LogisticRegression_D2V_SOURCE_OUTPUT_FOLDER,
    #                          alg=[ALG["LogisticRegression"]])

    # Logistic Regression for tfidf_on_lemots vectors
    print("Create output for 'TFIDF_On_Lemots' doc vectors.")
    create_output_for_groups(group1_path=TFIDF_LEMOTS_A_VECTORS, group2_path=TFIDF_LEMOTS_B_VECTORS,
                             group3_path=TFIDF_LEMOTS_C_VECTORS,
                             output_folder=LogisticRegression_TFIDF_LEMOTS_OUTPUT_FOLDER,
                             alg=[ALG["LogisticRegression"]])

    # # Logistic Regression for tfidf_on_words vectors
    # print("Create output for 'TFIDF_On_Words' doc vectors.")
    # create_output_for_groups(group1_path=TFIDF_WORDS_A_VECTORS, group2_path=TFIDF_WORDS_B_VECTORS,
    #                          group3_path=TFIDF_WORDS_C_VECTORS,
    #                          output_folder=LogisticRegression_TFIDF_WORDS_OUTPUT_FOLDER,
    #                          alg=[ALG["LogisticRegression"]])

    # # Logistic Regression for w2v_on_lemots vectors
    # print("Create output for 'W2V_On_Lemots' doc vectors.")
    # create_output_for_groups(group1_path=W2V_LEMOTS_A_VECTORS, group2_path=W2V_LEMOTS_B_VECTORS,
    #                          group3_path=W2V_LEMOTS_C_VECTORS,
    #                          output_folder=LogisticRegression_W2V_LEMOTS_OUTPUT_FOLDER,
    #                          alg=[ALG["LogisticRegression"]])
    #
    # # Logistic Regression for w2v_on_words vectors
    # print("Create output for 'W2V_On_Words' doc vectors.")
    # create_output_for_groups(group1_path=W2V_WORDS_A_VECTORS, group2_path=W2V_WORDS_B_VECTORS,
    #                          group3_path=W2V_WORDS_C_VECTORS,
    #                          output_folder=LogisticRegression_W2V_WORDS_OUTPUT_FOLDER,
    #                          alg=[ALG["LogisticRegression"]])


def main():
    # # ANN Clustering
    # ann_output()
    #
    # # Naive Bayes Clustering
    # naive_bayes_output()

    # Logistic Regression Clustering
    logistic_regression_output()


if __name__ == "__main__":
    main()
