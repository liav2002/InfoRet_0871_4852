from Utills import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import copy
import os
import json

# Set Data Path
TFIDF_LEMOTS_A_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"

# Use Get Vectors Function to retrive the vectors for 3 groups of data
print("START")
vectors_A = get_vectors_from(TFIDF_LEMOTS_A_VECTORS)
print("vectors A retrieved.")
vectors_B = get_vectors_from(TFIDF_LEMOTS_B_VECTORS)
print("vectors B retrieved.")
vectors_C = get_vectors_from(TFIDF_LEMOTS_C_VECTORS)
print("vectors C retrieved.")

# Save vectors according paris
vectors_A1 = copy.deepcopy(vectors_A)
print("copy of vectors_A to A&B was created.")
vectors_B1 = copy.deepcopy(vectors_B)
print("copy of vectors_B to A&B was created.")
vectors_A2 = copy.deepcopy(vectors_A)
print("copy of vectors_A to A&C was created.")
vectors_C2 = copy.deepcopy(vectors_C)
print("copy of vectors_C to A&C was created.")
vectors_B3 = copy.deepcopy(vectors_B)
print("copy of vectors_B to B&C was created.")
vectors_C3 = copy.deepcopy(vectors_C)
print("copy of vectors_C to B&C was created.")

# Use Get Feature Function to extract the features names
features_names = get_features_names_form(TFIDF_LEMOTS_A_VECTORS)
print("features names retrieved.")


def make_iteration(pair_name, vectors1, vectors2, iteration_num, output_folder, model_type):
    # Intialize vectors stack of two groups (group1 & group2)
    all_vectors = np.vstack((vectors1, vectors2))
    all_labels = np.hstack((np.zeros(len(vectors1)), np.ones(len(vectors2))))

    # Shuffle & Split data for prepare to the model fit
    all_vectors, all_labels = shuffle(all_vectors, all_labels, random_state=42)
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                              test_size=0.2, random_state=42)
    train_vectors, val_vectors, train_labels, val_labels = train_test_split(train_vectors, train_labels,
                                                                            test_size=0.1, random_state=42)
    # Intialize the model
    if model_type == "LoR":
        model = LogisticRegression(random_state=42)
    else:
        model = SVC(kernel='linear', random_state=42)

    # Fit the model
    model.fit(train_vectors, train_labels)

    # Evaluate Results
    predictions = model.predict(test_vectors)
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
    result_path = os.path.join(output_folder, f'result_{pair_name}-{iteration_num}.json')
    with open(result_path, 'w') as json_file:
        json.dump(result, json_file)
    print(f"Evaluate result saved in: {result_path}")

    # Get the indices of the top 50 positive & negative features
    top_positive_indices = model.coef_[0].argsort()[-100:][::-1]
    top_negative_indices = model.coef_[0].argsort()[:100]

    # Extract words and weights for the top positive & negative features
    top_positive_features = [[features_names[i], model.coef_[0][i]] for i in top_positive_indices]
    top_negative_features = [[features_names[i], model.coef_[0][i]] for i in top_negative_indices]

    # Save selected features to Excel file
    features_df = pd.DataFrame({'Positive Word': [pair[0] for pair in top_positive_features],
                                'Positive Weight': [pair[1] for pair in top_positive_features],
                                'Negative Word': [pair[0] for pair in top_negative_features],
                                'Negative Weight': [pair[1] for pair in top_negative_features]})

    features_path = os.path.join(output_folder,
                                 f'features_{pair_name}-{iteration_num}.xlsx')
    features_df.to_excel(features_path, index=False)
    print(f"Selected features saved in: {features_path}")


def main():
    model = ""
    iteration_num = 1
    user_option = 0
    finish = False

    # Get model type from the user
    print("\n\nchoose model:")
    print("1. Logistic Regression")
    print("2. Support Vector Machine.")
    while user_option not in [1, 2]:
        user_option = int(input("Enter your choice: "))
        if user_option == 1:
            print("Logistic Regression was selected.")
            model = "LoR"
        elif user_option == 2:
            print("Support Vector Machine was selected.")
            model = "SVM"
        else:
            print("Invalid choice, try again.")

    # start iterations
    print("\n\n")
    while not finish:
        print(f"Iteration: {iteration_num}")
        output_folder = f"./output/{model}/Iteration{str(iteration_num)}/"

        # prepare iteration for A&B
        global vectors_A1
        global vectors_B1
        pair_name = "A&B"
        features2remove_path = f"./input/words2remove-{model}_{pair_name[0]}{pair_name[2]}.xlsx"
        print(f"Working on pair: {pair_name}.")

        # Remove Feartures
        cont = False
        filtered_features = []
        while not cont:
            try:
                filtered_features = get_features2remove(features2remove_path)
                cont = True
            except Exception as e:
                print(e)
                print(f"make sure you close the file: {features2remove_path}.")
                input("Press any key for continue...")

        vectors_A1 = remove_features_from(vectors_A1, features_names, filtered_features)
        vectors_B1 = remove_features_from(vectors_B1, features_names, filtered_features)

        # make iteraion
        make_iteration(pair_name=pair_name, vectors1=vectors_A1, vectors2=vectors_B1, iteration_num=iteration_num,
                       output_folder=output_folder, model_type=model)

        # prepare iteration for A&C
        global vectors_A2
        global vectors_C2
        pair_name = "A&C"
        features2remove_path = f"./input/words2remove-{model}_{pair_name[0]}{pair_name[2]}.xlsx"
        print(f"Working on pair: {pair_name}.")

        # Remove Feartures
        cont = False
        filtered_features = []
        while not cont:
            try:
                filtered_features = get_features2remove(features2remove_path)
                cont = True
            except Exception as e:
                print(e)
                print(f"make sure you close the file: {features2remove_path}.")
                input("Press any key for continue...")

        vectors_A2 = remove_features_from(vectors_A2, features_names, filtered_features)
        vectors_C2 = remove_features_from(vectors_C2, features_names, filtered_features)

        # make iteraion
        make_iteration(pair_name=pair_name, vectors1=vectors_A2, vectors2=vectors_C2, iteration_num=iteration_num,
                       output_folder=output_folder, model_type=model)

        # prepare iteration for B&C
        global vectors_B3
        global vectors_C3
        pair_name = "B&C"
        features2remove_path = f"./input/words2remove-{model}_{pair_name[0]}{pair_name[2]}.xlsx"
        print(f"Working on pair: {pair_name}.")

        # Remove Feartures
        cont = False
        filtered_features = []
        while not cont:
            try:
                filtered_features = get_features2remove(features2remove_path)
                cont = True
            except Exception as e:
                print(e)
                print(f"make sure you close the file: {features2remove_path}.")
                input("Press any key for continue...")

        vectors_B3 = remove_features_from(vectors_B3, features_names, filtered_features)
        vectors_C3 = remove_features_from(vectors_C3, features_names, filtered_features)

        # make iteraion
        make_iteration(pair_name=pair_name, vectors1=vectors_B3, vectors2=vectors_C3, iteration_num=iteration_num,
                       output_folder=output_folder, model_type=model)

        print("\n")

        print("update features2remove before continure the next iteration\n\n")

        askContinue = input("Enter 'next' for continue, 'exit' for exit:\n")
        while askContinue != "next" and askContinue != "exit":
            askContinue = input("")

        if askContinue == "exit":
            finish = True
        else:
            iteration_num += 1


if __name__ == "__main__":
    main()
    print("END")
