# Set Data Path
TFIDF_LEMOTS_A_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"
LogisticRegression_TFIDF_LEMOTS_OUTPUT_FOLDER = "./Task2/output4dror/LoR/Iteration2/"
IterationNum = "LoR2"

# Get Vectors Function
import pandas as pd
def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]

# Use Get Vectors Function to Retrive the vectors for 3 groups of data
vectors_A = get_vectors_from(TFIDF_LEMOTS_A_VECTORS)
vectors_B = get_vectors_from(TFIDF_LEMOTS_B_VECTORS)
vectors_C = get_vectors_from(TFIDF_LEMOTS_C_VECTORS)

# Get Feature Function
def get_features_names_form(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, header=None)
        # Extract feature names from the first row
        feature_names = list(df.iloc[0, 1:])
        return feature_names
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return None

# Use Get Feature Function to extract the features names
features_names = get_features_names_form(TFIDF_LEMOTS_A_VECTORS)

# Function to remove features
def remove_features_from(vectors, features_names, features2remove):
    # Create a dictionary mapping feature names to their indices
    feature_index_map = {feature: index for index, feature in enumerate(features_names)}

    # Iterate through each vector
    for vector in vectors:
        # Iterate through features to remove
        for feature in features2remove:
            # Check if the feature exists in the vector
            if feature in feature_index_map:
                # Set the value of the feature to 0
                index = feature_index_map[feature]
                vector[index] = 0

    return vectors
    
# Remove Feartures
filtered_features = []
vectors_A = remove_features_from(vectors_A, features_names, filtered_features)
vectors_B = remove_features_from(vectors_B, features_names, filtered_features)
vectors_C = remove_features_from(vectors_C, features_names, filtered_features)

# Intialize vectors stack of two groups (group1 & group2)
import numpy as np
pair_name = "A&B"
all_vectors = np.vstack((vectors_A, vectors_B))
all_labels = np.hstack((np.zeros(len(vectors_A)), np.ones(len(vectors_B))))

# Shuffle & Split data for prepare to the model fit
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
all_vectors, all_labels = shuffle(all_vectors, all_labels, random_state=42)
train_vectors, test_vectors, train_labels, test_labels = train_test_split(all_vectors, all_labels,
                                                                                  test_size=0.2, random_state=42)
train_vectors, val_vectors, train_labels, val_labels = train_test_split(train_vectors, train_labels,
                                                                                test_size=0.1, random_state=42)
		
# Intialize the model        
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)

# Fit the model
model.fit(train_vectors, train_labels)

# Evaluate Results
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
predictions = model.predict(test_vectors)
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

# Save results to a JSON file
import os
import json
result = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}
result_path = os.path.join(LogisticRegression_TFIDF_LEMOTS_OUTPUT_FOLDER, f'result_{pair_name}-{IterationNum}.json')
with open(result_path, 'w') as json_file:
    json.dump(result, json_file)
print(f"Evaluate result saved in: {result_path}")

# Get the indices of the top 50 positive & negative features
top_positive_indices = model.coef_[0].argsort()[-50:][::-1]
top_negative_indices = model.coef_[0].argsort()[:50]

# Extract words and weights for the top positive & negative features
top_positive_features = [[features_names[i], model.coef_[0][i]] for i in top_positive_indices]
top_negative_features = [[features_names[i], model.coef_[0][i]] for i in top_negative_indices]

# Save selected features to Excel file
features_df = pd.DataFrame({'Positive Word': [pair[0] for pair in top_positive_features],
                            'Positive Weight': [pair[1] for pair in top_positive_features],
                            'Negative Word': [pair[0] for pair in top_negative_features],
                            'Negative Weight': [pair[1] for pair in top_negative_features]})

features_path = os.path.join(LogisticRegression_TFIDF_LEMOTS_OUTPUT_FOLDER, f'features_{pair_name}-{IterationNum}.xlsx')
features_df.to_excel(features_path, index=False)
print(f"Selected features saved in: {features_path}")