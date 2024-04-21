# Set Data Path
TFIDF_LEMOTS_A_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./Task2/data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"
SVC_TFIDF_LEMOTS_OUTPUT_FOLDER = "./Task2/output/SVC/TFIDF_On_Lemots_Groups/"

# Use Get Vectors Function to Retrieve the vectors for 3 groups of data
vectors_A = get_vectors_from(TFIDF_LEMOTS_A_VECTORS)
vectors_B = get_vectors_from(TFIDF_LEMOTS_B_VECTORS)
vectors_C = get_vectors_from(TFIDF_LEMOTS_C_VECTORS)

# Use Get Feature Function to extract the features names
features_names = get_features_names_form(TFIDF_LEMOTS_A_VECTORS)

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
from sklearn.svm import SVC
model = SVC(kernel='linear', random_state=42)

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
result_path = os.path.join(SVC_TFIDF_LEMOTS_OUTPUT_FOLDER, f'result_{pair_name}.json')
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

features_path = os.path.join(SVC_TFIDF_LEMOTS_OUTPUT_FOLDER, f'features_{pair_name}.xlsx')
features_df.to_excel(features_path, index=False)
print(f"Selected features saved in: {features_path}")
