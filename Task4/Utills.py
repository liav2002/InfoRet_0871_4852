import pandas as pd


def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]


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


def get_features2remove(file_path):
    df = pd.read_excel(file_path, header=None)  # Specify header=None to indicate no header
    first_col_values = df.iloc[:, 0].tolist()[1:]  # Extract the first column using iloc
    return first_col_values


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
