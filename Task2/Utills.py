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