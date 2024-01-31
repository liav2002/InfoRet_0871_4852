import pandas as pd


def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]
