import pandas as pd
import os


def all_files_exist(file_paths):
    return all(os.path.isfile(file_path) for file_path in file_paths)


def read_excel_file(file_path="./data/15000.xlsx"):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    return df
