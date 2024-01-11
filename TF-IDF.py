import pandas as pd
from collections import Counter
import string
import openpyxl
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

CONTENT_COL  = 2
ID_COL = 1
A_START = 2
A_FINISH = 5001
B_START = 5006
B_FINISH = 10005
C_START = 10010
C_FINISH = 15009
FILE_PATH = "C:\\Users\\nehor\\OneDrive - click\\שולחן העבודה\\InfoRet_0871_4852\\data\\clean_15000.xlsx"
SHEET = "Sheet1"
A_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\A_CLEAN_VOCA.xlsx"
B_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\B_CLEAN_VOCA.xlsx"
C_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\C_CLEAN_VOCA.xlsx"
A_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\A_CLEAN_LEN.xlsx"
B_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\B_CLEAN_LEN.xlsx"
C_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\C_CLEAN_LEN.xlsx"
METRIX_A_PATH = "C:\\Users\\nehor\\Downloads\\METRIX_A.xlsx"


def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
    column_values = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None, usecols=[column_index], skiprows=start_row-1, nrows=end_row-start_row+1)[column_index]
    all_text = ' '.join(map(str, column_values))
    word_frequency = Counter(all_text.split())
    result_list = list(word_frequency.items())
    result_list.sort(key=lambda x: x[1], reverse=True)
    df_output = pd.DataFrame(result_list, columns=["Word", "Count"])
    df_output.to_excel(output_excel_path, index=False)
    return result_list


def count_words_in_cell(cell_value):
    # Count the number of words in a cell
    words = cell_value.split()
    return len(words)


def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
    # Read Excel file using pandas
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    # Create a new DataFrame with only the required columns
    result_df = pd.DataFrame(columns=["מזהה/מספר הקובץ", "paragraph number of words"])

    # Iterate through the specified range and count words in each cell
    for index, row in df.iloc[start_row - 2:end_row - 1].iterrows():
        paragraph_id = row["מזהה/מספר הקובץ"]
        cell_value = row["תוכן הקובץ"]
        word_count = count_words_in_cell(str(cell_value))

        # Append the results to the new DataFrame
        result_df = pd.concat([result_df, pd.DataFrame({"מזהה/מספר הקובץ": [paragraph_id], "paragraph number of words": [word_count]})],
                              ignore_index=True)

    # Write the new DataFrame to a new Excel file
    result_df.to_excel(output_excel_path, index=False)


def c(w, d):
    return d.count(w)


def create_matrix(file1_path, file2_path):
    # Read the Excel file into a DataFrame
    df1 = pd.read_excel(file1_path)

    # Extract only the "Word" column without the index
    words_to_copy = df1["Word"].iloc[0:].tolist()

    # Create a new DataFrame with the values to copy in the first column and second row
    new_df = pd.DataFrame({"Word": words_to_copy})

    # Read the Excel file into a DataFrame
    df2 = pd.read_excel(file2_path)

    # Remove the column named "paragraph number of words"
    df2 = df2.drop("paragraph number of words", axis=1)

    # Transpose the DataFrame
    df_transposed = df2.T.reset_index()

    df_transposed.to_excel("C:\\Users\\nehor\\Downloads\\ID.xlsx", header=None, index=False)

    new_df.to_excel("C:\\Users\\nehor\\Downloads\\WORD.xlsx", header=None, index=False)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def excel_column_to_list(excel_file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)

    # Extract the values from the first column (assuming it's the index 0)
    first_column_values = df.iloc[:, 0].tolist()

    return first_column_values

    # # Merge on the "Word" column
    # merged_df = pd.merge(new_df, df_transposed, left_on='Word', right_on='index')
    #
    # # Drop unnecessary columns
    # merged_df = merged_df.drop(columns=['Word', 'index'])
    #
    # # Write the merged DataFrame to a new Excel file
    # merged_df.to_excel(output_path, header=None, index=False)

def main():
    # A_frequency_result = get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_OUTPUT_PATH)
    # B_frequency_result = get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_OUTPUT_PATH)
    # C_frequency_result = get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_OUTPUT_PATH)

    # get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_LEN_OUTPUT_PATH)
    # get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_LEN_OUTPUT_PATH)
    # get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_LEN_OUTPUT_PATH)

    create_matrix(B_OUTPUT_PATH, B_LEN_OUTPUT_PATH)
    print(len(excel_column_to_list("C:\\Users\\nehor\\Downloads\\WORD.xlsx")))
    print("--------")
    print(excel_column_to_list("C:\\Users\\nehor\\Downloads\\WORD.xlsx"))
    print("Word frequency in the specified range:")


if __name__ == "__main__":
    main()