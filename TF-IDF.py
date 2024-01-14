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

def excel_column_to_list(excel_file):
    df = pd.read_excel(excel_file)
    first_column_values = df.iloc[:, 0].tolist()
    return first_column_values


def list_of_words(X_OUTPUT_PATH, X_LEN_OUTPUT_PATH, input_string):
    create_matrix(X_OUTPUT_PATH, X_LEN_OUTPUT_PATH)
    word_list = excel_column_to_list("C:\\Users\\nehor\\Downloads\\WORD.xlsx")
    word_counts = {}
    for word in word_list:
        count = input_string.split().count(word)
        word_counts[word] = count
    return word_counts
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_IDs_and_words(X_LEN_OUTPUT_PATH, X_OUTPUT_PATH):
    all_words = excel_column_to_list(X_OUTPUT_PATH)
    all_IDs = excel_column_to_list(X_LEN_OUTPUT_PATH)
    return  all_IDs, all_words


def in_how_many_docs_the_word_appear(X_LEN_OUTPUT_PATH, X_OUTPUT_PATH, DOCS_PATH, output_path):
    all_IDs, all_words = get_IDs_and_words(X_LEN_OUTPUT_PATH, X_OUTPUT_PATH)
    df = pd.read_excel(DOCS_PATH)
    # Drop the first and fourth columns and stay only "תוכן הקובץ" and "מזהה/מספר הקובץ"
    df = df.drop(columns=[df.columns[0], df.columns[3]])
    words_dict = dict.fromkeys(all_words, 0)
    for doc_id in all_IDs:
        if doc_id in df['מזהה/מספר הקובץ'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['מזהה/מספר הקובץ'] == doc_id, 'תוכן הקובץ'].iloc[0]
            doc_as_list = content_value.split()
            current_doc_dict = dict.fromkeys(doc_as_list, 0)
            for word in doc_as_list:
                if current_doc_dict[word] == 0:
                    words_dict[word] = words_dict[word] + 1
                    current_doc_dict[word] = 1

    df = pd.DataFrame(list(words_dict.items()), columns=['Word', 'Count'])
    df.to_excel(output_path, index=False)

def list_to_dict(input_list):
    # Using dict.fromkeys to create a dictionary with default values (None)
    result_dict = dict.fromkeys(input_list, 0)
    return result_dict


def main():
    # A_frequency_result = get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_OUTPUT_PATH)
    # B_frequency_result = get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_OUTPUT_PATH)
    # C_frequency_result = get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_OUTPUT_PATH)

    # get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_LEN_OUTPUT_PATH)
    # get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_LEN_OUTPUT_PATH)
    # get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_LEN_OUTPUT_PATH)

    # x= list_of_words(A_OUTPUT_PATH, A_LEN_OUTPUT_PATH, ". אלו הן רק חלק קטן מהבעיות שאיתם תאלץ להתמודד הממשלה החדשה ")
    # create_vec(A_LEN_OUTPUT_PATH, A_OUTPUT_PATH, FILE_PATH, 1461289)

    in_how_many_docs_the_word_appear(A_LEN_OUTPUT_PATH, A_OUTPUT_PATH, FILE_PATH, "C:\\Users\\nehor\\Downloads\\A_in_how_many_docs_the_word_appear.xlsx")
    in_how_many_docs_the_word_appear(B_LEN_OUTPUT_PATH, B_OUTPUT_PATH, FILE_PATH, "C:\\Users\\nehor\\Downloads\\B_in_how_many_docs_the_word_appear.xlsx")
    in_how_many_docs_the_word_appear(C_LEN_OUTPUT_PATH, C_OUTPUT_PATH, FILE_PATH, "C:\\Users\\nehor\\Downloads\\C_in_how_many_docs_the_word_appear.xlsx")
    print("Word frequency in the specified range:")


if __name__ == "__main__":
    main()