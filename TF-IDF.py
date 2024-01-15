import pandas as pd
from collections import Counter
import math
import numpy as np
import string
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


CONTENT_COL  = 2
ID_COL = 1
A_START = 2
A_FINISH = 5001
B_START = 5006
B_FINISH = 10005
C_START = 10010
C_FINISH = 15009
SHEET = "Sheet1"
CLEAN_15000 = ".\\data\\clean_15000.xlsx"
A_CLEAN_VOCA = ".\\temp_files\\A_CLEAN_VOCA.xlsx"
B_CLEAN_VOCA = ".\\temp_files\\B_CLEAN_VOCA.xlsx"
C_CLEAN_VOCA = ".\\temp_files\\C_CLEAN_VOCA.xlsx"
A_CLEAN_LEN = ".\\temp_files\\A_CLEAN_LEN.xlsx"
B_CLEAN_LEN = ".\\temp_files\\B_CLEAN_LEN.xlsx"
C_CLEAN_LEN = ".\\temp_files\\C_CLEAN_LEN.xlsx"
A_CLEAN_APPEARANCES = ".\\temp_files\\A_CLEAN_APPEARANCES.xlsx"
B_CLEAN_APPEARANCES = ".\\temp_files\\B_CLEAN_APPEARANCES.xlsx"
C_CLEAN_APPEARANCES = ".\\temp_files\\C_CLEAN_APPEARANCES.xlsx"
A_CLEAN_MATRIX = ".\\output\\A_CLEAN_MATRIX .xlsx"
B_CLEAN_MATRIX = ".\\output\\B_CLEAN_MATRIX .xlsx"
C_CLEAN_MATRIX = ".\\output\\C_CLEAN_MATRIX .xlsx"


def get_voca(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
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


def get_len(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
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


def excel_column_to_list(excel_file):
    df = pd.read_excel(excel_file)
    first_column_values = df.iloc[:, 0].tolist()
    return first_column_values


def get_IDs_and_words(X_CLEAN_LEN, X_CLEAN_VOCA):
    all_words = excel_column_to_list(X_CLEAN_VOCA)
    all_IDs = excel_column_to_list(X_CLEAN_LEN)
    return  all_IDs, all_words


def in_how_many_docs_the_word_appear(X_CLEAN_LEN, X_CLEAN_VOCA, DOCS_PATH, output_path):
    all_IDs, all_words = get_IDs_and_words(X_CLEAN_LEN, X_CLEAN_VOCA)
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


def create_all_vec(X_CLEAN_LEN, X_CLEAN_VOCA, X_APPEARANCES,  DOCS_PATH, X_CLEAN_MATRIX):
    all_IDs, all_words = get_IDs_and_words(X_CLEAN_LEN, X_CLEAN_VOCA)
    df = pd.read_excel(DOCS_PATH)
    # Drop the first and fourth columns and stay only "תוכן הקובץ" and "מזהה/מספר הקובץ"
    df = df.drop(columns=[df.columns[0], df.columns[3]])
    avgl = get_avgl(X_CLEAN_LEN, all_IDs)
    ids_dict = {doc_id: {} for doc_id in all_IDs}
    appearances_dict = excel_to_dict(X_APPEARANCES)
    for doc_id in all_IDs:
        if doc_id in df['מזהה/מספר הקובץ'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['מזהה/מספר הקובץ'] == doc_id, 'תוכן הקובץ'].iloc[0]
            doc_as_list = content_value.split()
            # first iteration for "Doc frequency" (how many instances of any words exist in the current doc)
            for word in doc_as_list:
                if word in ids_dict[doc_id]:
                    ids_dict[doc_id][word] += 1
                else:
                    ids_dict[doc_id][word] = 1


            # second iteration for TF-IDF
            #  k is a positive constant controlling the term frequency saturation. Typical values are between 1.2 and 2.0. (ChatGPT)
            k = 1.5
            b = 1
            l = len(doc_as_list)
            unique_list = list(filter(lambda x: doc_as_list.count(x) == 1, doc_as_list))
            for word in unique_list:
                TF_IDF = (math.log10(5001/appearances_dict[word]))*ids_dict[doc_id][word]
                normalize = (1-b+(b*l/avgl))
                BM25 = (k+1)/((ids_dict[doc_id][word])+(k*normalize))
                ids_dict[doc_id][word] = round(TF_IDF * BM25, 4)
                # ids_dict[doc_id][word] = round(math.log10(5001/ids_dict[doc_id][word]), 3)

    # df_matrix = pd.DataFrame.from_dict(ids_dict, orient='index')
    # # Transpose the DataFrame
    # df_transposed = df_matrix.transpose()
    # # Write transposed DataFrame to Excel file
    # df_transposed.to_excel(X_CLEAN_MATRIX, index=True)
    return ids_dict


def create_nested_dict(external_list, inner_list):
    nested_dict = {}
    for key in external_list:
        nested_dict[key] = {inner_key: None for inner_key in inner_list}
    return nested_dict


def get_avgl(X_CLEAN_LEN, all_IDs):
    sum = 0
    df = pd.read_excel(X_CLEAN_LEN)
    for doc_id in all_IDs:
        if doc_id in df['מזהה/מספר הקובץ'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['מזהה/מספר הקובץ'] == doc_id, 'paragraph number of words'].iloc[0]
            sum = sum + content_value
    return sum/(len(all_IDs))


def excel_to_dict(X_APPEARANCES):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(X_APPEARANCES)

    # Create a dictionary from the two columns
    result_dict = dict(zip(df['Word'], df['Count']))

    return result_dict


def main():
    # A_frequency_result = get_voca(CLEAN_15000, SHEET, A_START, A_FINISH, CONTENT_COL, A_CLEAN_VOCA)
    # B_frequency_result = get_voca(CLEAN_15000, SHEET, B_START, B_FINISH, CONTENT_COL, B_CLEAN_VOCA)
    # C_frequency_result = get_voca(CLEAN_15000, SHEET, C_START, C_FINISH, CONTENT_COL, C_CLEAN_VOCA)

    # get_len(CLEAN_15000, SHEET, A_START, A_FINISH, CONTENT_COL, A_CLEAN_LEN)
    # get_len(CLEAN_15000, SHEET, B_START, B_FINISH, CONTENT_COL, B_CLEAN_LEN)
    # get_len(CLEAN_15000, SHEET, C_START, C_FINISH, CONTENT_COL, C_CLEAN_LEN)

    # in_how_many_docs_the_word_appear(A_CLEAN_LEN, A_CLEAN_VOCA, CLEAN_15000, A_CLEAN_APPEARANCES)
    # in_how_many_docs_the_word_appear(B_CLEAN_LEN, B_CLEAN_VOCA, CLEAN_15000, B_CLEAN_APPEARANCES)
    # in_how_many_docs_the_word_appear(C_CLEAN_LEN, C_CLEAN_VOCA, CLEAN_15000, C_CLEAN_APPEARANCES)

    # ids_dict = create_all_vec(A_CLEAN_LEN, A_CLEAN_VOCA, A_CLEAN_APPEARANCES, CLEAN_15000, A_CLEAN_MATRIX)
    # print(ids_dict[1461014])
    # ids_dict = create_all_vec(B_CLEAN_LEN, B_CLEAN_VOCA, B_CLEAN_APPEARANCES, CLEAN_15000, B_CLEAN_MATRIX)
    # print(ids_dict[2592098])
    # ids_dict = create_all_vec(C_CLEAN_LEN, C_CLEAN_VOCA, C_CLEAN_APPEARANCES, CLEAN_15000, C_CLEAN_MATRIX)
    # print(ids_dict[3036697])



    print("+++++finish+++++")


if __name__ == "__main__":
    main()