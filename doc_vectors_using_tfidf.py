import pandas as pd
from collections import Counter
import math
import numpy as np
import string
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix


# Data Cleaned From Punctuations and Stop-words Input
DCPS_A_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_A.xlsx"
DCPS_B_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_B.xlsx"
DCPS_C_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_C.xlsx"

# Output Destination Excels
OUTPUT_A_EXCEL = "./output/tfidf_on_dcps/dcps_A_vec.xlsx"
OUTPUT_B_EXCEL = "./output/tfidf_on_dcps/dcps_B_vec.xlsx"
OUTPUT_C_EXCEL = "./output/tfidf_on_dcps/dcps_C_vec.xlsx"

#Temp files for vocabulary.
A_CLEAN_VOCA = "./nehorai_temp_files/A_CLEAN_VOCA.xlsx"
B_CLEAN_VOCA = "./nehorai_temp_files/B_CLEAN_VOCA.xlsx"
C_CLEAN_VOCA = "./nehorai_temp_files/C_CLEAN_VOCA.xlsx"

#Temp files for documents length.
A_CLEAN_LEN = "./nehorai_temp_files/A_CLEAN_LEN.xlsx"
B_CLEAN_LEN = "./nehorai_temp_files/B_CLEAN_LEN.xlsx"
C_CLEAN_LEN = "./nehorai_temp_files/C_CLEAN_LEN.xlsx"

#Temp files for appearances (in how many document the word appear at least one time).
A_CLEAN_APPEARANCES = ".\\nehorai_temp_files\\A_CLEAN_APPEARANCES.xlsx"
B_CLEAN_APPEARANCES = ".\\nehorai_temp_files\\B_CLEAN_APPEARANCES.xlsx"
C_CLEAN_APPEARANCES = ".\\nehorai_temp_files\\C_CLEAN_APPEARANCES.xlsx"




def get_voca(DCPS_X_PATH, X_CLEAN_VOCA):
    df = pd.read_excel(DCPS_X_PATH)
    column_values = df["content"]
    all_text = ' '.join(map(str, column_values))
    word_frequency = Counter(all_text.split())
    result_list = list(word_frequency.items())
    result_list.sort(key=lambda x: x[1], reverse=True)
    df_output = pd.DataFrame(result_list, columns=["Word", "Count"])
    df_output.to_excel(X_CLEAN_VOCA, index=False)


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
                normalize = (1-b+((b*l)/avgl))
                BM25 = (k+1)/((ids_dict[doc_id][word])+(k*normalize))
                ids_dict[doc_id][word] = round((TF_IDF * BM25), 4)
                # ids_dict[doc_id][word] = round(math.log10(5001/ids_dict[doc_id][word]), 3)


    #++++++++++++++++++++++++++++++++++++++ More RAM request ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # df_matrix = pd.DataFrame.from_dict(ids_dict, orient='index')
    # # Transpose the DataFrame
    # df_transposed = df_matrix.transpose()
    # # Write transposed DataFrame to Excel file
    # df_transposed.to_excel(X_CLEAN_MATRIX, index=True)
    #++++++++++++++++++++++++++++++++++++++ More RAM request ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    df = pd.DataFrame(list(ids_dict.items()), columns=['ID', 'Value'])
    df.to_excel(X_CLEAN_MATRIX, index=False)
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
    # get_voca(DCPS_A_PATH, A_CLEAN_VOCA)
    # get_voca(DCPS_B_PATH, B_CLEAN_VOCA)
    # get_voca(DCPS_C_PATH, C_CLEAN_VOCA)

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