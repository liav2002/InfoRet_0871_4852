from Utills import *
import pandas as pd
from collections import Counter
import math


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




def get_voca(dcps_x_path, x_clean_voca):
    df = pd.read_excel(dcps_x_path)
    column_values = df["content"]
    all_text = ' '.join(map(str, column_values))
    word_frequency = Counter(all_text.split())
    result_list = list(word_frequency.items())
    result_list.sort(key=lambda x: x[1], reverse=True)
    df_output = pd.DataFrame(result_list, columns=["Word", "Count"])
    df_output.to_excel(x_clean_voca, index=False)


def count_words_in_cell(cell_value):
    # Count the number of words in a cell
    words = cell_value.split()
    return len(words)


def get_len(dcps_x_path, x_clean_len):
    # Read Excel file using pandas
    df = pd.read_excel(dcps_x_path)
    # Create a new DataFrame with only the required columns
    result_df = pd.DataFrame(columns=["file_id", "paragraph number of words"])
    # Iterate through the DataFrame and count words in each cell
    for index, row in df.iterrows():
        paragraph_id = row["file_id"]
        cell_value = row["content"]
        word_count = count_words_in_cell(str(cell_value))  # Assuming you have a count_words_in_cell function
        # Append the results to the new DataFrame
        result_df = pd.concat(
            [result_df, pd.DataFrame({"file_id": [paragraph_id], "paragraph number of words": [word_count]})],
            ignore_index=True)
    # Write the new DataFrame to a new Excel file
    result_df.to_excel(x_clean_len, index=False)


def excel_column_to_list(excel_file):
    df = pd.read_excel(excel_file)
    first_column_values = df.iloc[:, 0].tolist()
    return first_column_values


def get_IDs_and_words(x_clean_len, x_clean_voca):
    all_words = excel_column_to_list(x_clean_voca)
    all_IDs = excel_column_to_list(x_clean_len)
    return  all_IDs, all_words


def get_appearances(x_clean_len, x_clean_voca, dcps_x_path, x_clean_appearances):
    all_IDs, all_words = get_IDs_and_words(x_clean_len, x_clean_voca)
    df = pd.read_excel(dcps_x_path)
    # Drop the first and fourth columns and stay only "תוכן הקובץ" and "מזהה/מספר הקובץ"
    df = df.drop(columns=[df.columns[0]])
    words_dict = dict.fromkeys(all_words, 0)
    for doc_id in all_IDs:
        if doc_id in df['file_id'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['file_id'] == doc_id, 'content'].iloc[0]
            doc_as_list = content_value.split()
            current_doc_dict = dict.fromkeys(doc_as_list, 0)
            for word in doc_as_list:
                if current_doc_dict[word] == 0:
                    words_dict[word] = words_dict[word] + 1
                    current_doc_dict[word] = 1

    df = pd.DataFrame(list(words_dict.items()), columns=['Word', 'Count'])
    df.to_excel(x_clean_appearances, index=False)


def list_to_dict(input_list):
    # Using dict.fromkeys to create a dictionary with default values (None)
    result_dict = dict.fromkeys(input_list, 0)
    return result_dict


def generate_tfidf_vectors_and_save_2_excel(x_clean_len, x_clean_voca, x_appearances, dcps_a_path, output_x_excel):
    all_IDs, all_words = get_IDs_and_words(x_clean_len, x_clean_voca)
    df = pd.read_excel(dcps_a_path)
    # Drop the first and fourth columns and stay only "תוכן הקובץ" and "מזהה/מספר הקובץ"
    df = df.drop(columns=[df.columns[0]])
    avgl = get_avgl(x_clean_len, all_IDs)
    ids_dict = {doc_id: {} for doc_id in all_IDs}
    appearances_dict = excel_to_dict(x_appearances)
    for doc_id in all_IDs:
        if doc_id in df['file_id'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['file_id'] == doc_id, 'content'].iloc[0]
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

    print("Finish to create the dictionary.")
    print("Try to save the excel matrix.")
    #++++++++++++++++++++++++++++++++++++++ Recommended option, A lot of RAM is required ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    df_matrix = pd.DataFrame.from_dict(ids_dict, orient='index')
    # Transpose the DataFrame
    df_transposed = df_matrix.transpose()
    # Write transposed DataFrame to Excel file
    df_transposed.to_excel(output_x_excel, index=True)
    # ++++++++++++++++++++++++++++++++++++++ Recommended option, A lot of RAM is required ++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #++++++++++++++++++++++++++++++++++++++ Alternation option ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # df = pd.DataFrame(list(ids_dict.items()), columns=['file_id', 'content'])
    # df.to_excel(OUTPUT_X_EXCEL, index=False)
    #++++++++++++++++++++++++++++++++++++++ Alternation option ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print(f"Result saved in: {output_x_excel}")


def create_nested_dict(external_list, inner_list):
    nested_dict = {}
    for key in external_list:
        nested_dict[key] = {inner_key: None for inner_key in inner_list}
    return nested_dict


def get_avgl(x_clean_name, all_IDs):
    sum = 0
    df = pd.read_excel(x_clean_name)
    for doc_id in all_IDs:
        if doc_id in df['file_id'].values:
            # Get the corresponding value in the "תוכן הקובץ" column
            content_value = df.loc[df['file_id'] == doc_id, 'paragraph number of words'].iloc[0]
            sum = sum + content_value
    return sum/(len(all_IDs))


def excel_to_dict(x_appearances):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(x_appearances)

    # Create a dictionary from the two columns
    result_dict = dict(zip(df['Word'], df['Count']))

    return result_dict


def main():
    if not all_files_exist([A_CLEAN_VOCA, B_CLEAN_VOCA, C_CLEAN_VOCA]):
        get_voca(DCPS_A_PATH, A_CLEAN_VOCA)
        get_voca(DCPS_B_PATH, B_CLEAN_VOCA)
        get_voca(DCPS_C_PATH, C_CLEAN_VOCA)

    if not all_files_exist([A_CLEAN_LEN, B_CLEAN_LEN, C_CLEAN_LEN]):
        get_len(DCPS_A_PATH, A_CLEAN_LEN)
        get_len(DCPS_B_PATH, B_CLEAN_LEN)
        get_len(DCPS_C_PATH, C_CLEAN_LEN)

    if not all_files_exist([A_CLEAN_APPEARANCES, B_CLEAN_APPEARANCES, C_CLEAN_APPEARANCES]):
        get_appearances(A_CLEAN_LEN, A_CLEAN_VOCA, DCPS_A_PATH, A_CLEAN_APPEARANCES)
        get_appearances(B_CLEAN_LEN, B_CLEAN_VOCA, DCPS_B_PATH, B_CLEAN_APPEARANCES)
        get_appearances(C_CLEAN_LEN, C_CLEAN_VOCA, DCPS_C_PATH, C_CLEAN_APPEARANCES)

    # try run with the recommended option in generate_tfidf_vectors_and_save_2_excel().
    generate_tfidf_vectors_and_save_2_excel(A_CLEAN_LEN, A_CLEAN_VOCA, A_CLEAN_APPEARANCES, DCPS_A_PATH, OUTPUT_A_EXCEL)
    generate_tfidf_vectors_and_save_2_excel(B_CLEAN_LEN, B_CLEAN_VOCA, B_CLEAN_APPEARANCES, DCPS_B_PATH, OUTPUT_B_EXCEL)
    generate_tfidf_vectors_and_save_2_excel(C_CLEAN_LEN, C_CLEAN_VOCA, C_CLEAN_APPEARANCES, DCPS_C_PATH, OUTPUT_C_EXCEL)

    print("+++++finish+++++")


if __name__ == "__main__":
    main()