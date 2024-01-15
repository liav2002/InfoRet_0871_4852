from Utills import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from time import sleep

# Data Cleaned From Punctuations and Stop-words Input
DCPS_A_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_A.xlsx"
DCPS_B_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_B.xlsx"
DCPS_C_PATH = "./input/data_cleaned_from_punctuations_and_stopwords/dcps_C.xlsx"

# Data Cleaned From Punctuations and Stop-words Output
OUTPUT_DCPS_A = "./output/w2v_on_dcps/dcps_A_vec.json"
OUTPUT_DCPS_B = "./output/w2v_on_dcps/dcps_B_vec.json"
OUTPUT_DCPS_C = "./output/w2v_on_dcps/dcps_C_vec.json"

# Data Cleaned From Punctuations Input
DCP_A_PATH = "./input/data_cleaned_from_punctuations/dcp_A.xlsx"
DCP_B_PATH = "./input/data_cleaned_from_punctuations/dcp_B.xlsx"
DCP_C_PATH = "./input/data_cleaned_from_punctuations/dcp_C.xlsx"

# Data Cleaned From Punctuations Output
OUTPUT_DCP_A = "./output/w2v_on_dcp/dcp_A_vec.json"
OUTPUT_DCP_B = "./output/w2v_on_dcp/dcp_B_vec.json"
OUTPUT_DCP_C = "./output/w2v_on_dcp/dcp_C_vec.json"

# Data With Lemot Only Input
DWLO_A_PATH = "./input/data_with_lemot_only/dwlo_A.xlsx"
DWLO_B_PATH = "./input/data_with_lemot_only/dwlo_B.xlsx"
DWLO_C_PATH = "./input/data_with_lemot_only/dwlo_C.xlsx"

# Data With Lemot Only Output
OUTPUT_DWLO_A = "./output/w2v_on_dwlo/dwlo_A_vec.json"
OUTPUT_DWLO_B = "./output/w2v_on_dwlo/dwlo_B_vec.json"
OUTPUT_DWLO_C = "./output/w2v_on_dwlo/dwlo_C_vec.json"

# Word2Vec Model Path
W2V_VECTORS = "./model/w2v/words_vectors.npy"
W2V_VOCAB = "./model/w2v/words_list.txt"


def generate_document_vector(text, vocab=W2V_VOCAB, stored_vectors=W2V_VECTORS):
    # Split text into sentences
    sentences = text.split('\n')

    # Load words from words.txt
    with open(vocab, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]

    # Load word vectors from words_vectors.npy
    word_vectors = np.load(stored_vectors)

    # Initialize an empty vector for the document
    document_vector = np.zeros_like(word_vectors[0])

    # Process each sentence
    for sentence in sentences:
        # Split sentence into words
        words_in_sentence = sentence.split(' ')

        # Calculate and add vectors for each word
        for word in words_in_sentence:
            if word in words:
                # Find the index of the word in words.txt
                index = words.index(word)
                # Add the vector corresponding to the index
                document_vector += word_vectors[index]

    return document_vector.tolist()


def create_json_file(df, output_path):
    result_dict = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Documents"):
        file_id = row['file_id']
        content = row['content']

        document_vector = generate_document_vector(content)

        if document_vector:
            result_dict[file_id] = document_vector

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Output save in: {output_path}.")


def main():
    # Create doc vectors from data without punctuations and stop-words.
    # if all_files_exist([DCPS_A_PATH, DCPS_B_PATH, DCPS_C_PATH]):
    #     print("Creating document's vectors from DCPS_DATA using Word2Vec.")
    #
    #     # Load DataFrames
    #     df_A = pd.read_excel(DCPS_A_PATH)
    #     df_B = pd.read_excel(DCPS_B_PATH)
    #     df_C = pd.read_excel(DCPS_C_PATH)
    #
    #     # Create JSON files for each group
    #     print("Working on group A")
    #     sleep(0.1)
    #     create_json_file(df_A, OUTPUT_DCPS_A)
    #     print("Working on group B")
    #     sleep(0.1)
    #     create_json_file(df_B, OUTPUT_DCPS_B)
    #     sleep(0.1)
    #     print("Working on group C")
    #     create_json_file(df_C, OUTPUT_DCPS_C)

    # Create doc vectors from data without punctuations.
    if all_files_exist([DCP_A_PATH, DCP_B_PATH, DCP_C_PATH]):
        print("Creating document's vectors from DCP_DATA using Word2Vec.")

        # Load DataFrames
        df_A = pd.read_excel(DCP_A_PATH)
        df_B = pd.read_excel(DCP_B_PATH)
        df_C = pd.read_excel(DCP_C_PATH)

        # Create JSON files for each group
        print("Working on group A")
        sleep(0.1)
        create_json_file(df_A, OUTPUT_DCP_A)
        print("Working on group B")
        sleep(0.1)
        create_json_file(df_B, OUTPUT_DCP_B)
        print("Working on group C")
        sleep(0.1)
        create_json_file(df_C, OUTPUT_DCP_C)

    # Create doc vectors from data with lemot.
    if all_files_exist([DWLO_A_PATH, DWLO_B_PATH, DWLO_C_PATH]):
        print("Creating document's vectors from DWLO_DATA using Word2Vec.")

        # Load DataFrames
        df_A = pd.read_excel(DWLO_A_PATH)
        df_B = pd.read_excel(DWLO_B_PATH)
        df_C = pd.read_excel(DWLO_C_PATH)

        # Create JSON files for each group
        print("Working on group A")
        sleep(0.1)
        create_json_file(df_A, OUTPUT_DWLO_A)
        print("Working on group B")
        sleep(0.1)
        create_json_file(df_B, OUTPUT_DWLO_B)
        print("Working on group C")
        sleep(0.1)
        create_json_file(df_C, OUTPUT_DWLO_C)


if __name__ == "__main__":
    main()
