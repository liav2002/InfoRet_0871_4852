import string

from transformers import BertTokenizer, BertModel
from Utills import *
from tqdm import tqdm
from time import sleep
import pandas as pd
import numpy as np
import torch
import json

# Source Data Input Files
INPUT_SOURCE_A = "./Task1/input/source_data/original_A.xlsx"
INPUT_SOURCE_B = "./Task1/input/source_data/original_B.xlsx"
INPUT_SOURCE_C = "./Task1/input/source_data/original_C.xlsx"

# Output Destination JSONs
OUTPUT_A_JSON = "./Task1/output/bert_on_source/source_A_vec.json"
OUTPUT_B_JSON = "./Task1/output/bert_on_source/source_B_vec.json"
OUTPUT_C_JSON = "./Task1/output/bert_on_source/source_C_vec.json"

# Output Destination Excels
OUTPUT_A_EXCEL = "./Task1/output/bert_on_source/source_A_vec.xlsx"
OUTPUT_B_EXCEL = "./Task1/output/bert_on_source/source_B_vec.xlsx"
OUTPUT_C_EXCEL = "./Task1/output/bert_on_source/source_C_vec.xlsx"

# AlephBertGimel Model Path
ABG_Model_Path = "C:/Users/liavm/OneDrive - g.jct.ac.il/Year D/Annual courses/Bert2Vec - Final Project/Task 4 - Find Threashold/HWD_2_diffDS/alephbertgimmel-base/ckpt_73780--Max512Seq"


def organize_hidden_states(vectors):
    vectors_as_list = []
    for vec in vectors:
        vectors_as_list.append(torch.detach(vec).numpy())
    token_embeddings = np.stack(vectors_as_list, axis=0)
    token_embeddings = np.squeeze(token_embeddings, axis=1)
    token_embeddings = np.transpose(token_embeddings, axes=(1, 0, 2))

    # `token_embeddings` is a [N x 12 x 768].
    v_list = []
    for v in token_embeddings:
        v_list.append(v[-2])
    return np.array(v_list)


def load_ABG_model(model_path=ABG_Model_Path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path, output_hidden_states=True)

    # if not fine-tuning - disable dropout
    model.eval()

    return model, tokenizer


def get_vector_of_sentence(model, tokenizer, sentence):
    output = model(tokenizer.encode(sentence, return_tensors='pt'))

    hidden_states = output[2]

    vectors = organize_hidden_states(hidden_states)

    indexed_tokens = tokenizer.encode(sentence)
    tokenized_text = tokenizer.convert_ids_to_tokens(indexed_tokens)

    return vectors[tokenized_text.index('[CLS]')]


def generate_document_vector(text, model, tokenizer):
    #  Initialize an empty vector for the document
    document_vector = np.zeros((768,))

    # Split text into sentences
    sentences = text.split('\n')

    # Process each sentence
    for sentence in sentences:
        # Calculate and add vectors for each sentence
        try:
            current_vector = get_vector_of_sentence(model, tokenizer, sentence)
            document_vector += current_vector
        except Exception as e :
            print(f"Error accrued in sentence: {sentence}")
            print(f"Error message: {e}")
            print("Pass this sentence and continue with others.\n")

    return document_vector.tolist()


def create_json_file(df, output_path):
    result_dict = {}

    # Load Model
    print("Loading Aleph Bert Gimel Model...")
    model, tokenizer = load_ABG_model()
    print("Model load successfully.")
    sleep(0.1)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Documents"):
        file_id = row['file_id']
        content = row['content']

        document_vector = generate_document_vector(content, model, tokenizer)

        if document_vector:
            result_dict[file_id] = document_vector

    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Output save in: {output_path}.")


def json2excel(input_file="", output_file=""):
    # Read the JSON file into a dictionary
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Create a DataFrame from the vectors
    df = pd.DataFrame(data.values())

    # Write the DataFrame to an Excel file
    df.to_excel(output_file, header=False, index=False)


def main():
    # Create doc vectors from source data using AlephBertGimel model.
    if all_files_exist([INPUT_SOURCE_A, INPUT_SOURCE_B, INPUT_SOURCE_C]) and not all_files_exist(
            [OUTPUT_A_JSON, OUTPUT_B_JSON, OUTPUT_C_JSON]):
        print("Creating document vectors from source data using AlephBertGimel model.")

        # Load DataFrames
        df_A = pd.read_excel(INPUT_SOURCE_A)
        df_B = pd.read_excel(INPUT_SOURCE_B)
        df_C = pd.read_excel(INPUT_SOURCE_C)

        # Create JSON files for each group
        print("Working on group A")
        sleep(0.1)
        create_json_file(df_A, OUTPUT_A_JSON)
        print("\nWorking on group B")
        sleep(0.1)
        create_json_file(df_B, OUTPUT_B_JSON)
        sleep(0.1)
        print("\nWorking on group C")
        create_json_file(df_C, OUTPUT_C_JSON)
        sleep(0.1)

        # Generate Group A JSON to Excel
        print(f"\nTry to save {OUTPUT_A_JSON} as {OUTPUT_A_EXCEL}...")
        json2excel(OUTPUT_A_JSON, OUTPUT_A_EXCEL)
        print(f"File successfully saved in {OUTPUT_A_EXCEL}.\n")

        # Generate Group B JSON to Excel
        print(f"Try to save {OUTPUT_B_JSON} as {OUTPUT_B_EXCEL}...")
        json2excel(OUTPUT_B_JSON, OUTPUT_B_EXCEL)
        print(f"File successfully saved in {OUTPUT_B_EXCEL}.\n")

        # Generate Group C JSON to Excel
        print(f"Try to save {OUTPUT_C_JSON} as {OUTPUT_C_EXCEL}...")
        json2excel(OUTPUT_C_JSON, OUTPUT_C_EXCEL)
        print(f"File successfully saved in {OUTPUT_C_EXCEL}.\n")


if __name__ == "__main__":
    main()
