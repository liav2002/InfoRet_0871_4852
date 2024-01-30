from Utills import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from time import sleep

# Source Data Input Files
INPUT_SOURCE_A = "./Task1/input/source_data/original_A.xlsx"
INPUT_SOURCE_B = "./Task1/input/source_data/original_B.xlsx"
INPUT_SOURCE_C = "./Task1/input/source_data/original_C.xlsx"

# Output Destination JSONs
OUTPUT_A_JSON = "./Task1/output/d2c_on_source/source_A_vec.json"
OUTPUT_B_JSON = "./Task1/output/d2c_on_source/source_B_vec.json"
OUTPUT_C_JSON = "./Task1/output/d2c_on_source/source_C_vec.json"

# Output Destination Excels
OUTPUT_A_EXCEL = "./Task1/output/d2c_on_source/source_A_vec.xlsx"
OUTPUT_B_EXCEL = "./Task1/output/d2c_on_source/source_B_vec.xlsx"
OUTPUT_C_EXCEL = "./Task1/output/d2c_on_source/source_C_vec.xlsx"


def generate_document_vector(doc):
    # TODO: generate vector using d2v
    raise Exception("generate_document_vector not implemented yet by Doc2Vec Model.")



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


def json2excel(input_file="", output_file=""):
    # Read the JSON file into a dictionary
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Create a DataFrame from the vectors
    df = pd.DataFrame(data.values())

    # Write the DataFrame to an Excel file
    df.to_excel(output_file, header=False, index=False)


def main():
    # Create doc vectors from source data using doc2vec model.
    if all_files_exist([INPUT_SOURCE_A, INPUT_SOURCE_B, INPUT_SOURCE_C]) and not all_files_exist(
            [OUTPUT_A_JSON, OUTPUT_B_JSON, OUTPUT_C_JSON]):
        print("Creating document vectors from source data using doc2vec model.")

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
