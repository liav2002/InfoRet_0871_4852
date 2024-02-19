import pandas as pd
from transformers import pipeline
from tokenizers.decoders import WordPiece
import random
from tqdm import tqdm

# Load the NER pipeline
oracle = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')
# Set tokenizer decoder for the aggregation strategy
oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

def detect_person_names(text):
    entities = oracle(text)
    person_names = [entity['word'] for entity in entities if entity['entity_group'] == 'PER']
    return person_names

def generate_random_name(first_names, last_names):
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    return f"{first_name} {last_name}"

def clean_text_from_names(text, first_names, last_names, pbar):
    person_names = detect_person_names(text)
    for name in person_names:
        if ' ' in name:
            new_name = generate_random_name(first_names, last_names)
        else:
            choice = random.choice([1, 2])
            if choice == 1:
                new_name = random.choice(first_names)
            else:
                new_name = random.choice(last_names)
        text = text.replace(name, new_name)
    pbar.update(1)  # Update progress bar by 1
    return text

def clean_source_file(file_path, first_names, last_names):
    df = pd.read_excel(file_path)
    with tqdm(total=len(df), desc=f"Processing {file_path}") as pbar:
        df['content'] = df['content'].apply(lambda x: clean_text_from_names(x, first_names, last_names, pbar))
    return df

def main():
    # Load first names and last names
    first_names_df = pd.read_excel("./data/names/first-names.xlsx")
    last_names_df = pd.read_excel("./data/names/last-names.xlsx")
    first_names = first_names_df['first_name'].tolist()
    last_names = last_names_df['last_name'].tolist()

    # Process source files
    source_files = [
        "./data/15000_docs/original_A.xlsx",
        "./data/15000_docs/original_B.xlsx",
        "./data/15000_docs/original_C.xlsx"
    ]

    for file_path in source_files:
        group_id = file_path.split('_')[-1][0]
        cleaned_df = clean_source_file(file_path, first_names, last_names)
        cleaned_df.to_excel(f"./data/cleaned_from_names_15000_docs/without_names_{group_id}.xlsx", index=False)

    print("Cleaning process completed.")

if __name__ == "__main__":
    main()
