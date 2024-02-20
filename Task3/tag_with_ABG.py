import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json


def analyze_sentiment_with_ABG(text, pbar, result_dict):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Predict sentiment
    outputs = model(**inputs)
    predicted_label = pipeline_task_dict[outputs['logits'].argmax().item()]

    # Update the progress bar by 1
    pbar.update(1)

    # Update result dictionary
    result_dict[predicted_label] += 1

    return predicted_label


def process_excel_file_with_ABG(file_path, result_dict, isOriginalData=True):
    df = pd.read_excel(file_path)

    # Initialize tqdm with the total number of rows
    with tqdm(total=len(df), desc=f"Processing {file_path}", bar_format="{l_bar}{bar:10}{r_bar}",
              colour='blue') as pbar:
        # Apply sentiment analysis for each row
        df['Sentiment'] = df['content'].apply(lambda x: analyze_sentiment_with_ABG(x, pbar, result_dict))

    # Saving to a new file with tagged sentiments
    group_id = file_path.split('_')[-1][0]

    if isOriginalData:
        df.to_excel(f"./output/tagged_ABG_15000_docs/tagged_with_ABG_{group_id}.xlsx", index=False)
    else:
        df.to_excel(f"./output/tagged_ABG_1500_without_names_docs/result_ABG_without_names_{group_id}.xlsx",
                    index=False)


def main():
    # Create output for original data.
    print("Working on original data.")

    file_paths = [
        "./data/15000_docs/original_A.xlsx",
        "./data/15000_docs/original_B.xlsx",
        "./data/15000_docs/original_C.xlsx"
    ]

    for file_path in file_paths:
        group_id = file_path.split('_')[-1][0]
        result_dict = {'positive': 0, 'negative': 0, 'neutral': 0}
        process_excel_file_with_ABG(file_path=file_path, result_dict=result_dict, isOriginalData=True)
        with open(f"./output/tagged_ABG_15000_docs/result_ABG_{group_id}.json", 'w') as json_file:
            json.dump(result_dict, json_file)

    print("Sentiment analysis completed for all original files.\n")

    # Create output for data after NER
    print("Working on data without names.")

    file_paths = [
        "./data/cleaned_from_names_15000_docs/without_names_A.xlsx",
        "./data/cleaned_from_names_15000_docs/without_names_B.xlsx",
        "./data/cleaned_from_names_15000_docs/without_names_C.xlsx"
    ]

    for file_path in file_paths:
        group_id = file_path.split('_')[-1][0]
        result_dict = {'positive': 0, 'negative': 0, 'neutral': 0}
        process_excel_file_with_ABG(file_path=file_path, result_dict=result_dict, isOriginalData=False)
        with open(f"./output/tagged_ABG_1500_without_names_docs/result_ABG_without_names_{group_id}.json",
                  'w') as json_file:
            json.dump(result_dict, json_file)

    print("Sentiment analysis completed for all files without names.")


if __name__ == "__main__":
    # Load the AlephBertGimmel-Sentiment model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")

    # Define the sentiment labels
    pipeline_task_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

    main()
