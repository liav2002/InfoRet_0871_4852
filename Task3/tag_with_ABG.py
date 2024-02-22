import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


def analyze_sentiment_with_ABG(text, pbar, result_dict):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Predict sentiment
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1).tolist()[0]

    # Update the progress bar by 1
    pbar.update(1)

    # Extract positive and negative scores
    pos_score = scores[0]  # Label 0 corresponds to positive sentiment
    neg_score = scores[1]  # Label 1 corresponds to negative sentiment

    return pos_score, neg_score


def process_excel_file_with_ABG(file_path, file_id, result_dict, isOriginalData=True):
    df = pd.read_excel(file_path)

    # Initialize tqdm with the total number of rows
    with tqdm(total=len(df), desc=f"Processing {file_path}", bar_format="{l_bar}{bar:10}{r_bar}",
              colour='blue') as pbar:
        # Apply sentiment analysis for each row
        df['Pos_Score'], df['Neg_Score'] = zip(*df['content'].apply(lambda x: analyze_sentiment_with_ABG(x, pbar, result_dict)))

    # Determine sentiment based on scores
    df['Sentiment'] = df.apply(lambda row: 'positive' if row['Pos_Score'] > row['Neg_Score'] else 'negative', axis=1)

    # Saving to a new file with tagged sentiments
    df['File_ID'] = file_id

    if isOriginalData:
        df.to_excel(f"./output/tagged_ABG_15000_docs/tagged_with_ABG_{file_id}.xlsx", index=False)
    else:
        df.to_excel(f"./output/tagged_ABG_1500_without_names_docs/result_ABG_without_names_{file_id}.xlsx",
                    index=False)

    # Update result dictionary
    result_dict['positive'] += len(df[df['Sentiment'] == 'positive'])
    result_dict['negative'] += len(df[df['Sentiment'] == 'negative'])

    # Write results to JSON file
    if isOriginalData:
        with open(f"./output/tagged_ABG_15000_docs/result_ABG_{file_id}.json", 'w') as json_file:
            json.dump(result_dict, json_file)
    else:
        with open(f"./output/tagged_ABG_1500_without_names_docs/result_ABG_without_names_{file_id}.json", 'w') as json_file:
            json.dump(result_dict, json_file)


def main():
    groups = ["A", "B", "C"]

    # Create output for original data.
    print("Working on original data.")

    file_paths = [
        "./data/15000_docs/original_A.xlsx",
        "./data/15000_docs/original_B.xlsx",
        "./data/15000_docs/original_C.xlsx"
    ]

    i = 0

    for file_path in file_paths:
        result_dict = {'positive': 0, 'negative': 0, 'neutral': 0}
        process_excel_file_with_ABG(file_path=file_path, file_id=groups[i], result_dict=result_dict, isOriginalData=True)
        i += 1

    print("Sentiment analysis completed for all original files.\n")

    # Create output for data after NER
    print("Working on data without names.")

    file_paths = [
        "./data/cleaned_from_names_15000_docs/without_names_A.xlsx",
        "./data/cleaned_from_names_15000_docs/without_names_B.xlsx",
        "./data/cleaned_from_names_15000_docs/without_names_C.xlsx"
    ]

    i = 0

    for file_path in file_paths:
        result_dict = {'positive': 0, 'negative': 0, 'neutral': 0}
        process_excel_file_with_ABG(file_path=file_path, file_id=groups[i], result_dict=result_dict, isOriginalData=False)
        i += 1

    print("Sentiment analysis completed for all files without names.")


if __name__ == "__main__":
    # Load the AlephBertGimmel-Sentiment model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")

    main()
