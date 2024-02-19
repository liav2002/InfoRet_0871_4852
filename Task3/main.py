import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import json


def analyze_sentiment_with_progress(text, pbar, result_dict):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Update the progress bar by 1
    pbar.update(1)

    # Update result dictionary
    result_dict[sentiment] += 1

    return sentiment, compound_score, scores['neg'], scores['pos'], scores['neu']


def process_excel_file(file_path, result_dict):
    df = pd.read_excel(file_path)

    # Initialize tqdm with the total number of rows
    with tqdm(total=len(df), desc=f"Processing {file_path}", bar_format="{l_bar}{bar:10}{r_bar}", colour='blue') as pbar:
        # Apply sentiment analysis for each row
        df[['Sentiment', 'Compound_Score', 'Neg_Score', 'Pos_Score', 'Neu_Score']] = df['content'].apply(
            lambda x: pd.Series(analyze_sentiment_with_progress(x, pbar, result_dict)))

    # Saving to a new file with tagged sentiments
    group_id = file_path.split('_')[-1][0]
    output_file_path = f"./output/tagged_translated_15000_docs/tagged_translated_{group_id}.xlsx"
    df.to_excel(output_file_path, index=False)

    return output_file_path


def main():
    file_paths = [
        "./data/translated_15000_docs/translated_A.xlsx",
        "./data/translated_15000_docs/translated_B.xlsx",
        "./data/translated_15000_docs/translated_C.xlsx"
    ]

    for file_path in file_paths:
        group_id = file_path.split('_')[-1][0]
        result_dict = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        output_file_path = process_excel_file(file_path, result_dict)
        with open(f"./output/tagged_translated_15000_docs/result_{group_id}.json", 'w') as json_file:
            json.dump(result_dict, json_file)

    print("Sentiment analysis completed for all files.")


if __name__ == "__main__":
    main()
