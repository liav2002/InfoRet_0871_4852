import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel, pipeline


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
    with tqdm(total=len(df), desc=f"Processing {file_path}", bar_format="{l_bar}{bar:10}{r_bar}",
              colour='blue') as pbar:
        # Apply sentiment analysis for each row
        df[['Sentiment', 'Compound_Score', 'Neg_Score', 'Pos_Score', 'Neu_Score']] = df['content'].apply(
            lambda x: pd.Series(analyze_sentiment_with_progress(x, pbar, result_dict)))

    # Saving to a new file with tagged sentiments
    group_id = file_path.split('_')[-1][0]
    output_file_path = f"./output/tagged_translated_15000_docs/tagged_translated_{group_id}.xlsx"
    df.to_excel(output_file_path, index=False)

    return output_file_path


def analyze_sentiment_3(input_file, output_file, sentiment_analysis, n):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Drop the first column
    df = df.drop(columns=['No.'])

    # Initialize lists to store sentiment scores
    natural_scores = []
    positive_scores = []
    negative_scores = []

    # Perform sentiment analysis on the 'content' column for the first n rows
    for i in range(min(n, len(df))):
        content = df.loc[i, 'content']
        # Check if the content exceeds the maximum input length
        max_length = 512  # Example: Maximum input length for BERT-base
        if len(content) > max_length:
            # Truncate the content if it's too long
            content = content[:max_length]
        # Perform sentiment analysis on the truncated content
        sentiment_result = sentiment_analysis(content)
        # Extract sentiment scores
        natural_score = sentiment_result[0][0]['score']
        positive_score = sentiment_result[0][1]['score']
        negative_score = sentiment_result[0][2]['score']
        # Append scores to the lists
        natural_scores.append(natural_score)
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)

    # Determine the dominant sentiment for each row
    sentiments = []
    for i in range(min(n, len(df))):
        pos_score = positive_scores[i]
        neg_score = negative_scores[i]
        neu_score = natural_scores[i]
        if pos_score > neg_score and pos_score > neu_score:
            sentiment = "Positive"
        elif neg_score > pos_score and neg_score > neu_score:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments.append(sentiment)

    # Create a DataFrame with 'file_id', 'content', sentiment scores, and sentiment
    result_df = pd.DataFrame({
        'file_id': df['file_id'].head(n),
        'content': df['content'].head(n),
        'Neu_Score': natural_scores,
        'Pos_score': positive_scores,
        'Neg_score': negative_scores,
        'Sentiment': sentiments
    })

    # Save the DataFrame to a new Excel file
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


def create_sentiment_analysis():
    tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis")  # same as 'avichr/heBERT' tokenizer
    model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")
    sentiment_analysis = pipeline(
        "sentiment-analysis",
        model="avichr/heBERT_sentiment_analysis",
        tokenizer="avichr/heBERT_sentiment_analysis",
        return_all_scores=True
    )
    return sentiment_analysis


def count_sentiments(input_file, output_file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Count the number of each sentiment category in the "Sentiment" column
    sentiment_counts = df['Sentiment'].value_counts().to_dict()

    # Convert the counts to JSON format
    sentiment_json = json.dumps(sentiment_counts)

    # Save the JSON data to the output file
    with open(output_file, 'w') as json_file:
        json_file.write(sentiment_json)

    print(f"Sentiment counts saved to {output_file}")


def main():
    sentiment_analysis = create_sentiment_analysis()
    analyze_sentiment_3("./data/15000_docs/original_A.xlsx",
                        "./output/tagged_heBert_15000_docs/tagged_with_heBert_A.xlsx", sentiment_analysis, 5000)
    analyze_sentiment_3("./data/15000_docs/original_B.xlsx",
                        "./output/tagged_heBert_15000_docs/tagged_with_heBert_B.xlsx", sentiment_analysis, 5000)
    analyze_sentiment_3("./data/15000_docs/original_C.xlsx",
                        "./output/tagged_heBert_15000_docs/tagged_with_heBert_C.xlsx", sentiment_analysis, 5000)

    count_sentiments("./output/tagged_heBert_15000_docs/tagged_with_heBert_A.xlsx",
                     "./output/tagged_heBert_15000_docs/result_heBert_A.json")
    count_sentiments("./output/tagged_heBert_15000_docs/tagged_with_heBert_B.xlsx",
                     "./output/tagged_heBert_15000_docs/result_heBert_B.json")
    count_sentiments("./output/tagged_heBert_15000_docs/tagged_with_heBert_C.xlsx",
                     "./output/tagged_heBert_15000_docs/result_heBert_C.json")

    analyze_sentiment_3("./data/cleaned_from_names_15000_docs/without_names_A.xlsx",
                        "./output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_A.xlsx",
                        sentiment_analysis, 5000)
    analyze_sentiment_3("./data/cleaned_from_names_15000_docs/without_names_B.xlsx",
                        "./output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_B.xlsx",
                        sentiment_analysis, 5000)
    analyze_sentiment_3("./data/cleaned_from_names_15000_docs/without_names_C.xlsx",
                        "./output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_C.xlsx",
                        sentiment_analysis, 5000)

    count_sentiments("./output/tagged_heBert_15000_without_names_docs/HeBERT_without_names_A_new.xlsx",
                     "./output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_A.json")
    count_sentiments("./output/tagged_heBert_15000_without_names_docs/HeBERT_without_names_B_new.xlsx",
                     "./output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_B.json")
    count_sentiments("./output/tagged_heBert_15000_without_names_docs/HeBERT_without_names_C_new.xlsx",
                     "./output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_C.json")


if __name__ == "__main__":
    main()
