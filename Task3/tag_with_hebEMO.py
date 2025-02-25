import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel, pipeline
from HeBERT.src.HebEMO import *

HebEMO_model = HebEMO()


def analyze_sentiment_3(input_file, output_file, sentiment_analysis, n):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Drop the first column
    df = df.drop(columns=['No.'])

    # Initialize lists to store sentiment scores
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
        negative_score = sentiment_result[0][0]['score']
        positive_score = sentiment_result[0][1]['score']
        # Append scores to the lists
        negative_scores.append(negative_score)
        positive_scores.append(positive_score)

    # Determine the dominant sentiment for each row
    sentiments = []
    for i in range(min(n, len(df))):
        pos_score = positive_scores[i]
        neg_score = negative_scores[i]
        if pos_score > neg_score:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        sentiments.append(sentiment)

    # Create a DataFrame with 'file_id', 'content', sentiment scores, and sentiment
    result_df = pd.DataFrame({
        'file_id': df['file_id'].head(n),
        'content': df['content'].head(n),
        'Pos_score': positive_scores,
        'Neg_score': negative_scores,
        'Sentiment': sentiments
    })

    # Save the DataFrame to a new Excel file
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


def create_sentiment_analysis():
    tokenizer = AutoTokenizer.from_pretrained("avichr/hebEMO_trust")
    model = AutoModel.from_pretrained("avichr/hebEMO_trust")
    sentiment_analysis = pipeline(
        "text-classification",
        model="avichr/hebEMO_trust",
        tokenizer="avichr/hebEMO_trust",
        return_all_scores=True
    )
    return sentiment_analysis


def get_scores_from_hebEmo(content):
    hebEmotion_df = HebEMO_model.hebemo(text=content)

    pos_scores = [hebEmotion_df['anticipation'].values[0], hebEmotion_df['joy'].values[0],
                  hebEmotion_df['trust'].values[0],
                  hebEmotion_df['surprise'].values[0]]
    neg_scores = [hebEmotion_df['fear'].values[0], hebEmotion_df['anger'].values[0], hebEmotion_df['sadness'].values[0],
                  hebEmotion_df['disgust'].values[0]]

    pos = sum(pos_scores) / len(pos_scores)
    neg = sum(neg_scores) / len(neg_scores)

    relative_pos_score = pos / (pos + neg)
    relative_neg_score = neg / (pos + neg)

    return relative_pos_score, relative_neg_score


def analayze_with_hebEMO(input_file, output_file, n):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Drop the first column
    df = df.drop(columns=['No.'])

    # Initialize lists to store sentiment scores
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
        positive_score, negative_score = get_scores_from_hebEmo(content)
        # Append scores to the lists
        negative_scores.append(negative_score)
        positive_scores.append(positive_score)

    # Determine the dominant sentiment for each row
    sentiments = []
    for i in range(min(n, len(df))):
        pos_score = positive_scores[i]
        neg_score = negative_scores[i]
        if pos_score > neg_score:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        sentiments.append(sentiment)

    # Create a DataFrame with 'file_id', 'content', sentiment scores, and sentiment
    result_df = pd.DataFrame({
        'file_id': df['file_id'].head(n),
        'content': df['content'].head(n),
        'Pos_score': positive_scores,
        'Neg_score': negative_scores,
        'Sentiment': sentiments
    })

    # Save the DataFrame to a new Excel file
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


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
    # sentiment_analysis = create_sentiment_analysis()

    analayze_with_hebEMO("./data/15000_docs/original_A.xlsx",
                        "./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_A.xlsx", 5000)
    analayze_with_hebEMO("./data/15000_docs/original_B.xlsx",
                        "./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_B.xlsx", 5000)
    analayze_with_hebEMO("./data/15000_docs/original_C.xlsx",
                        "./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_C.xlsx", 5000)

    count_sentiments("./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_A.xlsx",
                     "./output/tagged_hebEMO_15000_docs/result_hebEMO_A.json")
    count_sentiments("./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_B.xlsx",
                     "./output/tagged_hebEMO_15000_docs/result_hebEMO_B.json")
    count_sentiments("./output/tagged_hebEMO_15000_docs/tagged_with_hebEMO_C.xlsx",
                     "./output/tagged_hebEMO_15000_docs/result_hebEMO_C.json")

    analayze_with_hebEMO("./data/cleaned_from_names_15000_docs/without_names_A.xlsx",
                        "./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_A.xlsx", 5000)
    analayze_with_hebEMO("./data/cleaned_from_names_15000_docs/without_names_B.xlsx",
                        "./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_B.xlsx", 5000)
    analayze_with_hebEMO("./data/cleaned_from_names_15000_docs/without_names_C.xlsx",
                        "./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_C.xlsx", 5000)

    count_sentiments("./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_A.xlsx",
                     "./output/tagged_hebEMO_15000_without_names_docs/result_hebEMO_without_names_A.json")
    count_sentiments("./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_B.xlsx",
                     "./output/tagged_hebEMO_15000_without_names_docs/result_hebEMO_without_names_B.json")
    count_sentiments("./output/tagged_hebEMO_15000_without_names_docs/tagged_hebEMO_without_names_C.xlsx",
                     "./output/tagged_hebEMO_15000_without_names_docs/result_hebEMO_without_names_C.json")


if __name__ == "__main__":
    main()
