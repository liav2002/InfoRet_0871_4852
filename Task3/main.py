import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def process_excel_file(file_path):
    df = pd.read_excel(file_path)
    df['Sentiment'] = df['content'].apply(analyze_sentiment)

    positive_count = df[df['Sentiment'] == 'Positive'].shape[0]
    negative_count = df[df['Sentiment'] == 'Negative'].shape[0]

    group_id = file_path.split('_')[-1][0]

    df.to_excel(f"tagged_translated_{group_id}.xlsx", index=False)

    return group_id, [positive_count, negative_count]


def main():
    result = {'A': [0, 0], 'B': [0, 0], 'C': [0, 0]}
    file_paths = [
        "./data/translated_15000_docs/translated_A.xlsx",
        "./data/translated_15000_docs/translated_B.xlsx",
        "./data/translated_15000_docs/translated_C.xlsx"
    ]

    for file_path in file_paths:
        group_id, sentiment_counts = process_excel_file(file_path)
        result[group_id][0] += sentiment_counts[0]
        result[group_id][1] += sentiment_counts[1]

    print("Result:", result)


if __name__ == "__main__":
    main()
