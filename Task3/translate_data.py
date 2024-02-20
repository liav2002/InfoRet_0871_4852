import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
from googletrans import Translator
import math


def translate_and_split_content(input_file, output_file):
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)

    # Initialize translator
    translator = Translator()

    # Function to split content into chunks of size 'chunk_size' (in words)
    def split_text(text, chunk_size=500):
        chunks = []
        words = text.split()
        num_chunks = math.ceil(len(words) / chunk_size)
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            chunk = ' '.join(words[start_index:end_index])
            chunks.append(chunk)
        return chunks

    # Translate and split the content column
    translated_content = []
    for content in df['content']:
        chunks = split_text(content)
        translated_chunks = [translator.translate(chunk).text for chunk in chunks]
        translated_content.append(' '.join(translated_chunks))

    # Update the DataFrame with translated content
    df['content'] = translated_content

    # Save the DataFrame to a new Excel file
    df.to_excel(output_file, index=False)


def main():
    # pip install googletrans==3.1.0a0
    # !pip install vaderSentiment
    translate_and_split_content("./data/15000_docs/original_A.xlsx", "./data/translated_15000_docs/translated_A_new.xlsx")
    translate_and_split_content("./data/15000_docs/original_B.xlsx", "./data/translated_15000_docs/translated_B_new.xlsx")
    translate_and_split_content("./data/15000_docs/original_C.xlsx", "./data/translated_15000_docs/translated_C_new.xlsx")

if __name__ == "__main__":
    main()
